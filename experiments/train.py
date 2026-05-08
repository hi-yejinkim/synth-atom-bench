"""Training loop for flow matching velocity networks."""

import math
import os
import sys

# Force unbuffered stdout/stderr so logs appear immediately when redirected to file
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Monkey-patch argparse for Python 3.14 compatibility with Hydra's LazyCompletionHelp
import argparse

_orig_add_argument = argparse.ArgumentParser.add_argument


def _patched_add_argument(self, *args, **kwargs):
    help_val = kwargs.get("help")
    if help_val is not None and not isinstance(help_val, str):
        kwargs["help"] = repr(help_val)
    return _orig_add_argument(self, *args, **kwargs)


argparse.ArgumentParser.add_argument = _patched_add_argument

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from data.chain_dataset import ChainDataset
from data.dataset import HardSphereDataset
from data.nbody_dataset import NBodyDataset
from data.vsepr_dataset import VSEPRDataset
from data.sequence_dataset import SequenceDataset
from data.unified_dataset import UnifiedDataset
from experiments.checkpointing import CheckpointManager
from experiments.logger import ComputeTracker, Logger, LoggerConfig
from flow_matching.sampling import sample_batched
from flow_matching.training import flow_matching_loss
from data.validate import pair_correlation
from metrics.bond_violation import bond_violation_rate_batched, nonbonded_clash_rate_batched
from metrics.clash_rate import clash_rate_batched
from metrics.gr_distance import gr_distance
from metrics.wasserstein_distance import energy_w2_batched
from experiments.model_registry import MODEL_REGISTRY, SIZE_PRESETS
from experiments.task_registry import get_violation_rate, infer_task_id
from metrics.wasserstein_distance import _w1_1d


def is_chain_config(cfg: DictConfig) -> bool:
    """Check if config describes a chain dataset (has bond_length but not n_fragments)."""
    return hasattr(cfg.data, "bond_length") and not hasattr(cfg.data, "n_fragments")


def is_vsepr_config(cfg: DictConfig) -> bool:
    """Check if config describes a single-center VSEPR dataset.

    VSEPR chains have both orbital_type AND bond_length; they are treated as
    chain configs (loaded by ChainDataset), not as single-center VSEPR.
    """
    return hasattr(cfg.data, "orbital_type") and not hasattr(cfg.data, "bond_length")


def is_sequence_config(cfg: DictConfig) -> bool:
    """Check if config describes a sequence/polymer dataset."""
    return hasattr(cfg.data, "n_fragments")


def is_unified_config(cfg: DictConfig) -> bool:
    """Check if config describes a unified 6-rule structured task."""
    return hasattr(cfg.data, "unified_structure") and cfg.data.unified_structure


def is_nbody_config(cfg: DictConfig) -> bool:
    """Check if config describes an n-body energy distribution task."""
    return hasattr(cfg.data, "nbody") and cfg.data.nbody


def is_nbody_chain_config(cfg: DictConfig) -> bool:
    """Check if config describes an n-body chain energy distribution task."""
    return hasattr(cfg.data, "nbody_chain") and cfg.data.nbody_chain


def _any_nbody(cfg: DictConfig) -> bool:
    """True for any energy-distribution task (nbody or nbody_chain)."""
    return is_nbody_config(cfg) or is_nbody_chain_config(cfg)


def _primary_energy_key(cfg: DictConfig) -> str:
    """Primary energy metric: W2 for nbody_chain (matches lower bound), W1 for nbody."""
    return "energy_w2" if is_nbody_chain_config(cfg) else "energy_w1"


def load_dataset(cfg: DictConfig, path: str, max_samples: int | None = None):
    """Load the appropriate dataset class based on config."""
    if is_unified_config(cfg):
        return UnifiedDataset(path, max_samples=max_samples)
    if is_nbody_chain_config(cfg) or is_nbody_config(cfg):
        return NBodyDataset(path, max_samples=max_samples)
    if is_vsepr_config(cfg):
        return VSEPRDataset(path, max_samples=max_samples)
    if is_sequence_config(cfg):
        return SequenceDataset(path, max_samples=max_samples)
    if is_chain_config(cfg):
        return ChainDataset(path, max_samples=max_samples)
    return HardSphereDataset(path, max_samples=max_samples)


def random_rotation_matrix(device: torch.device) -> torch.Tensor:
    """Sample a uniform random SO(3) rotation via QR decomposition."""
    z = torch.randn(3, 3)  # CPU: avoids cuSolver handle conflict when multiple GPU processes launch simultaneously
    q, r = torch.linalg.qr(z)
    # Fix sign to ensure proper rotation (det=+1)
    d = torch.diag(r.sign().diag())
    q = q @ d
    if q.det() < 0:
        q[:, 0] = -q[:, 0]
    return q.to(device)


def count_flops(
    model: nn.Module, n_atoms: int, batch_size: int, device: torch.device,
    atom_type_ids: torch.Tensor | None = None,
) -> int:
    """Estimate FLOPs for one forward+backward pass."""
    n_params = sum(p.numel() for p in model.parameters())
    # Use 6 * params * batch_size as estimate (2x forward + 4x backward)
    # Try torch FlopCounterMode if available
    try:
        from torch.utils.flop_counter import FlopCounterMode

        x = torch.randn(batch_size, n_atoms, 3, device=device)
        t = torch.rand(batch_size, device=device)
        with FlopCounterMode(display=False) as counter:
            out = model(x, t, atom_type_ids=atom_type_ids)
            loss = out.sum()
        forward_flops = counter.get_total_flops()
        # backward is ~2x forward
        return int(forward_flops * 3)
    except (ImportError, Exception):
        return 6 * n_params * batch_size


def build_model(cfg: DictConfig, box_size: float) -> nn.Module:
    """Instantiate velocity network from config."""
    arch = cfg.model.arch
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    kwargs = dict(cfg.model.model_kwargs)
    # Override cutoff to match data
    cutoff_key = "cutoff"
    if cutoff_key in kwargs:
        kwargs[cutoff_key] = box_size * 1.5
    # Chain tasks: enable sequential position bias so the model knows atom order
    if arch == "transformer" and is_nbody_chain_config(cfg):
        kwargs["use_chain_pe"] = True
    return MODEL_REGISTRY[arch](**kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with optional linear warmup and configurable minimum LR ratio.

    Chinchilla mode (warmup_fraction=0):
      - T_max = max_steps exactly (Chinchilla requirement).
      - LR decays from initial → min_lr_ratio * initial at the final step.
      - No warmup.

    Standard mode:
      - Linear warmup for warmup_fraction * max_steps steps.
      - Cosine decay from warmup peak → min_lr_ratio * initial at max_steps.
    """
    max_steps = cfg.train.max_steps
    warmup_steps = int(cfg.train.get("warmup_fraction", 0.05) * max_steps)
    min_lr_ratio = float(cfg.train.get("min_lr_ratio", 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        # Cosine from 1.0 down to min_lr_ratio
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: nn.Module,
    dataset,
    cfg: DictConfig,
    device: torch.device,
    gt_r: "np.ndarray | None" = None,
    gt_g_r: "np.ndarray | None" = None,
    gt_rg_mean: "float | None" = None,
    gt_rg_std: "float | None" = None,
) -> dict:
    """Generate samples and compute metrics.

    Returns dict with keys: clash_rate, gr_distance, samples,
    and task-specific extras (chain/vsepr/sequence).
    """
    import numpy as np

    model.eval()
    # Pass atom_type_ids conditioning if available (unified datasets)
    atom_type_ids = getattr(dataset, "atom_type_ids", None)
    if atom_type_ids is not None:
        atom_type_ids = atom_type_ids.to(device)
    samples = sample_batched(
        model,
        n_atoms=dataset.positions.shape[1],
        n_samples=cfg.eval.n_samples,
        n_steps=cfg.eval.n_ode_steps,
        batch_size=cfg.eval.sample_batch_size,
        device=str(device),
        atom_type_ids=atom_type_ids,
    )
    # Shift back to [0, box_size] (unified tasks are already origin-centered;
    # their metrics are translation-invariant so we skip the shift)
    if not is_unified_config(cfg):
        samples = samples + dataset.box_size / 2
    cr = clash_rate_batched(samples, dataset.radius)
    grd = float("inf")
    if gt_r is not None and gt_g_r is not None:
        grd = gr_distance(samples.numpy(), gt_r, gt_g_r, dataset.box_size)

    result = {"clash_rate": cr, "gr_distance": grd, "samples": samples}

    # n-body chain structural W2 metrics.
    # Potentials active per body order:
    #   body=2: bond-spring + LJ(non-bonded)
    #   body=3: + angle
    #   body=4: + dihedral
    # We measure W2 on the structural distributions that correspond to each
    # potential term.  Raw energy W2 is avoided: the (σ/r)^12 LJ term diverges
    # for non-bonded atom clashes, inflating W2 to 10^28 in early training.
    if is_nbody_chain_config(cfg):
        from metrics.wasserstein_distance import _w2_1d

        pos_np = samples.numpy()             # (n_eval, N, 3) — shifted to [0, box_size]
        ref_pos = dataset.positions.numpy()  # (N_train, N, 3) — shifted to [-box/2, +box/2]
        # Cap reference to 50 K for speed; structural W2 is stable at this scale
        if len(ref_pos) > 50_000:
            rng_ref = np.random.default_rng(0).choice(len(ref_pos), 50_000, replace=False)
            ref_pos = ref_pos[rng_ref]

        # --- body ≥ 2: bond-length W2 (structural diagnostic) ---
        gen_bl = np.linalg.norm(pos_np[:, 1:] - pos_np[:, :-1], axis=-1).ravel().astype(np.float64)
        ref_bl = np.linalg.norm(ref_pos[:, 1:] - ref_pos[:, :-1], axis=-1).ravel().astype(np.float64)
        w2_bl = _w2_1d(gen_bl, ref_bl)
        result["W2_bond_len"] = w2_bl
        result["ref_energy_std"] = float(dataset.energies.std())

        # --- body ≥ 3: bond-angle W2 (captures angle term) ---
        if dataset.body >= 3:
            def _bond_angles(pos: np.ndarray) -> np.ndarray:
                v1 = pos[:, :-2] - pos[:, 1:-1]
                v2 = pos[:, 2:]  - pos[:, 1:-1]
                cos = (v1 * v2).sum(-1) / (
                    np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-12
                )
                return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).ravel()
            result["W2_bond_angle"] = _w2_1d(
                _bond_angles(pos_np).astype(np.float64),
                _bond_angles(ref_pos).astype(np.float64),
            )

        # --- body ≥ 4: dihedral W2 (captures dihedral term) ---
        if dataset.body >= 4:
            def _dihedral_angles(pos: np.ndarray) -> np.ndarray:
                p0, p1, p2, p3 = pos[:, :-3], pos[:, 1:-2], pos[:, 2:-1], pos[:, 3:]
                b0 = p1 - p0; b1 = p2 - p1; b2 = p3 - p2
                n1 = np.cross(b0, b1)
                n2 = np.cross(b1, b2)
                n1 = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-12)
                n2 = n2 / (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-12)
                cos = np.clip((n1 * n2).sum(-1), -1.0, 1.0)
                b1u = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12)
                sin = (np.cross(n1, n2) * b1u).sum(-1)
                return np.degrees(np.arctan2(sin, cos)).ravel()
            result["W2_dihedral"] = _w2_1d(
                _dihedral_angles(pos_np).astype(np.float64),
                _dihedral_angles(ref_pos).astype(np.float64),
            )

        # --- Full chain energy W2 (primary Chinchilla metric) ---
        # float64 + clamp to ref_mean+20σ prevents LJ (σ/r)^12 overflow for
        # clashing structures. Covers all active terms: bond + LJ_nb + angle + dihedral.
        from data.generate_nbody_chain import ChainParams, total_energy as chain_total_energy
        chain_params = ChainParams(
            body=dataset.body,
            N=dataset.n_atoms_chain,
            k2=dataset.k2, r0=dataset.r0,
            sigma=dataset.sigma, epsilon_lj=dataset.epsilon,
            k3=dataset.k3, theta0=dataset.theta0,
            c1=dataset.c1, c2=dataset.c2,
            box_size=dataset.box_size,
        )
        ref_e = dataset.energies.numpy().astype(np.float64)
        ref_e_mean, ref_e_std = float(ref_e.mean()), float(ref_e.std())
        e_clip_hi = ref_e_mean + 20.0 * ref_e_std
        gen_e_f64 = np.array(
            [chain_total_energy(pos_np[i], chain_params)[0] for i in range(len(pos_np))],
            dtype=np.float64,
        )
        gen_e_clipped = np.clip(gen_e_f64, None, e_clip_hi)
        ref_e_clipped = np.clip(ref_e, None, e_clip_hi)
        result["chain_energy_w2"] = _w2_1d(gen_e_clipped, ref_e_clipped)
        result["chain_energy_w1"] = _w1_1d(gen_e_clipped, ref_e_clipped)
        result["energy_w2"] = result["chain_energy_w2"]
        result["energy_w1"] = result["chain_energy_w1"]

    # n-body energy W1/W2 metrics (spherical LJ potential)
    elif is_nbody_config(cfg):
        from metrics.wasserstein_distance import energy_w2_from_positions
        wd = energy_w2_from_positions(
            samples.numpy(),
            dataset.energies.numpy(),
            body=dataset.body,
            sigma=dataset.sigma,
            epsilon=dataset.epsilon,
            nu=dataset.nu,
            mu=dataset.mu,
            box_size=dataset.box_size,
            bc=dataset.bc,
        )
        result["energy_w1"] = wd["w1_total"]
        result["energy_w2"] = wd["w2_total"]
        result["ref_energy_std"] = float(dataset.energies.std())

    # Chain-specific metrics
    if is_chain_config(cfg):
        result["bond_violation_rate"] = bond_violation_rate_batched(samples, dataset.bond_length)
        result["nonbonded_clash_rate"] = nonbonded_clash_rate_batched(samples, dataset.radius)

    # VSEPR-specific metrics
    if is_vsepr_config(cfg):
        from metrics.vsepr_metrics import (
            angle_distribution_jsd,
            bond_length_in_peak_ratio_batched,
            torsional_out_of_bin_rate,
            valence_overcoordination_rate_batched,
        )
        from data.generate_vsepr import get_angle_sigma_deg
        angle_sigma = get_angle_sigma_deg(dataset.n_lonepairs)
        result["bond_length_in_peak_ratio"] = bond_length_in_peak_ratio_batched(
            samples, dataset.bond_range)
        result["angle_jsd"] = angle_distribution_jsd(
            samples.numpy(), dataset.target_angle, angle_sigma)
        result["torsional_out_of_bin_rate"] = torsional_out_of_bin_rate(
            samples.numpy(), dataset.has_pi)
        result["valence_overcoord_rate"] = valence_overcoordination_rate_batched(
            samples, dataset.bond_range)

    # Sequence-specific metrics
    if is_sequence_config(cfg):
        from metrics.sequence_metrics import (
            long_range_contact_recall_batched,
            sequence_bond_violation_rate_batched,
            radius_of_gyration_error_batched,
        )
        from data.generate_sequence import _build_linear_bonds, _build_branched_bonds, _build_crosslinked_bonds
        N = samples.shape[1]
        ptype = dataset.polymer_type
        if ptype == "linear":
            bond_list = _build_linear_bonds(N)
        elif ptype == "branched":
            bond_list, _ = _build_branched_bonds(N)
        else:
            bond_list, _ = _build_crosslinked_bonds(N)

        result["contact_recall"] = long_range_contact_recall_batched(
            samples, dataset.contact_pairs, dataset.contact_distance)
        result["seq_bond_violation_rate"] = sequence_bond_violation_rate_batched(
            samples, bond_list, dataset.bond_length)
        if gt_rg_mean is not None and gt_rg_std is not None:
            result["rg_error"] = radius_of_gyration_error_batched(
                samples, gt_rg_mean, gt_rg_std)

    # Unified 6-rule metrics
    if is_unified_config(cfg):
        from metrics.unified_metrics import unified_violation_rate
        unified_results = unified_violation_rate(samples, dataset.npz_meta)
        result.update(unified_results)

    model.train()
    return result


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    # Resolve size preset into model_kwargs
    size = cfg.model.get("size")
    if size and cfg.model.arch in SIZE_PRESETS:
        preset = SIZE_PRESETS[cfg.model.arch][size]
        with open_dict(cfg):
            for k, v in preset.items():
                cfg.model.model_kwargs[k] = v

    # Chinchilla trajectory logging setup
    _chinchilla_cfg = cfg.get("chinchilla", None)
    _traj_file = None
    _traj_task_id = None
    _traj_D_nominal: int | None = None
    if _chinchilla_cfg and _chinchilla_cfg.get("enabled", False):
        _traj_task_id = _chinchilla_cfg.get("task_id") or None
        _traj_path = _chinchilla_cfg.get("trajectory_path") or None
        _traj_D_nominal = _chinchilla_cfg.get("D_nominal") or None
        if _traj_D_nominal is not None:
            _traj_D_nominal = int(_traj_D_nominal)
        if _traj_path:
            import json as _json
            os.makedirs(os.path.dirname(_traj_path), exist_ok=True)
            # Open in write mode ("w") so a fresh run always starts clean.
            # Resuming from checkpoint will re-evaluate and re-log from start_step.
            _traj_file = open(_traj_path, "w")

    # Seed
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    # GPU assignment: distribute multirun jobs across GPUs
    if torch.cuda.is_available():
        try:
            from hydra.core.hydra_config import HydraConfig
            job_num = HydraConfig.get().job.num
        except Exception:
            job_num = 0
        gpu_id = job_num % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load dataset
    data_dir = cfg.data.data_dir
    train_path = os.path.join(data_dir, "train.npz")
    max_train_samples = cfg.train.get("max_train_samples") or None
    dataset = load_dataset(cfg, train_path, max_samples=max_train_samples)
    box_size = dataset.box_size
    n_atoms = dataset.positions.shape[1]
    print(f"Dataset: {len(dataset)} samples, N={n_atoms}, box_size={box_size:.4f}")

    # Precompute ground-truth g(r) for evaluation metric
    print("Precomputing ground-truth g(r)...")
    gt_r, gt_g_r = pair_correlation(dataset.positions.numpy(), box_size)

    # Precompute ground-truth Rg stats for sequence task
    gt_rg_mean, gt_rg_std = None, None
    if is_sequence_config(cfg):
        from metrics.sequence_metrics import compute_gt_rg_stats
        gt_rg_mean, gt_rg_std = compute_gt_rg_stats(dataset.positions.numpy())
        print(f"Ground-truth Rg: mean={gt_rg_mean:.3f}, std={gt_rg_std:.3f}")

    # Center positions for flow matching (noise is N(0,I))
    # Unified tasks are already origin-centered by the MCMC generator; skip the shift.
    if not is_unified_config(cfg):
        dataset.positions = dataset.positions - box_size / 2

    # Gradient accumulation setup
    grad_accum_steps = int(cfg.train.get("grad_accum_steps", 1))
    micro_batch_size = cfg.train.batch_size // grad_accum_steps
    effective_batch_size = micro_batch_size * grad_accum_steps  # = cfg.train.batch_size
    if grad_accum_steps > 1:
        print(f"Gradient accumulation: {grad_accum_steps} steps × {micro_batch_size} micro-batch = {effective_batch_size} effective batch")

    # DataLoader (uses micro_batch_size when grad_accum_steps > 1)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Build model
    model = build_model(cfg, box_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {cfg.model.arch} | Parameters: {n_params:,}")

    # Per-atom conditioning for unified datasets
    dataset_atom_type_ids = getattr(dataset, "atom_type_ids", None)
    if dataset_atom_type_ids is not None:
        dataset_atom_type_ids = dataset_atom_type_ids.to(device)

    # FLOPs estimation (per optimizer step = effective_batch_size samples)
    flops_per_step = count_flops(model, n_atoms, effective_batch_size, device,
                                 atom_type_ids=dataset_atom_type_ids)
    print(f"FLOPs per step: {flops_per_step:.2e}")

    # Budget mode: compute max_steps from budget / flops_per_step
    budget = cfg.train.get("budget")
    if budget is not None and float(budget) > 0:
        budget = float(budget)
        computed_steps = int(budget / flops_per_step)
        if computed_steps < 2000:
            print(f"Budget {budget:.0e}: only {computed_steps} steps (< 2000 min). Skipping.")
            return
        if computed_steps > 1_000_000:
            print(f"Budget {budget:.0e}: needs {computed_steps} steps (> 1M max). Skipping.")
            return
        with open_dict(cfg):
            cfg.train.max_steps = computed_steps
        print(f"Budget {budget:.0e}: training for {computed_steps} steps")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)

    # Checkpoint dir
    checkpoint_dir = cfg.checkpoint.get("dir") or os.path.join("outputs", "checkpoints", cfg.model.arch)
    primary_metric = _primary_energy_key(cfg) if _any_nbody(cfg) else "gr_distance"
    ckpt_mgr = CheckpointManager(checkpoint_dir, primary_metric=primary_metric)

    # Resume from checkpoint if available
    start_step = 0
    state = ckpt_mgr.load_latest(device=str(device))
    if state is not None:
        model.load_state_dict(state.model_state_dict)
        optimizer.load_state_dict(state.optimizer_state_dict)
        start_step = state.step
        # Fast-forward scheduler (suppress expected warning about ordering)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_step):
                scheduler.step()
        print(f"Resumed from step {start_step}")

    # Logger
    logger_config = LoggerConfig(
        project=cfg.logging.project,
        entity=cfg.logging.get("entity"),
        enabled=cfg.logging.enabled,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )
    if is_unified_config(cfg):
        rules_str = cfg.data.get("rules_str", "unified")
        run_name = f"{cfg.model.arch}_N{n_atoms}_unified_{rules_str}"
    elif is_vsepr_config(cfg):
        run_name = f"{cfg.model.arch}_N{n_atoms}_vsepr_{cfg.data.orbital_type}"
    elif is_sequence_config(cfg):
        run_name = f"{cfg.model.arch}_N{n_atoms}_seq_{cfg.data.polymer_type}"
    elif is_nbody_chain_config(cfg):
        run_name = f"{cfg.model.arch}_N{n_atoms}_nbody_chain_b{cfg.data.body}_T{cfg.data.T}"
    elif is_chain_config(cfg):
        run_name = f"{cfg.model.arch}_N{n_atoms}_chain"
    elif is_nbody_config(cfg):
        run_name = f"{cfg.model.arch}_N{n_atoms}_nbody_b{cfg.data.body}_T{cfg.data.T}"
    else:
        run_name = f"{cfg.model.arch}_N{n_atoms}_eta{cfg.data.eta}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = Logger(logger_config, run_name=run_name, model_config=config_dict)
    logger.log_model_config(cfg.model.arch, n_params, flops_per_step)

    # Compute tracker
    tracker = ComputeTracker()

    # N-body loss enhancements: SNR weighting + auxiliary pairwise distance loss
    _nbody_loss = _any_nbody(cfg)
    _aux_dist_w = 0.3 if _nbody_loss else 0.0
    if _nbody_loss:
        print("N-body loss: SNR weighting + auxiliary distance loss (0.3)")

    # Training loop
    model.train()
    step = start_step
    data_iter = iter(loader)
    use_rotation = cfg.augmentation.random_rotation
    print(f"\nTraining for {cfg.train.max_steps} steps (starting from {start_step})...")

    while step < cfg.train.max_steps:
        tracker.start()
        optimizer.zero_grad()
        accum_loss = 0.0

        for _accum_i in range(grad_accum_steps):
            # Get next micro-batch, cycling through dataset
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x_0 = batch["positions"].to(device)

            # Per-atom conditioning (unified datasets provide atom_type_ids)
            atom_type_ids = batch.get("atom_type_ids")
            if atom_type_ids is not None:
                atom_type_ids = atom_type_ids[0].to(device)  # (N,) — same for all samples

            # Random SO(3) augmentation
            if use_rotation:
                R = random_rotation_matrix(device)
                x_0 = x_0 @ R.T

            # Forward + backward (scale loss by accum steps for correct gradient magnitude)
            loss = flow_matching_loss(
                model, x_0, atom_type_ids=atom_type_ids,
                snr_weight=_nbody_loss, aux_dist_weight=_aux_dist_w,
            ) / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        # Clip after full accumulation, then step
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        tracker.stop()
        step += 1
        loss = accum_loss  # for logging

        # Log training metrics
        if step % logger_config.log_every_n_steps == 0:
            total_flops = flops_per_step * step
            lr = scheduler.get_last_lr()[0]
            logger.log_train(
                {"train/loss": loss, "train/lr": lr, "train/total_flops": total_flops},
                step=step,
            )
            logger.log_compute(tracker, step)
            print(f"  Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e} | FLOPs: {total_flops:.2e}")

        # Evaluate + checkpoint
        if step % cfg.eval.every_n_steps == 0:
            ev = evaluate(model, dataset, cfg, device, gt_r, gt_g_r, gt_rg_mean, gt_rg_std)
            cr, grd, samples = ev["clash_rate"], ev["gr_distance"], ev["samples"]
            total_flops = flops_per_step * step
            # Chinchilla trajectory logging
            if _traj_file is not None and _traj_task_id is not None:
                import json as _json
                try:
                    vr = get_violation_rate(ev, _traj_task_id)
                except Exception as _e:
                    # Fail loudly for KeyError — a missing metric key means the wrong
                    # metric would be silently substituted, corrupting the scaling fit.
                    if isinstance(_e, KeyError):
                        raise
                    vr = float(cr)
                traj_record = {
                    "step": step,
                    "max_steps": cfg.train.max_steps,
                    "total_flops": float(total_flops),
                    "D_seen": int(step * effective_batch_size),
                    # D_nominal = intended unique training samples (epoch-independent).
                    # Use this for scaling fits; D_seen inflates when epochs > 1.
                    "D_nominal": _traj_D_nominal if _traj_D_nominal is not None else int(step * effective_batch_size),
                    "violation_rate": float(vr),
                    "n_params": n_params,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "task": _traj_task_id,
                    "arch": cfg.model.arch,
                    "size": cfg.model.get("size", "unknown"),
                }
                # Per-rule violation rates for cross-rule Chinchilla comparisons
                for rk in ["clash_violation_rate", "slot_violation_rate",
                            "bond_angle_violation_rate",
                            "bb_bond_length_violation_rate", "sc_bond_length_violation_rate",
                            "pi_planarity_violation_rate", "contact_recall",
                            "periodicity_violation_rate",
                            "energy_w1", "energy_w2", "ref_energy_std",
                            "W2_bond_len", "W2_bond_angle", "W2_dihedral",
                            "chain_energy_w1", "chain_energy_w2",
                            "gr_distance"]:
                    if rk in ev:
                        traj_record[rk] = float(ev[rk])
                _traj_file.write(_json.dumps(traj_record) + "\n")
                _traj_file.flush()
            logger.log_eval(samples, dataset.radius, dataset.box_size, step)
            log_metrics = {"eval/clash_rate": cr, "eval/gr_distance": grd, "train/total_flops": total_flops}
            ckpt_kwargs = dict(gr_distance=grd)
            if is_chain_config(cfg):
                bvr = ev["bond_violation_rate"]
                ncr = ev["nonbonded_clash_rate"]
                log_metrics["eval/bond_violation_rate"] = bvr
                log_metrics["eval/nonbonded_clash_rate"] = ncr
                ckpt_kwargs["bond_violation_rate"] = bvr
                ckpt_kwargs["nonbonded_clash_rate"] = ncr
            if is_vsepr_config(cfg):
                log_metrics["eval/angle_jsd"] = ev["angle_jsd"]
                log_metrics["eval/bond_length_in_peak_ratio"] = ev["bond_length_in_peak_ratio"]
                log_metrics["eval/torsional_out_of_bin_rate"] = ev["torsional_out_of_bin_rate"]
                log_metrics["eval/valence_overcoord_rate"] = ev["valence_overcoord_rate"]
                ckpt_kwargs["angle_jsd"] = ev["angle_jsd"]
                ckpt_kwargs["bond_length_in_peak_ratio"] = ev["bond_length_in_peak_ratio"]
                ckpt_kwargs["torsional_out_of_bin_rate"] = ev["torsional_out_of_bin_rate"]
                ckpt_kwargs["valence_overcoord_rate"] = ev["valence_overcoord_rate"]
            if is_sequence_config(cfg):
                log_metrics["eval/contact_recall"] = ev["contact_recall"]
                log_metrics["eval/seq_bond_violation_rate"] = ev["seq_bond_violation_rate"]
                ckpt_kwargs["contact_recall"] = ev["contact_recall"]
                ckpt_kwargs["seq_bond_violation_rate"] = ev["seq_bond_violation_rate"]
                if "rg_error" in ev:
                    log_metrics["eval/rg_error"] = ev["rg_error"]
                    ckpt_kwargs["rg_error"] = ev["rg_error"]
            if _any_nbody(cfg) and "energy_w2" in ev:
                log_metrics["eval/energy_w1"] = ev["energy_w1"]
                log_metrics["eval/energy_w2"] = ev["energy_w2"]
                ckpt_kwargs["energy_w1"] = ev["energy_w1"]
                ckpt_kwargs["energy_w2"] = ev["energy_w2"]
            if is_unified_config(cfg):
                vr = ev.get("violation_rate", 0.0)
                log_metrics["eval/violation_rate"] = vr
                ckpt_kwargs["violation_rate"] = vr
                for mk in ["clash_violation_rate", "slot_violation_rate",
                            "bond_angle_violation_rate",
                            "bb_bond_length_violation_rate", "sc_bond_length_violation_rate",
                            "pi_planarity_violation_rate", "contact_recall",
                            "periodicity_violation_rate"]:
                    if mk in ev:
                        log_metrics[f"eval/{mk}"] = ev[mk]
                        ckpt_kwargs[mk] = ev[mk]
            logger.log_train(log_metrics, step=step)
            ckpt_mgr.save(model, optimizer, epoch=0, step=step, clash_rate=cr, config=config_dict, **ckpt_kwargs)
            msg = f"  Step {step:6d} | Eval clash rate: {cr:.4f} | g(r) dist: {grd:.4f}"
            if is_chain_config(cfg):
                msg += f" | bond viol: {ev['bond_violation_rate']:.4f} | nb clash: {ev['nonbonded_clash_rate']:.4f}"
            if is_vsepr_config(cfg):
                msg += f" | angle JSD: {ev['angle_jsd']:.4f} | in-peak: {ev['bond_length_in_peak_ratio']:.4f}"
            if is_sequence_config(cfg):
                msg += f" | contact recall: {ev['contact_recall']:.4f} | bond viol: {ev['seq_bond_violation_rate']:.4f}"
            if _any_nbody(cfg) and "energy_w1" in ev:
                e_label = "W2" if is_nbody_chain_config(cfg) else "W1"
                best_e_attr = "best_energy_w2" if is_nbody_chain_config(cfg) else "best_energy_w1"
                msg += f" | energy W1: {ev['energy_w1']:.4f} W2: {ev['energy_w2']:.4f}"
                if is_nbody_chain_config(cfg) and "W2_bond_len" in ev:
                    msg += f" | W2_bond: {ev['W2_bond_len']:.4f}"
                msg += f" | Best E-{e_label}: {getattr(ckpt_mgr, best_e_attr):.4f}"
            elif is_unified_config(cfg):
                msg += f" | violation: {ev.get('violation_rate', 0.0):.4f}"
                msg += f" | Best g(r): {ckpt_mgr.best_gr_distance:.4f}"
            else:
                msg += f" | Best g(r): {ckpt_mgr.best_gr_distance:.4f}"
            print(msg)

        # Periodic checkpoint (without eval)
        elif step % cfg.checkpoint.every_n_steps == 0:
            ckpt_mgr.save(
                model, optimizer, epoch=0, step=step,
                clash_rate=ckpt_mgr.best_clash_rate, config=config_dict,
            )
            print(f"  Step {step:6d} | Checkpoint saved")

    # Final evaluation
    print("\nFinal evaluation...")
    ev = evaluate(model, dataset, cfg, device, gt_r, gt_g_r, gt_rg_mean, gt_rg_std)
    cr, grd, samples = ev["clash_rate"], ev["gr_distance"], ev["samples"]
    logger.log_eval(samples, dataset.radius, dataset.box_size, step)
    ckpt_kwargs = dict(gr_distance=grd)
    if is_chain_config(cfg):
        ckpt_kwargs["bond_violation_rate"] = ev["bond_violation_rate"]
        ckpt_kwargs["nonbonded_clash_rate"] = ev["nonbonded_clash_rate"]
    if is_vsepr_config(cfg):
        ckpt_kwargs["angle_jsd"] = ev["angle_jsd"]
        ckpt_kwargs["bond_length_in_peak_ratio"] = ev["bond_length_in_peak_ratio"]
        ckpt_kwargs["torsional_out_of_bin_rate"] = ev["torsional_out_of_bin_rate"]
        ckpt_kwargs["valence_overcoord_rate"] = ev["valence_overcoord_rate"]
    if is_sequence_config(cfg):
        ckpt_kwargs["contact_recall"] = ev["contact_recall"]
        ckpt_kwargs["seq_bond_violation_rate"] = ev["seq_bond_violation_rate"]
        if "rg_error" in ev:
            ckpt_kwargs["rg_error"] = ev["rg_error"]
    if _any_nbody(cfg) and "energy_w1" in ev:
        ckpt_kwargs["energy_w1"] = ev["energy_w1"]
        ckpt_kwargs["energy_w2"] = ev["energy_w2"]
    if is_unified_config(cfg):
        ckpt_kwargs["violation_rate"] = ev.get("violation_rate", 0.0)
        for mk in ["clash_violation_rate", "slot_violation_rate",
                    "bond_angle_violation_rate",
                    "bb_bond_length_violation_rate", "sc_bond_length_violation_rate",
                    "pi_planarity_violation_rate", "contact_recall",
                    "periodicity_violation_rate"]:
            if mk in ev:
                ckpt_kwargs[mk] = ev[mk]
    ckpt_mgr.save(model, optimizer, epoch=0, step=step, clash_rate=cr, config=config_dict, **ckpt_kwargs)
    msg = f"Final clash rate: {cr:.4f} | g(r) dist: {grd:.4f}"
    if is_chain_config(cfg):
        msg += f" | bond viol: {ev['bond_violation_rate']:.4f} | nb clash: {ev['nonbonded_clash_rate']:.4f}"
    if is_vsepr_config(cfg):
        msg += f" | angle JSD: {ev['angle_jsd']:.4f} | in-peak: {ev['bond_length_in_peak_ratio']:.4f}"
    if is_sequence_config(cfg):
        msg += f" | contact recall: {ev['contact_recall']:.4f} | bond viol: {ev['seq_bond_violation_rate']:.4f}"
    if is_unified_config(cfg):
        msg += f" | violation: {ev.get('violation_rate', 0.0):.4f}"
    if _any_nbody(cfg):
        e_label = "W2" if is_nbody_chain_config(cfg) else "W1"
        best_e_attr = "best_energy_w2" if is_nbody_chain_config(cfg) else "best_energy_w1"
        msg += f" | Best E-{e_label}: {getattr(ckpt_mgr, best_e_attr):.4f}"
    else:
        msg += f" | Best g(r): {ckpt_mgr.best_gr_distance:.4f}"
    print(msg)

    # Write final eval to trajectory if this step wasn't already logged periodically
    # (happens when max_steps is not a multiple of eval.every_n_steps)
    if (_traj_file is not None and _traj_task_id is not None
            and step % cfg.eval.every_n_steps != 0):
        import json as _json
        try:
            vr_final = get_violation_rate(ev, _traj_task_id)
        except Exception:
            vr_final = float(ev.get("clash_rate", 1.0))
        final_record = {
            "step": step,
            "max_steps": cfg.train.max_steps,
            "total_flops": float(flops_per_step * step),
            "D_seen": int(step * effective_batch_size),
            "D_nominal": _traj_D_nominal if _traj_D_nominal is not None else int(step * effective_batch_size),
            "violation_rate": float(vr_final),
            "n_params": n_params,
            "lr": float(scheduler.get_last_lr()[0]),
            "task": _traj_task_id,
            "arch": cfg.model.arch,
            "size": cfg.model.get("size", "unknown"),
        }
        for rk in ["clash_violation_rate", "slot_violation_rate",
                    "bond_angle_violation_rate",
                    "bb_bond_length_violation_rate", "sc_bond_length_violation_rate",
                    "pi_planarity_violation_rate", "contact_recall",
                    "periodicity_violation_rate",
                    "energy_w1", "energy_w2", "ref_energy_std",
                    "W2_bond_len", "W2_bond_angle", "W2_dihedral",
                    "chain_energy_w1", "chain_energy_w2",
                    "gr_distance"]:
            if rk in ev:
                final_record[rk] = float(ev[rk])
        # Only write if this step differs from the last periodic log
        _traj_file.write(_json.dumps(final_record) + "\n")
        _traj_file.flush()

    if _traj_file is not None:
        _traj_file.close()
    logger.finish()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
