"""Training loop for flow matching velocity networks."""

import math
import os
import sys

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
from experiments.model_registry import MODEL_REGISTRY, SIZE_PRESETS
from experiments.task_registry import get_violation_rate, infer_task_id


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


def load_dataset(cfg: DictConfig, path: str, max_samples: int | None = None):
    """Load the appropriate dataset class based on config."""
    if is_unified_config(cfg):
        return UnifiedDataset(path, max_samples=max_samples)
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
    if _chinchilla_cfg and _chinchilla_cfg.get("enabled", False):
        _traj_task_id = _chinchilla_cfg.get("task_id") or None
        _traj_path = _chinchilla_cfg.get("trajectory_path") or None
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
    ckpt_mgr = CheckpointManager(checkpoint_dir)

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
    elif is_chain_config(cfg):
        run_name = f"{cfg.model.arch}_N{n_atoms}_chain"
    else:
        run_name = f"{cfg.model.arch}_N{n_atoms}_eta{cfg.data.eta}"
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = Logger(logger_config, run_name=run_name, model_config=config_dict)
    logger.log_model_config(cfg.model.arch, n_params, flops_per_step)

    # Compute tracker
    tracker = ComputeTracker()

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
            loss = flow_matching_loss(model, x_0, atom_type_ids=atom_type_ids) / grad_accum_steps
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
                except (KeyError, Exception):
                    vr = float(cr)
                traj_record = {
                    "step": step,
                    "total_flops": float(total_flops),
                    "D_seen": int(step * effective_batch_size),
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
                            "periodicity_violation_rate"]:
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
            if is_unified_config(cfg):
                msg += f" | violation: {ev.get('violation_rate', 0.0):.4f}"
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
    msg += f" | Best g(r): {ckpt_mgr.best_gr_distance:.4f}"
    print(msg)

    if _traj_file is not None:
        _traj_file.close()
    logger.finish()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
