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
from experiments.checkpointing import CheckpointManager
from experiments.logger import ComputeTracker, Logger, LoggerConfig
from flow_matching.sampling import sample_batched
from flow_matching.training import flow_matching_loss
from data.validate import pair_correlation
from metrics.bond_violation import bond_violation_rate_batched, nonbonded_clash_rate_batched
from metrics.clash_rate import clash_rate_batched
from metrics.gr_distance import gr_distance
from experiments.model_registry import MODEL_REGISTRY, SIZE_PRESETS


def is_chain_config(cfg: DictConfig) -> bool:
    """Check if config describes a chain dataset (has bond_length)."""
    return hasattr(cfg.data, "bond_length")


def load_dataset(cfg: DictConfig, path: str):
    """Load ChainDataset or HardSphereDataset based on config."""
    if is_chain_config(cfg):
        return ChainDataset(path)
    return HardSphereDataset(path)


def random_rotation_matrix(device: torch.device) -> torch.Tensor:
    """Sample a uniform random SO(3) rotation via QR decomposition."""
    z = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(z)
    # Fix sign to ensure proper rotation (det=+1)
    d = torch.diag(r.sign().diag())
    q = q @ d
    if q.det() < 0:
        q[:, 0] = -q[:, 0]
    return q


def count_flops(model: nn.Module, n_atoms: int, batch_size: int, device: torch.device) -> int:
    """Estimate FLOPs for one forward+backward pass."""
    n_params = sum(p.numel() for p in model.parameters())
    # Use 6 * params * batch_size as estimate (2x forward + 4x backward)
    # Try torch FlopCounterMode if available
    try:
        from torch.utils.flop_counter import FlopCounterMode

        x = torch.randn(batch_size, n_atoms, 3, device=device)
        t = torch.rand(batch_size, device=device)
        with FlopCounterMode(display=False) as counter:
            out = model(x, t)
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
    """Cosine decay with linear warmup."""
    max_steps = cfg.train.max_steps
    warmup_steps = int(cfg.train.warmup_fraction * max_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: nn.Module,
    dataset,
    cfg: DictConfig,
    device: torch.device,
    gt_r: "np.ndarray | None" = None,
    gt_g_r: "np.ndarray | None" = None,
) -> dict:
    """Generate samples and compute metrics.

    Returns dict with keys: clash_rate, gr_distance, samples,
    and for chain data: bond_violation_rate, nonbonded_clash_rate.
    """
    import numpy as np

    model.eval()
    samples = sample_batched(
        model,
        n_atoms=dataset.positions.shape[1],
        n_samples=cfg.eval.n_samples,
        n_steps=cfg.eval.n_ode_steps,
        batch_size=cfg.eval.sample_batch_size,
        device=str(device),
    )
    # Shift back to [0, box_size]
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
    dataset = load_dataset(cfg, train_path)
    box_size = dataset.box_size
    n_atoms = dataset.positions.shape[1]
    print(f"Dataset: {len(dataset)} samples, N={n_atoms}, box_size={box_size:.4f}")

    # Precompute ground-truth g(r) for evaluation metric
    print("Precomputing ground-truth g(r)...")
    gt_r, gt_g_r = pair_correlation(dataset.positions.numpy(), box_size)

    # Center positions for flow matching (noise is N(0,I))
    dataset.positions = dataset.positions - box_size / 2

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Build model
    model = build_model(cfg, box_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {cfg.model.arch} | Parameters: {n_params:,}")

    # FLOPs estimation
    flops_per_step = count_flops(model, n_atoms, cfg.train.batch_size, device)
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
    if is_chain_config(cfg):
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
        # Get next batch, cycling through dataset
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x_0 = batch["positions"].to(device)

        # Random SO(3) augmentation
        if use_rotation:
            R = random_rotation_matrix(device)
            x_0 = x_0 @ R.T

        tracker.start()

        # Forward + backward
        loss = flow_matching_loss(model, x_0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        tracker.stop()
        step += 1

        # Log training metrics
        if step % logger_config.log_every_n_steps == 0:
            total_flops = flops_per_step * step
            lr = scheduler.get_last_lr()[0]
            logger.log_train(
                {"train/loss": loss.item(), "train/lr": lr, "train/total_flops": total_flops},
                step=step,
            )
            logger.log_compute(tracker, step)
            print(f"  Step {step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | FLOPs: {total_flops:.2e}")

        # Evaluate + checkpoint
        if step % cfg.eval.every_n_steps == 0:
            ev = evaluate(model, dataset, cfg, device, gt_r, gt_g_r)
            cr, grd, samples = ev["clash_rate"], ev["gr_distance"], ev["samples"]
            total_flops = flops_per_step * step
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
            logger.log_train(log_metrics, step=step)
            ckpt_mgr.save(model, optimizer, epoch=0, step=step, clash_rate=cr, config=config_dict, **ckpt_kwargs)
            msg = f"  Step {step:6d} | Eval clash rate: {cr:.4f} | g(r) dist: {grd:.4f}"
            if is_chain_config(cfg):
                msg += f" | bond viol: {bvr:.4f} | nb clash: {ncr:.4f}"
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
    ev = evaluate(model, dataset, cfg, device, gt_r, gt_g_r)
    cr, grd, samples = ev["clash_rate"], ev["gr_distance"], ev["samples"]
    logger.log_eval(samples, dataset.radius, dataset.box_size, step)
    ckpt_kwargs = dict(gr_distance=grd)
    if is_chain_config(cfg):
        ckpt_kwargs["bond_violation_rate"] = ev["bond_violation_rate"]
        ckpt_kwargs["nonbonded_clash_rate"] = ev["nonbonded_clash_rate"]
    ckpt_mgr.save(model, optimizer, epoch=0, step=step, clash_rate=cr, config=config_dict, **ckpt_kwargs)
    msg = f"Final clash rate: {cr:.4f} | g(r) dist: {grd:.4f}"
    if is_chain_config(cfg):
        msg += f" | bond viol: {ev['bond_violation_rate']:.4f} | nb clash: {ev['nonbonded_clash_rate']:.4f}"
    msg += f" | Best g(r): {ckpt_mgr.best_gr_distance:.4f}"
    print(msg)

    logger.finish()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
