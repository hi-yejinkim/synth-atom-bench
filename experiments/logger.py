"""W&B logger and GPU compute tracker."""

import tempfile
import time
from dataclasses import dataclass, field

import torch


@dataclass
class LoggerConfig:
    """Configuration for experiment logging."""

    project: str = "synthbench3d"
    entity: str | None = None
    enabled: bool = True
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    checkpoint_every_n_epochs: int = 5


class ComputeTracker:
    """Track elapsed GPU-hours using CUDA events or wall clock."""

    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        self._elapsed_seconds: float = 0.0
        self._running = False
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._wall_start: float = 0.0

    def start(self) -> None:
        self._running = True
        if self._use_cuda:
            self._start_event.record()
        else:
            self._wall_start = time.monotonic()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            self._elapsed_seconds += self._start_event.elapsed_time(self._end_event) / 1000.0
        else:
            self._elapsed_seconds += time.monotonic() - self._wall_start

    def reset(self) -> None:
        self._elapsed_seconds = 0.0
        self._running = False

    @property
    def gpu_hours(self) -> float:
        return self._elapsed_seconds / 3600.0


class Logger:
    """W&B experiment logger. All methods are no-ops when disabled."""

    def __init__(
        self,
        config: LoggerConfig,
        run_name: str | None = None,
        model_config: dict | None = None,
    ):
        self._config = config
        self._wandb = None
        self._scaling_table = None

        if not config.enabled:
            return

        import wandb

        self._wandb = wandb
        init_kwargs = {
            "project": config.project,
            "name": run_name,
        }
        if config.entity:
            init_kwargs["entity"] = config.entity
        if model_config:
            init_kwargs["config"] = model_config
        wandb.init(**init_kwargs)
        self._scaling_table = wandb.Table(
            columns=["architecture", "compute_budget", "best_clash_rate", "best_gr_distance", "param_count", "flops_per_step"]
        )

    def log_train(self, metrics: dict, step: int) -> None:
        if self._wandb is None:
            return
        self._wandb.log(metrics, step=step)

    def log_eval(
        self,
        positions: torch.Tensor,
        radius: float,
        box_size: float,
        step: int,
    ) -> None:
        if self._wandb is None:
            return

        import numpy as np
        from data.validate import pair_correlation
        from viz.metrics import plot_gr, plot_min_distance_hist
        from metrics.clash_rate import clash_rate_batched

        cr = clash_rate_batched(positions, radius)
        metrics: dict = {"eval/clash_rate": cr}

        # g(r) plot
        import os
        import matplotlib.pyplot as plt

        pos_np = positions.cpu().numpy()
        r, g_r = pair_correlation(pos_np, box_size)
        tmp_files = []

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            gr_path = f.name
        tmp_files.append(gr_path)
        fig = plot_gr(r, g_r, radius)
        fig.savefig(gr_path, dpi=150)
        plt.close(fig)
        metrics["eval/pair_correlation"] = self._wandb.Image(gr_path)

        # Min distance histogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            hist_path = f.name
        tmp_files.append(hist_path)
        fig = plot_min_distance_hist(pos_np, radius)
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        metrics["eval/min_distance_hist"] = self._wandb.Image(hist_path)

        self._wandb.log(metrics, step=step)

        for p in tmp_files:
            os.unlink(p)

    def log_model_config(
        self,
        architecture: str,
        param_count: int,
        flops_per_step: float | None = None,
    ) -> None:
        if self._wandb is None:
            return
        self._wandb.config.update(
            {
                "architecture": architecture,
                "param_count": param_count,
                "flops_per_step": flops_per_step,
            },
            allow_val_change=True,
        )

    def log_compute(self, tracker: ComputeTracker, step: int) -> None:
        if self._wandb is None:
            return
        self._wandb.log({"compute/gpu_hours": tracker.gpu_hours}, step=step)

    def log_scaling_point(
        self,
        architecture: str,
        compute_budget: float,
        best_clash_rate: float,
        best_gr_distance: float = float("inf"),
        param_count: int | None = None,
        flops_per_step: float | None = None,
    ) -> None:
        if self._wandb is None or self._scaling_table is None:
            return
        self._scaling_table.add_data(
            architecture, compute_budget, best_clash_rate, best_gr_distance, param_count, flops_per_step
        )
        self._wandb.log({"scaling/results": self._scaling_table})

    def finish(self) -> None:
        if self._wandb is None:
            return
        self._wandb.finish()
