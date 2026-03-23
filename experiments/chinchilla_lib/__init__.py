"""Chinchilla-style scaling law study modules."""

from experiments.chinchilla_lib.config import (
    BATCH_SIZE, D_TARGETS, D_STEPS, D_NAMES, D_VALUES, EVAL_EVERY,
    LRS, LR_NAMES,
    CHINCHILLA_SIZES, CHINCHILLA_5_SIZES, CHINCHILLA_7_SIZES,
    ALL_TASKS, ALL_ARCHS,
)
from experiments.chinchilla_lib.helpers import (
    _lr_name, _ckpt_dir, _traj_path, _grid_meta_path,
    _results_path, _fits_path, _fits_approach1_path, _csv_path,
    _get_grad_accum, _measure_flops, _is_complete,
)
from experiments.chinchilla_lib.generate import generate
from experiments.chinchilla_lib.run import run
from experiments.chinchilla_lib.collect import collect
from experiments.chinchilla_lib.fit import fit, fit_approach1
from experiments.chinchilla_lib.plot import plot
