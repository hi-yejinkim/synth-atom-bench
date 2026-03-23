"""PyTorch dataset for unified 6-rule structured tasks.

Loads from .npz files produced by data/generate_unified.py.
Backbone atoms are indices 0..B-1; sidechain atoms are indices B..N_total-1.

Mirrors the interface of vsepr_plus_dataset.py but handles all 6 rules.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


ORBITAL_PARAMS: dict = {
    "sp3": {"bond_range": (1.4, 1.6)},
    "sp2": {"bond_range": (1.2, 1.4)},
    "sp":  {"bond_range": (1.0, 1.2)},
}

LONE_PAIR_REDUCTION: float = 2.5
ANGLE_ACCEPT_SIGMA: float = 3.0
BASE_ANGLE: dict = {"sp3": 109.5, "sp2": 120.0, "sp": 180.0}
ANGLE_SIGMA: dict = {0: 2.0, 1: 5.0, 2: 5.0}


class UnifiedDataset(Dataset):
    """Dataset for unified 6-rule structured tasks.

    Loads from a single .npz file produced by data/generate_unified.py.

    Attributes:
        positions (Tensor):            (N, N_total, 3) float32, centered at origin.
        N_backbone (int):              number of backbone atoms.
        box_size (float):              cubic box side length (Å).
        radius (float):                hard-sphere radius (Å).
        rule_flags (dict):             active rules as booleans.
        orbital_types (list[str]):     orbital type per backbone atom.
        n_lonepairs (Tensor):          (B,) int64.
        has_pi (Tensor):               (B,) bool.
        sidechain_counts (Tensor):     (B,) int64.
        bond_list (Tensor):            (n_bonds, 2) int64.
        contact_pairs (Tensor):        (n_contacts, 2) int64.
        contact_distances (Tensor):    (n_contacts,) float32.
        contact_bonded (Tensor):       (n_contacts,) bool.
        period_length (int):           periodicity period (0 = disabled).
        sidechain_parent (Tensor):     (N_sc,) int64, backbone index per sidechain.
        backbone_bond_ranges (Tensor): (B-1, 2) float32 [bond_min, bond_max].
        sidechain_bond_ranges (Tensor):(N_sc, 2) float32.
        target_angles (Tensor):        (B,) float32 VSEPR target angle in degrees.
        angle_tols (Tensor):           (B,) float32 angle tolerance in degrees.
        has_pi_arr (Tensor):           alias for has_pi.
        npz_meta (dict):               all fields needed by unified_violation_rate().
    """

    def __init__(self, path: str, max_samples: Optional[int] = None):
        """Load unified dataset from a .npz file.

        Args:
            path:        path to the .npz file produced by generate_unified.py.
            max_samples: if set, truncate to first max_samples examples.
        """
        data = np.load(path, allow_pickle=True)

        # ── Core positions ────────────────────────────────────────────────────
        positions = data["positions"]  # (N, N_total, 3)
        if max_samples is not None:
            positions = positions[:max_samples]

        self.positions = torch.tensor(positions, dtype=torch.float32)

        # ── Core attributes ───────────────────────────────────────────────────
        self.N_backbone = int(data["N_backbone"])
        self.box_size = float(data["box_size"])
        self.radius = float(data["radius"])
        B = self.N_backbone

        # ── Rule flags ────────────────────────────────────────────────────────
        self.rule_flags: dict = json.loads(str(data["rule_flags"]))

        # ── Structural metadata ───────────────────────────────────────────────
        raw_ot = data["orbital_types"]  # (B,) S4 bytes
        self.orbital_types: list = []
        for ot in raw_ot:
            if isinstance(ot, (bytes, np.bytes_)):
                self.orbital_types.append(ot.decode())
            else:
                self.orbital_types.append(str(ot))

        self.n_lonepairs = torch.tensor(
            data["n_lonepairs"].astype(np.int64), dtype=torch.long
        )  # (B,)

        self.has_pi = torch.tensor(
            data["has_pi"].astype(bool), dtype=torch.bool
        )  # (B,)
        self.has_pi_arr = self.has_pi  # alias

        self.sidechain_counts = torch.tensor(
            data["sidechain_counts"].astype(np.int64), dtype=torch.long
        )  # (B,)

        self.bond_list = torch.tensor(
            data["bond_list"].astype(np.int64), dtype=torch.long
        )  # (n_bonds, 2)

        # ── Global pair constraints ───────────────────────────────────────────
        cp = data["contact_pairs"]
        cp = cp.reshape(-1, 2) if cp.size > 0 else np.empty((0, 2), dtype=np.int32)
        self.contact_pairs = torch.tensor(cp.astype(np.int64), dtype=torch.long)

        cd = data["contact_distances"]
        self.contact_distances = torch.tensor(cd.astype(np.float32), dtype=torch.float32)

        cb = data["contact_bonded"]
        self.contact_bonded = torch.tensor(cb.astype(np.bool_), dtype=torch.bool)

        # ── Repulsive pairs (Rule 5 extension) ───────────────────────────────
        rp = data.get("repulsive_pairs", np.empty((0, 2), dtype=np.int32))
        if hasattr(rp, 'ndim') and rp.ndim == 2 and rp.shape[0] > 0:
            self.repulsive_pairs = torch.tensor(rp.astype(np.int64), dtype=torch.long)
        else:
            self.repulsive_pairs = torch.zeros((0, 2), dtype=torch.long)

        rd = data.get("repulsive_distances", np.empty((0,), dtype=np.float32))
        self.repulsive_distances = torch.tensor(rd.astype(np.float32), dtype=torch.float32)

        # ── Periodicity ───────────────────────────────────────────────────────
        self.period_length = int(data["period_length"])
        self.period_tol = float(data.get("period_tol", 0.5))
        self.angle_tol_factor = float(data.get("angle_tol_factor", 1.0))

        # ── Per-atom conditioning (N_total length) ────────────────────────────
        N_total = int(data.get("N_total", B + int(data["sidechain_counts"].sum())))
        if "atom_type_ids" in data:
            self.atom_type_ids = torch.tensor(
                data["atom_type_ids"].astype(np.int64), dtype=torch.long
            )
        else:
            # Backward compat: build from orbital_types
            ORBITAL_TO_ID = {"sp3": 0, "sp2": 1, "sp": 2}
            ids = np.zeros(N_total, dtype=np.int64)
            for i in range(B):
                ids[i] = ORBITAL_TO_ID.get(self.orbital_types[i], 0)
            ids[B:] = 3
            self.atom_type_ids = torch.tensor(ids, dtype=torch.long)

        if "is_backbone" in data:
            self.is_backbone = torch.tensor(data["is_backbone"].astype(bool), dtype=torch.bool)
        else:
            mask = np.zeros(N_total, dtype=bool)
            mask[:B] = True
            self.is_backbone = torch.tensor(mask, dtype=torch.bool)

        # ── Sidechain parent mapping ──────────────────────────────────────────
        if "sidechain_parent" in data:
            self.sidechain_parent = torch.tensor(
                data["sidechain_parent"].astype(np.int64), dtype=torch.long
            )
        else:
            sc_parent_list = []
            for bb_i, cnt in enumerate(data["sidechain_counts"]):
                sc_parent_list.extend([bb_i] * int(cnt))
            self.sidechain_parent = torch.tensor(sc_parent_list, dtype=torch.long)

        # ── Bond length ranges per bond ───────────────────────────────────────
        # Backbone bond i uses orbital type of atom i (same as generate_unified.py)
        bb_bond_ranges = []
        for i in range(B - 1):
            orbital = self.orbital_types[i]
            lo, hi = ORBITAL_PARAMS[orbital]["bond_range"]
            bb_bond_ranges.append([lo, hi])
        if bb_bond_ranges:
            self.backbone_bond_ranges = torch.tensor(
                bb_bond_ranges, dtype=torch.float32
            )  # (B-1, 2)
        else:
            self.backbone_bond_ranges = torch.zeros((0, 2), dtype=torch.float32)

        sc_bond_ranges = []
        for bb_i, cnt in enumerate(data["sidechain_counts"]):
            orbital = self.orbital_types[bb_i]
            lo, hi = ORBITAL_PARAMS[orbital]["bond_range"]
            for _ in range(int(cnt)):
                sc_bond_ranges.append([lo, hi])
        if sc_bond_ranges:
            self.sidechain_bond_ranges = torch.tensor(
                sc_bond_ranges, dtype=torch.float32
            )  # (N_sc, 2)
        else:
            self.sidechain_bond_ranges = torch.zeros((0, 2), dtype=torch.float32)

        # ── VSEPR target angles and tolerances ────────────────────────────────
        target_angles = []
        angle_tols = []
        for i in range(B):
            orbital = self.orbital_types[i]
            nlp = int(data["n_lonepairs"][i])
            base = BASE_ANGLE[orbital]
            target = base - LONE_PAIR_REDUCTION * nlp
            sigma = ANGLE_SIGMA.get(nlp, 5.0)
            tol = ANGLE_ACCEPT_SIGMA * sigma
            target_angles.append(target)
            angle_tols.append(tol)
        self.target_angles = torch.tensor(target_angles, dtype=torch.float32)  # (B,)
        self.angle_tols = torch.tensor(angle_tols, dtype=torch.float32)       # (B,)

        # ── npz_meta for unified_violation_rate() ─────────────────────────────
        self.npz_meta: dict = {
            "N_backbone": self.N_backbone,
            "n_lonepairs": self.n_lonepairs,
            "bond_list": self.bond_list,
            "radius": self.radius,
            "backbone_bond_ranges": self.backbone_bond_ranges,
            "sidechain_parent": self.sidechain_parent,
            "sidechain_counts": self.sidechain_counts,
            "sidechain_bond_ranges": self.sidechain_bond_ranges,
            "target_angles": self.target_angles,
            "angle_tols": self.angle_tols,
            "has_pi_arr": self.has_pi_arr,
            "contact_pairs": self.contact_pairs,
            "contact_distances": self.contact_distances,
            "contact_bonded": self.contact_bonded,
            "repulsive_pairs": self.repulsive_pairs,
            "repulsive_distances": self.repulsive_distances,
            "period_length": self.period_length,
            "period_tol": self.period_tol,
            "angle_tol_factor": self.angle_tol_factor,
            "orbital_types": self.orbital_types,
            "atom_type_ids": self.atom_type_ids,
            "is_backbone": self.is_backbone,
            "sidechain_counts_list": data["sidechain_counts"].tolist(),
            # Rule flags (individual booleans for easy access)
            "use_vsepr_slots": self.rule_flags.get("use_vsepr_slots", True),
            "use_vsepr_angles": self.rule_flags.get("use_vsepr_angles", True),
            "use_vsepr_bond_lengths": self.rule_flags.get("use_vsepr_bond_lengths", True),
            "use_vsepr_torsions": self.rule_flags.get("use_vsepr_torsions", False),
            "use_global_pairs": self.rule_flags.get("use_global_pairs", False),
            "use_global_periodicity": self.rule_flags.get("use_global_periodicity", False),
        }

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample with conditioning info.

        Returns:
            dict with:
              positions:     (N_total, 3) float32
              atom_type_ids: (N_total,) int64 — orbital type encoding per atom
              is_backbone:   (N_total,) bool — backbone vs sidechain mask
        """
        return {
            "positions": self.positions[idx],
            "atom_type_ids": self.atom_type_ids,
            "is_backbone": self.is_backbone,
        }
