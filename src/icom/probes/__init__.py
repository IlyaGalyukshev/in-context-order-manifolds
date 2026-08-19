"""Offline probe catalog for BCS activations (pure CPU numpy/scipy/sklearn).

Runs over the stored `[N_entities x (L+1) x D]` fp16 per-stimulus npz written by
extract_activations.py. Nothing here touches a GPU — extraction is paid once,
the whole Phase-C catalog is cheap and re-runnable.

Modules:
  loading    — read per-stimulus records, slice a layer, interior masks.
  linear     — interior-only ridge rank probe + permutation null + per-rank MAE.
  nonlinear  — MLP probe (curvature signature vs the linear probe).
  geometry   — RSA (dist vs |Δrank|), TwoNN intrinsic dimension, projections.
  transfer   — train-on-A / test-on-B (condition / N / family generalization).
  depth      — onset / peak / band from a per-layer profile (the "which layers").
  curvature  — G4: linear vs nonlinear vs geodesic, principal-curve curvature,
               curved-irreducible, Engels separability (is the manifold curved?).
  circular   — G8: circle-fit, cyclic-vs-linear RSA, angular decode (is it a ring?).
  alignment  — G10: source-axis reuse — contrast/rank directions, cosine, subspaces.
"""

from icom.probes.loading import (load_records, n_layers, stack_layer, stack_all_layers,
                                  interior_mask)
from icom.probes.linear import cv_spearman, cv_predict, reduce, probe_with_null, per_rank_mae
from icom.probes.nonlinear import mlp_cv_spearman
from icom.probes.geometry import rsa_rank, twonn, intrinsic_dim, project
from icom.probes.transfer import transfer_spearman
from icom.probes.depth import depth_stats
from icom.probes.curvature import (curvature_profile, geodesic_cv_spearman,
                                   principal_curve_curvature, curved_irreducible,
                                   separability_index)
from icom.probes.circular import circle_fit, circular_rsa, angular_decode
from icom.probes.crossnobis import (crossnobis_rdm, noise_sd, whitened_rsa,
                                    line_rdm, ring_rdm)
from icom.probes.alignment import (contrast_direction, rank_direction,
                                   rank_contrast_direction, direction_cosine,
                                   subspace_alignment)

__all__ = [
    "load_records", "n_layers", "stack_layer", "stack_all_layers", "interior_mask",
    "cv_spearman", "cv_predict", "reduce", "probe_with_null", "per_rank_mae",
    "mlp_cv_spearman", "rsa_rank", "twonn", "intrinsic_dim", "project",
    "transfer_spearman", "depth_stats",
    "curvature_profile", "geodesic_cv_spearman", "principal_curve_curvature",
    "curved_irreducible", "separability_index",
    "circle_fit", "circular_rsa", "angular_decode",
    "crossnobis_rdm", "noise_sd", "whitened_rsa", "line_rdm", "ring_rdm",
    "contrast_direction", "rank_direction", "rank_contrast_direction",
    "direction_cosine", "subspace_alignment",
]
