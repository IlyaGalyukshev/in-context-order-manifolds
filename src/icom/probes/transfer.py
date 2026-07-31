"""Transfer probes: train on subset A, test on subset B.

Cross-condition (shuffle→forward), cross-N (9→12), and cross-family (S0→S1,
size→loud) generalization. A code that transfers is a reusable order manifold,
not a per-stimulus / position-tied artifact. The PCA + scaler are fit on TRAIN
only and applied to TEST (no leakage).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def transfer_spearman(Xtr, ytr, Xte, yte, pca=64, alpha=10.0, seed=0):
    """Spearman of a ridge rank decode trained on (Xtr,ytr), evaluated on
    (Xte,yte). Returns nan if either side is too small."""
    if len(Xtr) < 5 or len(Xte) < 3:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    k = min(pca, Xtr.shape[0] - 1, Xtr.shape[1])
    if k >= 1:
        pc = PCA(k, random_state=seed).fit(Ztr)
        Ztr, Zte = pc.transform(Ztr), pc.transform(Zte)
    pred = Ridge(alpha=alpha).fit(Ztr, ytr).predict(Zte)
    rho = spearmanr(pred, yte)[0]
    # SIGNED: a shared code should preserve orientation across A->B; a negative
    # rho means the code is anti-aligned/inconsistent, which abs() would hide.
    return float(rho) if not np.isnan(rho) else float("nan")
