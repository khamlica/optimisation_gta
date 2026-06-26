"""Cœur Bi2DPCA : entraînement C2DPCA-R2DPCA par régime, scores et seuils.

Pour chaque régime, on apprend deux sous-espaces :

- **temporel** (C2DPCA) via la covariance moyenne ``G_time = mean(A Aᵀ)`` (t×t),
  on garde ``d`` directions (CPV) → matrice ``V (t×d)`` ;
- **spatial** (R2DPCA) via ``G_space = mean(Dᵀ D)`` (m×m) avec ``D = Vᵀ A``,
  on garde ``p`` directions (CPV) → matrice ``U (m×p)``.

Les indices de monitoring sont ``Q_time``, ``Q_space`` et, en option, ``T2_time``,
``T2_space`` (formules exactes de ref/model_reference.md). Les seuils par régime
et par indice combinent un quantile KDE et un quantile empirique respectant le
FAR cible. Le jeu sain est nettoyé itérativement par quantiles robustes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gaussian_kde

from . import config
from .config import Params
from .healthy import robust_keep_mask

# Indices toujours calculés / activés en V1 (conseil MVP : Q d'abord).
QUALITY_INDICES = ("Q_time", "Q_space")
T2_INDICES = ("T2_time", "T2_space")


# --------------------------------------------------------------------------- #
# Algèbre : sous-espaces et scores
# --------------------------------------------------------------------------- #
def _eigh_desc(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Décomposition spectrale symétrique, valeurs/vecteurs en ordre décroissant."""
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)  # covariances : valeurs propres >= 0
    return vals, vecs[:, order]


def _rank_by_cpv(eigvals: np.ndarray, cpv: float, rank_max: int) -> int:
    """Plus petit rang dont la variance cumulée atteint ``cpv`` (borné par rank_max)."""
    total = eigvals.sum()
    if total <= 0:
        return 1
    cum = np.cumsum(eigvals) / total
    d = int(np.searchsorted(cum, cpv) + 1)
    return int(np.clip(d, 1, rank_max))


def _mean_gram_time(windows: np.ndarray) -> np.ndarray:
    """``G_time = mean_i (A_i A_iᵀ)`` (t×t) à partir d'un tenseur ``(N, t, m)``."""
    # einsum : pour chaque i, A_i @ A_i.T puis moyenne sur i.
    return np.einsum("itm,ism->ts", windows, windows) / windows.shape[0]


def _mean_gram_space(D: np.ndarray) -> np.ndarray:
    """``G_space = mean_i (D_iᵀ D_i)`` (m×m) à partir d'un tenseur ``(N, d, m)``."""
    return np.einsum("idm,idn->mn", D, D) / D.shape[0]


def score_window(
    A: np.ndarray,
    V: np.ndarray,
    U: np.ndarray,
    lambda_time: np.ndarray,
    lambda_space: np.ndarray,
) -> dict[str, float]:
    """Indices de monitoring d'une fenêtre ``A (t×m)`` (formules de la référence)."""
    D = V.T @ A                 # (d, m)
    A_hat_time = V @ D          # (t, m)

    F = D @ U                   # (d, p)
    D_hat_space = F @ U.T       # (d, m)

    q_time = float(((A - A_hat_time) ** 2).sum())
    q_space = float(((D - D_hat_space) ** 2).sum())

    t2_time = float(
        sum((D[j, :] ** 2).sum() / max(lambda_time[j], 1e-12) for j in range(len(lambda_time)))
    )
    t2_space = float(
        sum((F[:, k] ** 2).sum() / max(lambda_space[k], 1e-12) for k in range(len(lambda_space)))
    )
    return {"Q_time": q_time, "Q_space": q_space, "T2_time": t2_time, "T2_space": t2_space}


def score_windows(
    windows: np.ndarray,
    V: np.ndarray,
    U: np.ndarray,
    lambda_time: np.ndarray,
    lambda_space: np.ndarray,
) -> dict[str, np.ndarray]:
    """Scores vectorisés sur un tenseur de fenêtres ``(N, t, m)``."""
    D = np.einsum("td,itm->idm", V, windows)        # (N, d, m)
    A_hat = np.einsum("td,idm->itm", V, D)          # (N, t, m)
    F = np.einsum("idm,mp->idp", D, U)              # (N, d, p)
    D_hat = np.einsum("idp,mp->idm", F, U)          # (N, d, m)

    q_time = ((windows - A_hat) ** 2).sum(axis=(1, 2))
    q_space = ((D - D_hat) ** 2).sum(axis=(1, 2))
    t2_time = (D ** 2 / np.maximum(lambda_time, 1e-12)[None, :, None]).sum(axis=(1, 2))
    t2_space = (F ** 2 / np.maximum(lambda_space, 1e-12)[None, None, :]).sum(axis=(1, 2))
    return {"Q_time": q_time, "Q_space": q_space, "T2_time": t2_time, "T2_space": t2_space}


# --------------------------------------------------------------------------- #
# Seuils
# --------------------------------------------------------------------------- #
def _kde_quantile(samples: np.ndarray, q: float, n_grid: int = 2048) -> float:
    """Quantile estimé par KDE (intégration numérique de la densité).

    Repli sur le quantile empirique si la KDE est instable (échantillon trop
    petit ou variance nulle).
    """
    samples = samples[np.isfinite(samples)]
    if samples.size < 5 or np.ptp(samples) <= 0:
        return float(np.quantile(samples, q)) if samples.size else 0.0
    try:
        kde = gaussian_kde(samples)
        lo = float(samples.min())
        hi = float(samples.max() + 3.0 * samples.std())
        grid = np.linspace(lo, hi, n_grid)
        pdf = kde(grid)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        return float(np.interp(q, cdf, grid))
    except Exception:
        return float(np.quantile(samples, q))


def _empirical_far_threshold(samples: np.ndarray, far: float) -> float:
    """Plus petit seuil tel que la part d'échantillons au-dessus <= ``far``."""
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("inf")
    return float(np.quantile(samples, 1.0 - far))


def fit_threshold(samples: np.ndarray, params: Params) -> float:
    """Seuil d'un indice : ``max(quantile KDE, quantile empirique@FAR)``."""
    stat = _kde_quantile(samples, params.threshold_quantile)
    emp = _empirical_far_threshold(samples, params.far_target)
    return max(stat, emp)


# --------------------------------------------------------------------------- #
# Modèle par régime
# --------------------------------------------------------------------------- #
@dataclass
class RegimeModel:
    """Modèle Bi2DPCA d'un régime : sous-espaces, valeurs propres, seuils."""

    regime_id: int
    variables: list[str]
    mean: np.ndarray            # moyenne par variable (standardisation)
    std: np.ndarray             # écart-type par variable
    V: np.ndarray               # (t, d)
    U: np.ndarray               # (m, p)
    lambda_time: np.ndarray     # (d,)
    lambda_space: np.ndarray    # (p,)
    thresholds: dict[str, float]
    t: int
    n_train: int
    active_indices: tuple[str, ...]

    def standardize(self, windows: np.ndarray) -> np.ndarray:
        """Standardise par variable un tenseur ``(N, t, m)`` avec les stats du régime."""
        return (windows - self.mean[None, None, :]) / self.std[None, None, :]

    def score(self, windows_std: np.ndarray) -> dict[str, np.ndarray]:
        return score_windows(windows_std, self.V, self.U, self.lambda_time, self.lambda_space)


def _fit_subspaces(
    windows_std: np.ndarray, params: Params
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apprend V, U et les valeurs propres associées sur des fenêtres standardisées."""
    t, m = windows_std.shape[1], windows_std.shape[2]

    g_time = _mean_gram_time(windows_std)
    eig_t, vec_t = _eigh_desc(g_time)
    d = _rank_by_cpv(eig_t, params.cpv_time, min(t - 1, params.d_max))
    V = vec_t[:, :d]
    lambda_time = eig_t[:d]

    D = np.einsum("td,itm->idm", V, windows_std)  # (N, d, m)
    g_space = _mean_gram_space(D)
    eig_s, vec_s = _eigh_desc(g_space)
    p = _rank_by_cpv(eig_s, params.cpv_space, min(m - 1, params.p_max))
    U = vec_s[:, :p]
    lambda_space = eig_s[:p]
    return V, U, lambda_time, lambda_space


def train_regime_model(
    regime_id: int,
    variables: list[str],
    train_windows: np.ndarray,
    calib_windows: np.ndarray,
    params: Params = config.DEFAULT_PARAMS,
) -> RegimeModel:
    """Entraîne le modèle d'un régime, avec nettoyage robuste itératif du jeu sain.

    Boucle (``healthy_clean_iters`` fois) : standardiser → apprendre V,U →
    scorer le train → rejeter les fenêtres dont Q dépasse médiane + k·IQR →
    recommencer. Les seuils sont calés sur train + calib **nettoyés**.
    """
    active = QUALITY_INDICES + (T2_INDICES if params.use_t2 else ())

    kept = train_windows
    mean = std = None
    V = U = lambda_time = lambda_space = None

    for _ in range(max(1, params.healthy_clean_iters)):
        mean = kept.reshape(-1, kept.shape[2]).mean(axis=0)
        std = kept.reshape(-1, kept.shape[2]).std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        kept_std = (kept - mean[None, None, :]) / std[None, None, :]

        V, U, lambda_time, lambda_space = _fit_subspaces(kept_std, params)

        scores = score_windows(kept_std, V, U, lambda_time, lambda_space)
        keep = robust_keep_mask(scores["Q_time"], params.healthy_iqr_k) & robust_keep_mask(
            scores["Q_space"], params.healthy_iqr_k
        )
        if keep.all() or keep.sum() < max(10, V.shape[1] + 1):
            break
        kept = kept[keep]

    assert mean is not None and V is not None

    # Seuils : sur train nettoyé + calib (standardisés avec les stats finales).
    train_std = (kept - mean[None, None, :]) / std[None, None, :]
    calib_std = (
        (calib_windows - mean[None, None, :]) / std[None, None, :]
        if calib_windows.size
        else np.empty((0, kept.shape[1], kept.shape[2]))
    )
    cal_for_thr = np.concatenate([train_std, calib_std], axis=0) if calib_std.size else train_std
    cal_scores = score_windows(cal_for_thr, V, U, lambda_time, lambda_space)

    thresholds = {idx: fit_threshold(cal_scores[idx], params) for idx in active}

    return RegimeModel(
        regime_id=regime_id,
        variables=list(variables),
        mean=mean,
        std=std,
        V=V,
        U=U,
        lambda_time=lambda_time,
        lambda_space=lambda_space,
        thresholds=thresholds,
        t=kept.shape[1],
        n_train=int(kept.shape[0]),
        active_indices=active,
    )
