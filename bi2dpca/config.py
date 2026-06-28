"""Configuration centrale du détecteur de dérive GTA.

Tout ce qui dépend d'un GTA particulier (noms de colonnes brutes) ou d'un
réglage de méthode est rassemblé ici, afin que le reste du pipeline reste
agnostique au GTA et facilement extensible aux 4 JFC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Variables canoniques
# --------------------------------------------------------------------------- #
# Ordre canonique imposé dans la matrice 2D A (colonnes = variables).
# MP n'existe que sur certains GTA (JFC3) : il est inséré seulement s'il est
# présent dans les données.
CANONICAL_ORDER: tuple[str, ...] = ("HP", "MP", "BP", "EE")

# Variables utilisées pour définir les régimes de fonctionnement.
# EE est volontairement exclue (règle de la référence : ne pas définir les
# régimes à partir de la variable surveillée/production).
REGIME_VARS: tuple[str, ...] = ("HP", "MP", "BP")


def _raw_columns(jfc: int, *, has_mp: bool) -> dict[str, str]:
    """Construit le mapping variable canonique -> nom de colonne brute."""
    cols = {
        "HP": f"Admission_HP_GTA_JFC{jfc}",
        "BP": f"Soutirage_BP_GTA_JFC{jfc}",
        "EE": f"Prod_EE_GTA_JFC{jfc}",
    }
    if has_mp:
        cols["MP"] = f"Soutirage_MP_GTA_JFC{jfc}"
    return cols


# Mapping gta_id -> {variable canonique: colonne brute}.
# Seul JFC3 possède une mesure MP (Soutirage_MP_GTA_JFC3).
GTA_CONFIGS: dict[str, dict[str, str]] = {
    "JFC1": _raw_columns(1, has_mp=False),
    "JFC2": _raw_columns(2, has_mp=False),
    "JFC3": _raw_columns(3, has_mp=True),
    "JFC4": _raw_columns(4, has_mp=False),
}

# Nom de la colonne d'horodatage dans les CSV bruts.
DATE_COLUMN: str = "Date"

# Exclusion explicite de variables par GTA (décision V1, documentée).
# JFC3 : MP est un canal dégénéré (quasi constant à ~0, voir
# artifacts/JFC3/diagnostic_mp.json) qui rend le GTA non surveillable. On
# l'écarte explicitement plutôt que par une règle automatique générique.
GTA_EXCLUDE_VARS: dict[str, tuple[str, ...]] = {
    "JFC3": ("MP",),
}

# Justification associée (sérialisée dans les artefacts).
GTA_EXCLUDE_REASON: dict[str, str] = {
    "JFC3": "MP excluded for JFC3 because diagnostic shows quasi-constant/degenerated channel",
}


def exclude_vars_for(gta_id: str) -> tuple[str, ...]:
    """Variables à écarter par défaut pour un GTA (config V1)."""
    return GTA_EXCLUDE_VARS.get(gta_id, ())


def exclude_reason_for(gta_id: str) -> str:
    """Justification de l'exclusion par défaut pour un GTA (vide si aucune)."""
    return GTA_EXCLUDE_REASON.get(gta_id, "")


# --------------------------------------------------------------------------- #
# Paramètres de méthode (réglages par défaut, surchargables)
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    """Hyperparamètres du pipeline Bi2DPCA.

    Les valeurs par défaut correspondent aux choix figés pour la V1 :
    données à 15 min, fenêtre 4 h (t=16), overlap 50 %, CPV 85 %.
    """

    # Échantillonnage / fenêtrage
    dt_minutes: int = 15           # pas régulier des données
    # Fenêtre 2 h -> t = 120/15 = 8. À 15 min le sous-espace temporel est quasi
    # 1-D (1er axe ~90 % de variance) : 2 h capte la dynamique tout en gardant
    # ~2x plus de fenêtres, plus de régimes surveillés et moins d'angles morts
    # (cf. étude t). Au-delà (4 h) on n'ajoute que des axes de faible énergie.
    window_minutes: int = 120      # durée de fenêtre 2D -> t = 120/15 = 8
    overlap: float = 0.50          # chevauchement offline -> stride = t*(1-overlap)

    # Réduction de dimension (Cumulative Percent Variance)
    cpv_time: float = 0.85
    cpv_space: float = 0.85
    d_max: int = 20                # garde-fou rang temporel : d <= min(t-1, d_max)
    p_max: int = 4                 # garde-fou rang spatial  : p <= min(m-1, p_max)

    # Indices activés (conseil MVP : Q d'abord, T2 ensuite)
    use_t2: bool = False

    # Seuils
    threshold_quantile: float = 0.99   # quantile KDE / empirique
    far_target: float = 0.01           # FAR cible (0.5 %–2 %)

    # Décision / persistance
    persistence_minutes: int = 120     # horizon de persistance pour alert
    exceed_ratio: float = 0.60         # ratio de dépassement sur l'horizon

    # Cadence du scoring en ligne (indépendante du stride d'entraînement) :
    # 1 pas = 15 min -> détection fine et persistance correctement échelonnée.
    online_stride_steps: int = 1

    # Split temporel (par régime, sans mélange aléatoire)
    train_frac: float = 0.70
    calib_frac: float = 0.15
    # test_frac = reste

    # Effectifs minimaux par régime : en-dessous, le régime est exclu du
    # modèle et marqué insufficient_data (seuils instables sinon).
    min_train_windows_per_regime: int = 100
    min_calib_windows_per_regime: int = 30

    # Régimes (clustering GMM)
    regime_n_components_grid: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    regime_smooth_window: int = 16     # médiane glissante sur les variables de régime
    regime_label_smooth_window: int = 8  # lissage majoritaire des labels (anti-papillotement)
    regime_random_state: int = 0
    regime_min_fit_points: int = 50    # garde-fou : minimum de points exploitables pour entraîner le GMM
    # Sélection du nombre de régimes : le BIC est calculé sur un sous-échantillon
    # DÉCORRÉLÉ (1 point sur regime_bic_thin) car les pas à 15 min sont fortement
    # autocorrélés (et pré-lissés) ; sans cela le BIC sur-segmente toujours.
    # Le fit final reste sur TOUTES les données. On retient ensuite le plus petit
    # k au « coude » : on cesse d'ajouter un régime dès que le gain marginal de
    # BIC tombe sous regime_bic_elbow_frac × (gain du tout premier ajout).
    regime_bic_thin: int = 16
    regime_bic_elbow_frac: float = 0.4
    # Marge de transition exclue APRÈS chaque changement de régime, en pas.
    # Découplée de t : le non-chevauchement des fenêtres est déjà garanti par
    # enumerate_windows ; cette marge ne sert qu'à écarter la rampe de
    # stabilisation du nouveau régime. 2 pas = 30 min (cf. décision V1).
    transition_steps: int = 2

    # Préfiltrage
    long_gap_steps: int = 4            # un trou > N pas consécutifs = trou long
    stuck_window: int = 8              # fenêtre de détection capteur bloqué
    stuck_min_std: float = 1e-6        # variance glissante en-dessous = bloqué
    # Comblage des trous COURTS (<= N pas) par interpolation linéaire, pour ne
    # pas perdre ~t fenêtres à cause d'un point isolé. Garde-fou : on ne comble
    # que si le saut aux bords reste petit (sinon le trou masque une vraie
    # discontinuité) — voir _fill_short_gaps. 0 = désactivé.
    max_fill_steps: int = 2
    fill_max_jump_k: float = 4.0       # saut bord max = k * pas-type * (L+1)

    # États d'arrêt (machine non surveillée : exclus du fenêtrage et du jeu sain)
    stop_vars: tuple[str, ...] = ("EE",)  # variable(s) de charge servant à détecter l'arrêt
    stop_frac: float = 0.10               # arrêt si charge < stop_frac * médiane opérationnelle
    # Bornes physiques par variable canonique ; None => dérivées des données
    # (quantiles robustes) au préfiltrage.
    physical_ranges: dict[str, tuple[float, float]] | None = None

    # Nettoyage robuste du jeu sain (médiane + k * IQR sur les scores Q)
    healthy_iqr_k: float = 3.0
    healthy_clean_iters: int = 2

    @property
    def t(self) -> int:
        """Nombre de points temporels par fenêtre 2D."""
        return int(self.window_minutes // self.dt_minutes)

    @property
    def stride(self) -> int:
        """Pas de glissement offline (overlap)."""
        return max(1, int(self.t * (1.0 - self.overlap)))

    @property
    def online_stride_minutes(self) -> int:
        """Cadence du scoring en ligne, en minutes."""
        return self.online_stride_steps * self.dt_minutes


# Instance de paramètres par défaut, prête à l'emploi.
DEFAULT_PARAMS = Params()


def variables_for(gta_id: str) -> list[str]:
    """Liste des variables canoniques déclarées pour un GTA, dans l'ordre canonique."""
    if gta_id not in GTA_CONFIGS:
        raise KeyError(f"GTA inconnu : {gta_id!r}. Connus : {sorted(GTA_CONFIGS)}")
    declared = GTA_CONFIGS[gta_id]
    return [v for v in CANONICAL_ORDER if v in declared]
