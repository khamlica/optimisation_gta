"""Chargement modulaire des données GTA.

Un seul point d'entrée, `load_gta`, sait lire aussi bien un CSV mono-GTA
(`Data_Energie_JFC1.csv`) que le CSV multi-GTA (`Data_Energie_4JFC.csv`) et
renvoie un DataFrame indexé par le temps, avec des colonnes **canoniques**
`[HP, (MP), BP, EE]`. Le reste du pipeline n'a donc jamais à connaître les noms
de colonnes bruts propres à chaque JFC.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from . import config

# Emplacements possibles du CSV multi-GTA (fallback quand pas de fichier dédié).
FOURJFC_CANDIDATES: tuple[str, ...] = (
    "data/Data_Energie_4JFC.csv",
    "../gta/data/Data_Energie_4JFC.csv",
)


@dataclass
class GtaData:
    """Données chargées pour un GTA.

    Attributes
    ----------
    gta_id:
        Identifiant du GTA (ex. ``"JFC1"``).
    df:
        DataFrame indexé par un ``DatetimeIndex`` trié, colonnes = ``variables``.
    variables:
        Variables canoniques réellement présentes, dans l'ordre canonique.
    """

    gta_id: str
    df: pd.DataFrame
    variables: list[str]

    @property
    def n_vars(self) -> int:
        return len(self.variables)


def resolve_data_path(gta_id: str) -> str:
    """Détermine le CSV à utiliser pour un GTA.

    Priorité au fichier dédié ``data/Data_Energie_<GTA>.csv`` s'il existe (cas de
    JFC1, dont la colonne EE est absente du CSV multi-GTA), sinon repli sur le
    CSV multi-GTA ``Data_Energie_4JFC.csv``.
    """
    dedicated = f"data/Data_Energie_{gta_id}.csv"
    if os.path.exists(dedicated):
        return dedicated
    for cand in FOURJFC_CANDIDATES:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Aucune source de données trouvée pour {gta_id} "
        f"(ni {dedicated}, ni {FOURJFC_CANDIDATES})."
    )


def load_gta(path: str, gta_id: str) -> GtaData:
    """Charge les données d'un GTA depuis un CSV (mono- ou multi-GTA).

    Parameters
    ----------
    path:
        Chemin du fichier CSV. La première colonne (index entier brut) est ignorée.
    gta_id:
        Clé de ``config.GTA_CONFIGS`` (ex. ``"JFC1"``).

    Returns
    -------
    GtaData
        DataFrame canonique indexé par le temps + liste des variables présentes.

    Raises
    ------
    KeyError
        Si ``gta_id`` est inconnu ou si aucune colonne attendue n'est trouvée.
    """
    if gta_id not in config.GTA_CONFIGS:
        raise KeyError(
            f"GTA inconnu : {gta_id!r}. Connus : {sorted(config.GTA_CONFIGS)}"
        )

    raw = pd.read_csv(path)

    # La 1re colonne des CSV est un index entier brut sans nom utile -> on la
    # repère par position et on la retire si elle n'est pas la date.
    first_col = raw.columns[0]
    if first_col != config.DATE_COLUMN and raw[first_col].dtype.kind in "iu":
        raw = raw.drop(columns=[first_col])

    if config.DATE_COLUMN not in raw.columns:
        raise KeyError(
            f"Colonne d'horodatage {config.DATE_COLUMN!r} absente de {path}"
        )

    # Sélection + renommage canonique des colonnes présentes pour ce GTA.
    col_map = config.GTA_CONFIGS[gta_id]  # {canonique: brute}
    present = {
        canon: brute for canon, brute in col_map.items() if brute in raw.columns
    }
    if not present:
        raise KeyError(
            f"Aucune colonne de {gta_id} trouvée dans {path}. "
            f"Attendu (parmi) : {sorted(col_map.values())}"
        )

    keep = [config.DATE_COLUMN, *present.values()]
    df = raw[keep].copy()
    df = df.rename(columns={brute: canon for canon, brute in present.items()})

    # Index temporel trié, strictement utilisable en aval.
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    df = df.set_index(config.DATE_COLUMN).sort_index()

    # Ordonner les variables selon l'ordre canonique.
    variables = [v for v in config.CANONICAL_ORDER if v in present]
    df = df[variables].astype(float)

    return GtaData(gta_id=gta_id, df=df, variables=variables)
