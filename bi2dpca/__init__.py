"""Détecteur de dérive GTA par Bi2DPCA (C2DPCA-R2DPCA) dynamique par régime.

Implémentation V1 fidèle à ref/model_reference.md : la détection repose
uniquement sur des fenêtres 2D (temps x variables) et les indices
Q_time / Q_space / T2_time / T2_space, par régime. EE est une variable
surveillée, jamais prédite.
"""

__all__ = ["config", "io_data"]
