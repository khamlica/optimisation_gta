## Points à garder en tête

### Objectif du projet

Le projet implémente une méthode de **monitoring dynamique du GTA par régimes**, fondée sur des fenêtres 2D `temps × variables`.

Le but n’est pas de prédire `EE`, mais de détecter si le comportement dynamique actuel du GTA reste cohérent avec les comportements appris par régime.

### Principe méthodologique

* `EE` est une variable surveillée.
* `EE` n’est jamais prédite.
* Les régimes sont définis à partir des variables de conduite : `HP`, `MP` si disponible, `BP`.
* `EE` est volontairement exclue du clustering des régimes.
* Le monitoring se fait par régime : on ne compare jamais une fenêtre d’un régime avec un autre régime.

### Pipeline actuel

1. **Chargement des données**

   * mapping des colonnes PI vers les variables canoniques ;
   * support JFC1/JFC2/JFC3/JFC4 ;
   * `MP` uniquement si disponible.

2. **Préprocessing**

   * grille temporelle régulière ;
   * suppression logique des points non exploitables ;
   * pas d’interpolation pour le monitoring ;
   * tout point avec `NaN` est non exploitable ;
   * détection des trous longs, capteurs bloqués et valeurs hors bornes.

3. **Identification des régimes**

   * clustering GMM sur `HP`, `MP`, `BP` ;
   * sélection du nombre de régimes par BIC ;
   * lissage des labels ;
   * marquage des transitions ;
   * les points non exploitables restent toujours non labellisés.

4. **Fenêtres 2D**

   * une fenêtre est valide seulement si tous ses points sont :

     * exploitables ;
     * hors transition ;
     * hors arrêt ;
     * dans le même régime.
   * un régime avec beaucoup de points éparpillés n’est pas suffisant : il faut des blocs consécutifs assez longs pour construire des fenêtres.

5. **Jeu sain et modèle**

   * split temporel par régime ;
   * pas de mélange aléatoire ;
   * entraînement Bi2DPCA par régime ;
   * scores vectorisés ;
   * nettoyage robuste du jeu sain ;
   * seuils KDE + seuils empiriques.

### Résultats actuels sur JFC1

| Régime | d / p | n_train nettoyé | FAR calib | FAR test |
| ------ | ----: | --------------: | --------: | -------: |
| 0      | 1 / 2 |             164 |     2,6 % |      0 % |
| 2      | 1 / 1 |             589 |     6,3 % |   55,9 % |
| 3      | 1 / 1 |             317 |     3,8 % |      0 % |
| 4      | 1 / 2 |              52 |       0 % |    7,7 % |

### Points de vigilance

* `d=1` partout : cohérent avec la CPV 85 %, mais à tester plus tard avec CPV 90/95 %.
* Régime 2 : FAR test très élevé. Cela peut indiquer soit une dérive réelle récente, soit un régime trop hétérogène.
* Régime 4 : peu de fenêtres d’entraînement, donc fiabilité plus limitée.
* Les résultats actuels valident la mécanique, pas encore la performance finale.

### Décisions importantes

* Ne pas réintroduire de prédiction `EE`.
* Ne pas utiliser Kalman/RANSAC dans cette méthode.
* Ne pas mélanger les régimes.
* Ne pas scorer les transitions.
* Ne pas interpréter un FAR élevé comme un bug avant l’étape de validation.
* La validation finale devra examiner les régimes instables, notamment le régime 2.

### Prochaine étape

Implémenter `online.py` :

* scoring d’une fenêtre courante ;
* exclusion des transitions et régimes inconnus ;
* comparaison aux seuils ;
* états `normal`, `warning`, `alert`, `non_scoré` ;
* logique de persistance ;
* `reason_codes` explicites.

Étape 6 validée fonctionnellement.

online.py implémente :
- n_persist = 8 pour 2 h à pas 15 min ;
- transition → statut transition, non scoré ;
- régime inconnu → unknown_regime ;
- reason_codes explicites ;
- warning pour dépassements isolés ;
- alert pour dépassements persistants.

Test régime 2 :
- 64 normal
- 4 warning
- 77 alert

Les alertes du régime 2 prolongent le signal déjà observé à l’étape 5 :
- soit dérive réelle récente ;
- soit régime 2 trop hétérogène.

Ce point n’est pas traité comme bug.
Il sera analysé à l’étape validation.