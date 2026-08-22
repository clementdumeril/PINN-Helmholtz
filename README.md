# Résonateur de Helmholtz — étude numérique par différences finies

Étude numérique d'un résonateur de Helmholtz axisymétrique, en deux volets complémentaires :

| # | Notebook | Question | Résultat vérifié |
|---|---|---|---|
| **1** | `etude_helmholtz.ipynb` | Où est la résonance et de quoi dépend-elle ? | V&V complète : MMS ordre **2,03**, GCI **≈2 %**, validation Selamet **0,6 / 2,2 %** |
| **2** | `resonance_transitoire.ipynb` | À quoi ressemble la résonance **en temps réel** ? | col ouvert + extérieur maillé → **f₀ = 204,6 Hz** (extrapolée, GCI 1,5 %), à **0,47 %** de la formule corrigée |

![Résonance de Helmholtz — col ouvert](plots/helmholtz_resonance.gif)

*L'impulsion se propage, entre par le col, la cavité se remplit — puis l'impulsion repart et la
cavité **sonne seule** à sa fréquence propre. Volet 2.*

## Démarrage

```bash
pip install -r requirements.txt
jupyter lab etude_helmholtz.ipynb          # volet 1 — FDM / V&V
jupyter lab resonance_transitoire.ipynb    # volet 2 — résonance en transitoire
```

Les notebooks sont livrés **avec leurs sorties** (figures visibles sans rien exécuter).

---

## Volet 1 — FDM : vérification, validation, longueur effective

| Niveau | Question | Méthode | Résultat |
|---|---|---|---|
| **Code** | le schéma est-il bien programmé ? | solution manufacturée (MMS) | ordre **2,03** |
| **Solution** | l'erreur de maillage est-elle bornée ? | convergence GCI (Roache) + 4ᵉ grille | incertitude **≈ 2 %** |
| **Modèle** | reproduit-il la réalité ? | mesures publiées (Selamet *et al.* 1997) | écarts **0,6 % / 2,2 %** |

- **Loi d'échelle** : après propagation de l'incertitude numérique, loi de puissance et modèle à
  longueur effective sont statistiquement indiscernables (ΔAICc non décisif) ; le modèle physique
  `f₀ = A_H/√(L+ΔL_eff)` est préféré (**A_H ≈ 48**, **ΔL_eff ≈ 0,66 R_col**, prédiction
  leave-one-out 10× meilleure). L'exposant apparent *b ≈ −0,42* est un artefact de plage restreinte.
- **Correction de bout** décomposée intérieur/extérieur via une condition d'impédance de rayonnement.
- **Pertes** dominées par la couche limite du col (**Q ≈ 47**) ; absorption volumique négligeable.

## Volet 2 — Résonance en régime transitoire (col ouvert, extérieur maillé)

Le col est **réellement ouvert** sur un demi-espace extérieur **maillé** (baffle plan, couches
absorbantes), excité par une **impulsion large bande**. Le résonateur **sélectionne** sa fréquence
propre : après le passage de l'impulsion, la cavité continue d'osciller.

| Grandeur | Mesuré | Référence | Écart |
|---|---|---|---|
| f₀ **extrapolée à maillage nul** | **204,6 Hz** (GCI 1,5 %) | Helmholtz **avec** corrections de bout : 205,6 Hz | **0,47 %** |
| — | — | Helmholtz **sans** correction : 241,3 Hz | 15 % |
| Ordre de convergence observé | 0,86 | 3 grilles : 2 / 1 / 0,5 mm | — |
| Q par rayonnement | **non mesurable ici** | varie d'un facteur 6 selon les frontières | artefact numérique |

Le maillage extérieur **calcule** la correction de bout au lieu de la postuler : c'est la
**perspective n°1** du projet, désormais réalisée.

**Vérification (ajoutée après coup, et elle corrige deux résultats).** Le solveur place ses parois
une demi-maille au-delà du dernier nœud — la géométrie simulée vaut donc `R+h/2` et `H+h`, ce qui
biaise f₀ au premier ordre. Trois conséquences :

- la fréquence brute à h=1 mm (209,8 Hz) semblait coïncider avec le calcul fréquentiel par
  impédance (209,84 Hz) ; cette coïncidence était **fortuite**, due au biais de maillage ;
- une fois extrapolée, f₀ = **204,6 Hz**, à 0,47 % de la théorie corrigée — un accord plus faible
  en apparence, mais cette fois **contrôlé** et assorti d'une incertitude ;
- f₀ est **parfaitement robuste** au traitement des frontières (variation 0,00 %), alors que **Q
  varie d'un facteur 6** : ce calcul mesure une fréquence propre, pas un amortissement.

Vérification de code : `mms_transient.py` (solution manufacturée espace-temps, **ordre 1,98**),
qui teste le masque, les flux nuls et l'axe — ce que la MMS du volet 1 ne couvrait pas.

## Reproduire

```bash
python fdm_open_resonator.py          # volet 2 : résonance (~6 min)
python mms_transient.py               # vérification de code du transitoire (~10 s)
python convergence_f0.py              # convergence en maillage de f0
python make_resonance_anim.py         # animation (après un run OR_TAG=_anim, cf. notebook)
python make_mode_anim.py              # animation du mode résonant établi (~30 s)
```

## Contenu

```
etude_helmholtz.ipynb          volet 1 — FDM fréquentiel, V&V, longueur effective
resonance_transitoire.ipynb    volet 2 — résonance transitoire (col ouvert)
fdm_open_resonator.py          solveur transitoire col ouvert + extérieur maillé
mms_transient.py               vérification de code du transitoire (MMS espace-temps)
convergence_f0.py              convergence en maillage de f0 + extrapolation
make_resonance_anim.py         animation du régime transitoire
make_mode_anim.py              animation du mode résonant établi (quadrature col/cavité)
build_resonance_notebook.py    génération du notebook du volet 2
data/  plots/                  données et figures
docs/PROTOCOLE_EXPERIMENTAL.md protocole de mesure sur résonateur réel
```

## Perspectives

PML formelle (au lieu des couches absorbantes) ; pertes viscothermiques résolues dans le
transitoire ; col émergeant sans baffle ; campagne expérimentale (cf. `docs/`).
