# Résonateur de Helmholtz — étude numérique (FDM, résonance transitoire, PINN)

Étude numérique d'un résonateur de Helmholtz axisymétrique, en trois volets complémentaires :

| # | Notebook | Question | Résultat vérifié |
|---|---|---|---|
| **1** | `etude_helmholtz.ipynb` | Où est la résonance et de quoi dépend-elle ? | V&V complète : MMS ordre **2,03**, GCI **≈2 %**, validation Selamet **0,6 / 2,2 %** |
| **2** | `resonance_transitoire.ipynb` | À quoi ressemble la résonance **en temps réel** ? | col ouvert + extérieur maillé → **f₀ = 209,8 Hz**, à **2,0 %** de la formule corrigée |
| **3** | `pinn_3d_transient.ipynb` | Un PINN peut-il résoudre le transitoire ? | piège du zéro diagnostiqué et **levé** → champ vérifié à **L2 ≈ 4,7 %** vs FDM |

![Résonance de Helmholtz — col ouvert](plots/helmholtz_resonance.gif)

*L'impulsion se propage, entre par le col, la cavité se remplit — puis l'impulsion repart et la
cavité **sonne seule** à sa fréquence propre. Volet 2.*

## Démarrage

```bash
pip install -r requirements.txt
jupyter lab etude_helmholtz.ipynb          # volet 1 — FDM / V&V
jupyter lab resonance_transitoire.ipynb    # volet 2 — résonance en transitoire
jupyter lab pinn_3d_transient.ipynb        # volet 3 — PINN
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
| f₀ (FFT du régime libre) | **209,8 Hz** | Helmholtz **avec** corrections de bout : 205,6 Hz | **2,0 %** |
| — | — | Helmholtz **sans** correction : 241,3 Hz | 15 % |
| L_eff identifiée | 52,9 mm | col géométrique 40 mm → correction **1,29 R_col** | attendu ~1,5 |
| Q_rayonnement | ≈ 105 | pertes visqueuses (Q≈47) dominent ⇒ Q réel ≈ 32 | — |

Le maillage extérieur **calcule** la correction de bout au lieu de la postuler : c'est la
**perspective n°1** du projet, désormais réalisée.

## Volet 3 — Time-Marching PINN transitoire

Un PINN mesh-free naïf **s'effondre** vers une solution non physique (~3·10⁻³ Pa, résidu pire que
`p≡0`). On établit une **vérité-terrain FDM**, on diagnostique les deux minima triviaux (zéro
« mou » et **mode DC constant**), puis trois correctifs rendent la méthode fonctionnelle :

1. **Non-dimensionnement** (résidu ÷ *S₀*, sortie *O(1)*).
2. **Gate d'IC-rampe** `g(t)=(t/T)·tanh(t/τ)` : repos initial en dur, interdit le mode DC constant.
3. **Contrainte intégrale du mode uniforme** `⟨p⟩(t)=∬⟨F⟩` (dérivée de la source, pas du FDM).

Entraînement par **curriculum temporel causal**. Résultat : **L2 ≈ 4,7 %** vs FDM (4,79 vs 4,96 Pa).

> ⚠️ **Périmètre** : ce volet utilise une **cavité fermée** (parois rigides partout), dont la
> réponse est une **rampe de pression** et non une résonance — c'est un banc d'essai
> *méthodologique* pour les pathologies d'entraînement des PINN. La résonance proprement dite est
> traitée au volet 2. Le PINN n'apporte ici **aucun gain de calcul** sur le FDM ; l'intérêt est
> méthodologique.

## Reproduire

```bash
python fdm_open_resonator.py          # volet 2 : résonance (~6 min)
python make_resonance_anim.py         # animation (après un run OR_TAG=_anim, cf. notebook)
python fdm_transient_reference.py     # volet 3 : vérité-terrain FDM (~90 s)
python train_pinn.py                  # volet 3 : entraînement PINN (~10 min CPU)
```

## Contenu

```
etude_helmholtz.ipynb          volet 1 — FDM, V&V, longueur effective
resonance_transitoire.ipynb    volet 2 — résonance transitoire (col ouvert)
pinn_3d_transient.ipynb        volet 3 — PINN (diagnostic + méthode + vérification)
fdm_open_resonator.py          solveur transitoire col ouvert + extérieur maillé
fdm_transient_reference.py     référence FDM transitoire (cavité fermée)
train_pinn.py                  entraînement du time-marching PINN
make_resonance_anim.py         animation de la résonance
make_showcase_anim.py          animation du champ PINN
build_*_notebook.py            génération des notebooks
data/  plots/  models/         données, figures, modèle entraîné
docs/PROTOCOLE_EXPERIMENTAL.md protocole de mesure sur résonateur réel
```

## Perspectives

PML formelle (au lieu des couches absorbantes) ; pertes viscothermiques résolues dans le
transitoire ; col émergeant sans baffle ; campagne expérimentale (cf. `docs/`).
