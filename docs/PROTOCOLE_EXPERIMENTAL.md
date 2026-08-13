# Protocole expérimental — validation de f₀ et Q sur résonateur réel

Objectif : confronter les prédictions numériques (fréquence propre et facteur de
qualité) à une mesure physique simple, avec un téléphone comme microphone.
C'est la seule étape qu'aucun calcul ne remplace.

## 1. Matériel

- Une bouteille ou un récipient rigide au col aussi cylindrique que possible
  (bouteille en verre ; éviter le plastique souple qui ajoute des pertes de paroi).
- Un téléphone avec une application d'enregistrement WAV/M4A (fréquence
  d'échantillonnage ≥ 44,1 kHz), ou un micro USB.
- Un pied à coulisse / règle pour mesurer la géométrie.

## 2. Mesure de la géométrie (à reporter dans le script)

| Grandeur | Symbole | Comment |
|---|---|---|
| Rayon intérieur du col | a | pied à coulisse, moyenne de 2 diamètres ⊥ |
| Longueur du col | L | du plan de l'embouchure au début de l'évasement |
| Volume de la cavité | V | remplir d'eau jusqu'à la base du col, peser (1 g = 1 cm³) |

Prédictions à confronter (formules du rapport) :
- f₀ = (c/2π)·√(S/(V·(L + ΔL))) avec S = πa², ΔL ≈ 0,66a (intérieur) + 0,85a
  (extérieur bafflé) — utiliser ΔL total ≈ 1,5a pour une embouchure affleurante,
  plutôt 0,6a extérieur si le col dépasse (non bafflé).
- Q_visc = a/(F_t·δ_v), δ_v = √(2μ/(ρ·2πf₀)), F_t ≈ 1,48.
  Le Q total mesuré inclut aussi le rayonnement : 1/Q_tot = 1/Q_visc + 1/Q_rad.

## 3. Excitation et enregistrement (méthode du ring-down)

1. Pièce calme, micro à ~5–10 cm de l'embouchure, hors du jet d'air.
2. Exciter par une impulsion : claquer la paume à plat sur l'embouchure et
   retirer immédiatement, ou donner une pichenette sèche tangentielle au col.
   (Éviter de souffler : l'écoulement ajoute un décalage de fréquence et du bruit.)
3. Enregistrer 3–5 secondes ; répéter 5 fois (statistique).
4. Exporter en WAV et lancer :

```bash
cd codes
python src/analyze_recording.py chemin/vers/enregistrement.wav
```

## 4. Ce que fait le script d'analyse

- FFT du signal complet → f₀ mesurée (pic dominant, interpolation parabolique) ;
- filtrage passe-bande autour de f₀ → enveloppe de Hilbert du ring-down →
  décrément logarithmique → **Q mesuré en domaine temporel** (même méthode que
  le solveur, comparaison directe) ;
- affichage des prédictions vs mesures avec écarts relatifs.

## 5. Sources d'écart attendues (à discuter dans le rapport)

- géométrie réelle non cylindrique (épaulement, évasement du col) ;
- ΔL extérieur : embouchure ni parfaitement bafflée ni libre ;
- pertes de paroi (verre mince) et fuite au niveau de la main ;
- température (c varie de ±0,6 m/s par °C ; mesurer T ambiante).

Un accord à ~5 % sur f₀ et un Q mesuré entre 20 et 60 seraient conformes à la
littérature (Moloney 2004 : la théorie surestime légèrement Q).
