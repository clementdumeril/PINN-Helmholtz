"""
Convergence en maillage de la frequence propre f0 (resonateur a col ouvert).

Deux ameliorations sur la mesure brute par FFT :

1. ESTIMATEUR PRECIS. La FFT est limitee par sa resolution (4,9 Hz sur 250 ms),
   trop grossiere pour mesurer une derive de maillage de quelques Hz. On ajuste
   donc directement une sinusoide amortie sur le regime libre :

        p(t) = A * exp(-t/theta) * cos(2*pi*f0*t + phi)

   ce qui donne f0 bien en dessous de la largeur d'un canal FFT.

2. EXTRAPOLATION. Le solveur place ses parois une DEMI-MAILLE au-dela du
   dernier noeud (cf. mms_transient.py) : la geometrie effective depend de h,
   ce qui produit un biais du PREMIER ordre sur f0. On mesure l'ordre observe
   sur trois grilles et on extrapole a maillage nul (Richardson).

Entrees : data/open_resonator_h2.npz (2 mm), data/open_resonator.npz (1 mm),
          data/open_resonator_h05.npz (0,5 mm)
Sortie  : data/convergence_f0.npz, plots/convergence_f0.png
"""
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

C = 343.0
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
T_FREE = 0.045          # debut du regime libre


def fit_f0(t, p):
    """Ajuste A*exp(-t/theta)*cos(2*pi*f*t+phi) sur le regime libre."""
    m = t >= T_FREE
    tt = t[m] - t[m][0]
    pp = p[m]
    # amorce : FFT pour f0, decroissance grossiere pour theta
    sp = np.abs(np.fft.rfft(pp*np.hanning(pp.size)))
    fr = np.fft.rfftfreq(pp.size, tt[1]-tt[0])
    band = (fr > 60) & (fr < 900)
    f_guess = fr[band][np.argmax(sp[band])]
    env0 = np.abs(pp[:pp.size//10]).max()
    env1 = np.abs(pp[-pp.size//10:]).max()
    theta_guess = (tt[-1]) / max(np.log(max(env0/max(env1, 1e-30), 1.0001)), 1e-3)
    A_guess = env0

    def model(x, A, theta, f, phi):
        return A*np.exp(-x/theta)*np.cos(2*np.pi*f*x + phi)

    best = None
    for phi0 in (0.0, np.pi/2, np.pi, 3*np.pi/2):
        try:
            po, _ = curve_fit(model, tt, pp,
                              p0=[A_guess, theta_guess, f_guess, phi0],
                              maxfev=40000)
            r = np.sum((pp - model(tt, *po))**2)
            if best is None or r < best[1]:
                best = (po, r)
        except Exception:
            pass
    if best is None:
        return f_guess, np.nan, np.nan
    A, theta, f, phi = best[0]
    Q = np.pi*abs(f)*abs(theta)
    resid = np.sqrt(best[1]/np.sum(pp**2))
    return abs(f), Q, resid


CASES = [(2.0e-3, "data/open_resonator_h2.npz"),
         (1.0e-3, "data/open_resonator.npz"),
         (0.5e-3, "data/open_resonator_h05.npz")]

hs, f0s, Qs = [], [], []
print("=== MESURE PRECISE PAR AJUSTEMENT DE SINUSOIDE AMORTIE ===\n")
for h, path in CASES:
    if not os.path.exists(path):
        print(f"h = {h*1e3:.2f} mm : {path} absent — ignore")
        continue
    d = np.load(path)
    f0, Q, res = fit_f0(d["t"], d["p_cav"])
    f_fft = float(d["f0_meas"])
    hs.append(h); f0s.append(f0); Qs.append(Q)
    print(f"h = {h*1e3:5.2f} mm | f0(FFT) = {f_fft:7.2f} Hz | "
          f"f0(ajust.) = {f0:8.3f} Hz | Q = {Q:7.1f} | residu {res*100:.2f} %")

hs = np.array(hs); f0s = np.array(f0s)

# --- geometrie effective : parois a une demi-maille au-dela ---
def f_helm(Rn, Rc, Hc, Leff):
    S = np.pi*Rn**2; V = np.pi*Rc**2*Hc
    return C/(2*np.pi)*np.sqrt(S/(V*Leff))

print("\n=== BIAIS GEOMETRIQUE DE DEMI-MAILLE (prediction) ===")
print("le solveur simule R + h/2 et H + h, pas les cotes nominales")
for h in hs:
    fn = f_helm(R_NECK, R_CAV, H_CAV, L_NECK + 1.29*R_NECK)
    fe = f_helm(R_NECK + h/2, R_CAV + h/2, H_CAV + h, L_NECK + 1.29*R_NECK)
    print(f"  h = {h*1e3:5.2f} mm : ecart predit sur f0 = {(fe/fn - 1)*100:+5.2f} %")

if len(hs) >= 3:
    f3, f2, f1 = f0s[0], f0s[1], f0s[2]        # grossier, moyen, fin
    p = np.log(abs((f3 - f2)/(f2 - f1)))/np.log(2.0)
    f_ext = f1 + (f1 - f2)/(2**p - 1)
    gci = 1.25*abs((f2 - f1)/f1)/(2**p - 1)
    print(f"\n=== EXTRAPOLATION (3 grilles) ===")
    print(f"  ordre observe p      = {p:.3f}")
    print(f"  f0 extrapolee        = {f_ext:.2f} Hz")
    print(f"  incertitude GCI      = {gci*100:.2f} %")
elif len(hs) == 2:
    f2, f1 = f0s[0], f0s[1]
    f_ext = f1 + (f1 - f2)          # ordre 1 suppose (biais geometrique)
    p, gci = 1.0, abs((f1-f2)/f1)
    print(f"\n=== EXTRAPOLATION (2 grilles, ordre 1 SUPPOSE) ===")
    print(f"  f0 extrapolee        = {f_ext:.2f} Hz   (ordre non verifie)")
else:
    f_ext, p, gci = np.nan, np.nan, np.nan

f_theo = f_helm(R_NECK, R_CAV, H_CAV, L_NECK + 0.85*R_NECK + 0.66*R_NECK)
print(f"\n=== COMPARAISON ===")
print(f"  Helmholtz + corrections de bout : {f_theo:.2f} Hz")
if np.isfinite(f_ext):
    print(f"  f0 extrapolee (ce calcul)       : {f_ext:.2f} Hz   -> ecart {abs(f_ext-f_theo)/f_theo*100:.2f} %")
print(f"  volet frequentiel (impedance)   : 209.84 Hz")

np.savez("data/convergence_f0.npz", h=hs, f0=f0s, Q=np.array(Qs),
         f_ext=f_ext, order=p, gci=gci, f_theo=f_theo)

if len(hs) >= 2:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(hs*1e3, f0s, "o-", color="#1d5480", label="f0 mesuree (ajustement)")
    if np.isfinite(f_ext):
        ax.axhline(f_ext, ls="--", color="#1d5480", lw=1,
                   label=f"extrapolee h=0 : {f_ext:.1f} Hz")
    ax.axhline(f_theo, ls=":", color="#a8442a", lw=1.4,
               label=f"Helmholtz corrigee : {f_theo:.1f} Hz")
    ax.set_xlabel("pas de maillage h (mm)"); ax.set_ylabel("f0 (Hz)")
    ax.set_title("Convergence en maillage de la frequence propre")
    ax.grid(ls=":"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("plots/convergence_f0.png", dpi=110)
    print("\nFigure : plots/convergence_f0.png")
