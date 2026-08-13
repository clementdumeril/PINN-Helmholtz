"""
Animation : propagation de l'onde, entree par le col, ACCUMULATION et RESONANCE
dans un resonateur de Helmholtz a col ouvert (domaine exterieur maille).

Lit data/open_resonator_anim.npz (produit par fdm_open_resonator.py) et rend
plots/helmholtz_resonance.gif :
  * panneau haut  : coupe meridienne (miroir) du champ de pression, exterieur inclus ;
  * panneau bas   : signaux sonde exterieure / sonde cavite + curseur temporel.

Le champ est renormalise (systeme lineaire) pour que le PIC INCIDENT vaille 1 Pa.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
d = np.load("data/open_resonator_anim.npz")
r, z, dom = d["r"], d["z"], d["dom"]
frames, ft = d["frames"], d["frame_t"]
t, p_cav, p_out = d["t"], d["p_cav"], d["p_out"]
f0 = float(d["f0_meas"])

R_NECK, R_CAV, L_NECK, Z_TOP = 0.01, 0.04, 0.04, 0.12

# --- renormalisation : pic incident = 1 Pa (systeme lineaire) ---
K = 1.0/np.abs(p_out).max()
frames = frames*K; p_cav = p_cav*K; p_out = p_out*K

# --- miroir en r pour une coupe meridienne complete ---
r_full = np.concatenate([-r[::-1], r[1:]])
def mirror(a):
    return np.concatenate([a[::-1, :], a[1:, :]], axis=0)
dom_f = mirror(dom)
vmax = 1.15*np.abs(p_cav).max()          # echelle calee sur la resonance (la source sature)

# sous-echantillonnage temporel (poids du GIF)
STEP = int(os.environ.get("AN_STEP", 2))
idx = np.arange(0, len(ft), STEP)

plt.rcParams.update({"font.size": 9.5})
fig = plt.figure(figsize=(8.8, 6.9))
gs = GridSpec(2, 1, height_ratios=[1.5, 1.0], hspace=0.38)
axF = fig.add_subplot(gs[0]); axS = fig.add_subplot(gs[1])

zmm, rmm = z*1e3, r_full*1e3
qm = axF.pcolormesh(zmm, rmm, np.where(dom_f, mirror(frames[0]), np.nan),
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
axF.set(xlabel="z (mm)", ylabel="r (mm)", title="")
axF.set_aspect("equal")
axF.set_xlim(-72, 122); axF.set_ylim(-64, 64)     # masque les couches absorbantes
fig.colorbar(qm, ax=axF, fraction=0.026, pad=0.012, label="p (Pa)")
axF.plot(-55, 0, "k*", ms=8); axF.annotate("source", (-55, -9), fontsize=8,
                                           color="0.3", ha="center")

# contours de la geometrie
for sgn in (1, -1):
    axF.plot([0, L_NECK*1e3], [sgn*R_NECK*1e3]*2, "k", lw=1.1)
    axF.plot([L_NECK*1e3, L_NECK*1e3], [sgn*R_NECK*1e3, sgn*R_CAV*1e3], "k", lw=1.1)
    axF.plot([L_NECK*1e3, Z_TOP*1e3], [sgn*R_CAV*1e3]*2, "k", lw=1.1)
    axF.plot([0, 0], [sgn*R_NECK*1e3, sgn*rmm.max()], "k", lw=2.2)   # baffle
axF.plot([Z_TOP*1e3, Z_TOP*1e3], [-R_CAV*1e3, R_CAV*1e3], "k", lw=1.1)
axF.annotate("extérieur", (-66, 52), fontsize=8.5, color="0.35")
axF.annotate("col ouvert", (20, 26), fontsize=8.5, color="0.35", ha="center")
axF.annotate("cavité", (80, 50), fontsize=8.5, color="0.35", ha="center")
axF.annotate("baffle", (3, 56), fontsize=8.5, color="0.35", ha="left")

axS.plot(t*1e3, p_out, color="0.6", lw=0.8, label="sonde extérieure (impulsion incidente)")
axS.plot(t*1e3, p_cav, "C0", lw=0.9, label="sonde cavité (résonance)")
axS.set(xlabel="t (ms)", ylabel="p (Pa)", xlim=(0, ft.max()*1e3))
axS.grid(alpha=0.3); axS.legend(fontsize=8, loc="upper right")
cur = axS.axvline(ft[0]*1e3, color="C3", lw=1.4)
sup = fig.suptitle("", fontsize=10.5, y=0.978)

def phase_label(tt):
    if tt < 0.010: return "l'impulsion se propage"
    if tt < 0.022: return "l'onde entre par le col, la cavité se remplit"
    return "impulsion partie : la cavité sonne seule"

def update(k):
    i = idx[k]
    qm.set_array(np.where(dom_f, mirror(frames[i]), np.nan).ravel())
    cur.set_xdata([ft[i]*1e3, ft[i]*1e3])
    sup.set_text(f"Résonateur de Helmholtz à col ouvert  |  t = {ft[i]*1e3:5.1f} ms  "
                 f"—  {phase_label(ft[i])}   (f₀ ≈ {f0:.0f} Hz)")
    return qm, cur, sup

anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
out = "plots/helmholtz_resonance.gif"
anim.save(out, writer=PillowWriter(fps=20))

# --- allegement du GIF (palette reduite) ---
from PIL import Image, ImageSequence
im = Image.open(out)
fr = [f.convert("RGB").quantize(colors=80, method=Image.MEDIANCUT)
      for f in ImageSequence.Iterator(im)]
fr[0].save(out, save_all=True, append_images=fr[1:], duration=50, loop=0, optimize=True)
print(f"Anime : {out}  ({len(idx)} frames, f0={f0:.0f} Hz, "
      f"{os.path.getsize(out)/1e6:.1f} Mo)")
