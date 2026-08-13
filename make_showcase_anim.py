# -*- coding: utf-8 -*-
"""Animation vitrine du time-marching PINN (champ physique verifie vs FDM).
Trois panneaux synchronises :
  (1) champ PINN p(r,z,t) qui se remplit (rampe du mode uniforme) ;
  (2) partie acoustique p - <p>(t) (structure spatiale, amplifiee) ;
  (3) sondes PINN vs FDM avec curseur temporel.
Sortie : plots/pinn_field_showcase.gif
"""
import os
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

R_NECK, R_CAV, L_NECK = 0.01, 0.04, 0.04
Z_MAX = 0.12; T_MAX = 0.025
F_START, F_END = 50.0, 800.0

# --- modele ---
class MLP(nn.Module):
    def __init__(self, hidden=96, layers=5):
        super().__init__()
        net = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(layers-1): net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)

ck = torch.load("models/pinn_marching.pth", map_location="cpu")
SCALE_P, TAU, HID = ck["scale_p"], ck["tau"], ck["hid"]
model = MLP(HID); model.load_state_dict(ck["state"]); model.eval()

def p_field(r, z, t):
    r = torch.as_tensor(r, dtype=torch.float32).reshape(-1,1)
    z = torch.as_tensor(z, dtype=torch.float32).reshape(-1,1)
    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1,1)
    x = torch.cat([r/R_CAV, z/Z_MAX, t/T_MAX], 1)
    gate = (t/T_MAX)*torch.tanh(t/TAU)
    with torch.no_grad():
        return (SCALE_P*gate*model(x)).numpy().flatten()

# --- grille + reference FDM ---
ref = np.load("data/fdm_transient_reference.npz")
rg, zg, dom = ref["r"], ref["z"], ref["dom"]
t_ref, pc_ref, pn_ref = ref["t"], ref["probe_cav"], ref["probe_neck"]
RG, ZG = np.meshgrid(rg, zg, indexing="ij")
Zc = np.where(dom, 0.0, np.nan)

NF = 96
times = np.linspace(0.3e-3, T_MAX, NF)
# precalcul des champs PINN
fields = np.empty((NF, RG.shape[0], RG.shape[1]))
for i, tt in enumerate(times):
    fields[i] = p_field(RG.ravel(), ZG.ravel(), np.full(RG.size, tt)).reshape(RG.shape)
vfull = 5.3
pc_pinn = p_field(np.zeros_like(t_ref), np.full_like(t_ref, Z_MAX), t_ref)
pn_pinn = p_field(np.zeros_like(t_ref), np.full_like(t_ref, 0.006), t_ref)
fchirp = F_START + (F_END - F_START)/T_MAX*times

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(11, 6.4))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.15], hspace=0.5)
axA = fig.add_subplot(gs[0, 0]); axC = fig.add_subplot(gs[1, 0])

zmm, rmm = zg*1e3, rg*1e3
qmA = axA.pcolormesh(zmm, rmm, np.where(dom, fields[0], np.nan), cmap="YlOrRd",
                     vmin=0.0, vmax=vfull, shading="auto")
axA.set(xlabel="z (mm)", ylabel="r (mm)",
        title="Champ de pression PINN — le mode uniforme « se remplit »  [Pa]")
axA.set_aspect("equal"); fig.colorbar(qmA, ax=axA, fraction=0.046, pad=0.02)
axA.annotate("col", (6, 5), fontsize=8, color="0.3")
axA.annotate("cavité", (75, 20), fontsize=8, color="0.3")

axC.plot(t_ref*1e3, pc_ref, "k", lw=1.5, label="FDM cavité")
axC.plot(t_ref*1e3, pc_pinn, "C1", lw=1.2, label="PINN cavité")
axC.plot(t_ref*1e3, pn_ref, color="0.6", lw=1.1, label="FDM col")
axC.plot(t_ref*1e3, pn_pinn, "C0", lw=1.0, alpha=0.85, label="PINN col")
axC.set(xlabel="t (ms)", ylabel="p (Pa)", xlim=(0, T_MAX*1e3), ylim=(-0.5, 5.6))
axC.grid(alpha=0.3); axC.legend(ncol=4, fontsize=8, loc="upper left")
cursor = axC.axvline(times[0]*1e3, color="C3", lw=1.5)
sup = fig.suptitle("", fontsize=12.5, y=0.98)

def update(i):
    qmA.set_array(np.where(dom, fields[i], np.nan).ravel())
    cursor.set_xdata([times[i]*1e3, times[i]*1e3])
    sup.set_text(f"Time-Marching PINN vérifié vs FDM (L2 ≈ 4,7 %)   |   "
                 f"t = {times[i]*1e3:5.1f} ms   (chirp {fchirp[i]:4.0f} Hz)")
    return qmA, cursor, sup

anim = FuncAnimation(fig, update, frames=NF, blit=False)
out = "plots/pinn_field_showcase.gif"
anim.save(out, writer=PillowWriter(fps=14))
print(f"Anime : {out}  ({NF} frames)")
