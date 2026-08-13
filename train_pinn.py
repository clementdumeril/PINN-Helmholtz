"""
Time-Marching PINN (curriculum causal) -- trainer DEFINITIF.

Champ p(r,z,t) = scale * g(t) * N(r,z,t),  g(t) = (t/T) * tanh(t/tau)
  -> IC de repos EN DUR (p(.,0)=0, p_t(.,0)=0 exacts),
  -> biais de RAMPE lineaire (interdit le mode DC constant = piege),
  -> pas de forme parabolique imposee (t>>tau : g ~ t/T).

Trois ingredients qui font qu'il MARCHE (le notebook d'origine s'effondrait a 3e-3 Pa) :
  1. Non-dim : sortie O(1) x scale_p=5 Pa (echelle FDM) ; residu PDE / SRC_S0.
  2. Contrainte INTEGRALE exacte du mode uniforme : <p_tt>(t) = <F>(t).
     (moyenne spatiale ; <lap p>=0 sous Neumann). Pilote la rampe de facon
     STABLE et peu bruitee, la ou le residu ponctuel de la source
     (non-resoluble a l'echelle sub-mm) deraille.
  3. Source ponctuelle DE-PONDEREE (src_frac faible) pour ne pas injecter
     de bruit une fois la rampe etablie.

Curriculum : on etend l'horizon [0,H_k], H_k=k*T/K (time-marching causal),
reseau rechauffe, nouveau bandeau sur-echantillonne.

Verification vs FDM (data/fdm_transient_reference.npz). Selection du meilleur
modele sur la LOSS INTERNE (aucune fuite de la reference).

Sorties : models/pinn_marching.pth, plots/train_pinn.png, data/train_pinn.npz
Env : TP_K(12) TP_ITERS(400/palier) TP_NCOL(1500) TP_HID(96) TP_TAU(0.0015) TP_LR(1e-3)
      TP_CURR(1)  (0 = fenetre unique)
"""
import os, time
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"; torch.set_num_threads(4)

R_NECK, R_CAV, L_NECK = 0.01, 0.04, 0.04
Z_MAX = 0.12; C = 343.0
F_START, F_END, T_MAX = 50.0, 800.0, 0.025
SRC_ZC, SRC_W, SRC_S0 = 0.006, 0.005, 8.0e7

K      = int(os.environ.get("TP_K", 12))
ITERS  = int(os.environ.get("TP_ITERS", 400))
N_COL  = int(os.environ.get("TP_NCOL", 1500))
HID    = int(os.environ.get("TP_HID", 96))
TAU    = float(os.environ.get("TP_TAU", 0.0015))
LR     = float(os.environ.get("TP_LR", 1e-3))
CURR   = int(os.environ.get("TP_CURR", 1))
SCALE_P = 5.0
W_DC, W_WALL, W_SRC = 10.0, 2.0, 1.0
SRC_FRAC = 0.10

def temporal_np(t):
    phase = 2.0*np.pi*(F_START*t + (F_END - F_START)/(2.0*T_MAX)*t*t)
    return np.sin(phase)*np.exp(-((t - T_MAX/2)**2)/(2.0*(T_MAX/3)**2))
def temporal_t(t):
    phase = 2.0*np.pi*(F_START*t + (F_END - F_START)/(2.0*T_MAX)*t*t)
    return torch.sin(phase)*torch.exp(-((t - T_MAX/2)**2)/(2.0*(T_MAX/3)**2))
def spatial_t(r, z):
    return torch.exp(-(r**2 + (z - SRC_ZC)**2)/(2.0*SRC_W**2))
def forcing_torch(r, z, t):
    return SRC_S0*spatial_t(r, z)*temporal_t(t)

# --- K_F = <spatial>_domaine (moyenne volumique axisymetrique), une fois ---
def compute_KF():
    nr, nz = 400, 1200
    r = np.linspace(0, R_CAV, nr); z = np.linspace(0, Z_MAX, nz)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    dom = ((ZZ < L_NECK) & (RR <= R_NECK)) | ((ZZ >= L_NECK) & (RR <= R_CAV))
    w = RR*dom                                   # poids axisymetrique 2*pi*r (le 2pi s'annule)
    sp = np.exp(-(RR**2 + (ZZ - SRC_ZC)**2)/(2.0*SRC_W**2))
    return float((sp*w).sum()/w.sum())
K_F = compute_KF()
print(f"K_F = <spatial>_dom = {K_F:.4e} | <F>_max ~ {SRC_S0*K_F:.2f}")

# --- Cible du mode uniforme P_target(t) = doubl. integrale de <F> (analytique,
#     issue de la SOURCE, pas du FDM) : consequence exacte de <p>''=<F>, <p>(0)=<p'>(0)=0
_tg = np.linspace(0.0, T_MAX, 400000)
_Fbar = SRC_S0*K_F*temporal_np(_tg)
_v = np.concatenate([[0.0], np.cumsum(0.5*(_Fbar[1:]+_Fbar[:-1])*np.diff(_tg))])
_P = np.concatenate([[0.0], np.cumsum(0.5*(_v[1:]+_v[:-1])*np.diff(_tg))])
def P_target(times):
    return np.interp(times, _tg, _P)
print(f"P_target(T) = {_P[-1]:.3f} Pa (cible rampe du mode uniforme)")

class MLP(nn.Module):
    def __init__(self, hidden=96, layers=5):
        super().__init__()
        net = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(layers-1): net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()
    def forward(self, x): return self.net(x)

model = MLP(HID).to(DEV)
def Tn(a, g=False):
    return torch.tensor(a, dtype=torch.float32, device=DEV).reshape(-1,1).requires_grad_(bool(g))
def p_of(r, z, t):
    x = torch.cat([r/R_CAV, z/Z_MAX, t/T_MAX], 1)
    return SCALE_P * ((t/T_MAX)*torch.tanh(t/TAU)) * model(x)

def sample_interior(n):
    rs, zs = [], []; need = n
    while need > 0:
        m = int(need*2.2)+64
        rr = np.random.uniform(0, R_CAV, m); zz = np.random.uniform(0, Z_MAX, m)
        k = ((zz < L_NECK) & (rr <= R_NECK)) | ((zz >= L_NECK) & (rr <= R_CAV))
        rs.append(rr[k]); zs.append(zz[k]); need = n - sum(len(a) for a in rs)
    return np.concatenate(rs)[:n], np.concatenate(zs)[:n]

def loss_pde(H, Hprev):
    n_src = int(N_COL*SRC_FRAC); n_uni = N_COL - n_src
    r1, z1 = sample_interior(n_uni)
    r2 = np.clip(np.abs(np.random.normal(0, SRC_W, n_src)), 0, R_NECK)
    z2 = np.clip(np.random.normal(SRC_ZC, 1.5*SRC_W, n_src), 0, L_NECK)
    r = np.concatenate([r1, r2]); z = np.concatenate([z1, z2])
    nt = r.size; n_new = int(nt*0.3)
    t = np.empty(nt)
    t[:nt-n_new] = np.random.uniform(0, H, nt-n_new)
    t[nt-n_new:] = np.random.uniform(Hprev, H, n_new)
    r = Tn(r,1); z = Tn(z,1); t = Tn(t,1)
    p = p_of(r, z, t)
    pr = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    pz = torch.autograd.grad(p, z, torch.ones_like(p), create_graph=True)[0]
    pt = torch.autograd.grad(p, t, torch.ones_like(p), create_graph=True)[0]
    prr = torch.autograd.grad(pr, r, torch.ones_like(pr), create_graph=True)[0]
    pzz = torch.autograd.grad(pz, z, torch.ones_like(pz), create_graph=True)[0]
    ptt = torch.autograd.grad(pt, t, torch.ones_like(pt), create_graph=True)[0]
    inv_r = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    pr_over_r = torch.where(r < 1e-6, prr, pr*inv_r)
    F = forcing_torch(r, z, t)
    res = (ptt - C**2*(prr + pr_over_r + pzz) - F)/SRC_S0
    return torch.mean(res**2)

def loss_dc(H, n_t=32, n_s=64):
    """Contrainte du mode uniforme au niveau DEPLACEMENT : <p>(t) = P_target(t).
    (aucune derivee -> cible lisse, stable ; pilote la rampe a la bonne amplitude)"""
    times = np.random.uniform(0, H, n_t)
    r, z = sample_interior(n_t*n_s)
    t = np.repeat(times, n_s)
    p = p_of(Tn(r), Tn(z), Tn(t))
    p_m = p.reshape(n_t, n_s).mean(1)
    P_m = torch.tensor(P_target(times), dtype=torch.float32).reshape(n_t)
    return torch.mean(((p_m - P_m)/SCALE_P)**2)

def loss_walls(H):
    m = N_COL//4//4
    segs = [(np.full(m, R_NECK), np.random.uniform(0, L_NECK, m), 1., 0.),
            (np.random.uniform(R_NECK, R_CAV, m), np.full(m, L_NECK), 0., 1.),
            (np.full(m, R_CAV), np.random.uniform(L_NECK, Z_MAX, m), 1., 0.),
            (np.random.uniform(0, R_CAV, m), np.full(m, Z_MAX), 0., 1.)]
    rw = np.concatenate([s[0] for s in segs]); zw = np.concatenate([s[1] for s in segs])
    nx = np.concatenate([np.full(m, s[2]) for s in segs]); ny = np.concatenate([np.full(m, s[3]) for s in segs])
    r = Tn(rw,1); z = Tn(zw,1); t = Tn(np.random.uniform(0, H, rw.size),1)
    p = p_of(r, z, t)
    pr = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    pz = torch.autograd.grad(p, z, torch.ones_like(p), create_graph=True)[0]
    gradn = (Tn(nx)*pr + Tn(ny)*pz)/(SCALE_P/R_CAV)
    return torch.mean(gradn**2)

# --- reference FDM pour validation ---
ref = np.load("data/fdm_transient_reference.npz")
tt = ref["t"]; pc_ref = ref["probe_cav"]; pn_ref = ref["probe_neck"]
def l2_val():
    with torch.no_grad():
        pc = p_of(Tn(np.zeros_like(tt)), Tn(np.full_like(tt, Z_MAX)), Tn(tt)).numpy().flatten()
    return np.linalg.norm(pc-pc_ref)/np.linalg.norm(pc_ref), np.abs(pc).max()

horizons = [(k+1)*T_MAX/K for k in range(K)] if CURR else [T_MAX]
opt = torch.optim.Adam(model.parameters(), lr=LR)
best = {"loss": 1e9, "state": None}
t0 = time.time()
print(f"CURR={CURR} K={len(horizons)} ITERS={ITERS} ncol={N_COL} hid={HID} tau={TAU*1e3}ms lr={LR}")
for k, H in enumerate(horizons):
    Hprev = horizons[k-1] if k > 0 else 0.0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for g in opt.param_groups: g['lr'] = LR
    for it in range(1, ITERS+1):
        opt.zero_grad()
        lp = loss_pde(H, Hprev); ld = loss_dc(H); lw = loss_walls(H)
        loss = W_SRC*lp + W_DC*ld + W_WALL*lw
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if H >= T_MAX - 1e-9 and loss.item() < best["loss"]:
            best = {"loss": loss.item(), "state": {kk: v.clone() for kk, v in model.state_dict().items()}}
    l2i, mxi = l2_val()
    print(f"  palier {k+1:2d}/{len(horizons)} H={H*1e3:4.1f}ms | PDE={lp.item():.2e} DC={ld.item():.2e} "
          f"Wall={lw.item():.2e} | L2={l2i*100:5.1f}% max|p|={mxi:.2f} | {time.time()-t0:.0f}s")

if best["state"] is not None:
    model.load_state_dict(best["state"])
    print(f"[best interne] loss={best['loss']:.3e}")
torch.save({"state": model.state_dict(), "scale_p": SCALE_P, "hid": HID, "tau": TAU}, "models/pinn_marching.pth")

# --- verdict + figures ---
l2c, mxc = l2_val()
with torch.no_grad():
    pc = p_of(Tn(np.zeros_like(tt)), Tn(np.full_like(tt, Z_MAX)), Tn(tt)).numpy().flatten()
    pn = p_of(Tn(np.zeros_like(tt)), Tn(np.full_like(tt, SRC_ZC)), Tn(tt)).numpy().flatten()
l2n = np.linalg.norm(pn-pn_ref)/np.linalg.norm(pn_ref)
print("\n=== VERDICT (vs FDM) ===")
print(f"max|p| PINN sonde cavite : {mxc:.3f} Pa (FDM {np.abs(pc_ref).max():.3f})")
print(f"L2 sonde cavite : {l2c*100:.1f} %   L2 sonde col : {l2n*100:.1f} %")

np.savez("data/train_pinn.npz", t=tt, pc_pinn=pc, pc_ref=pc_ref, pn_pinn=pn, pn_ref=pn_ref, l2c=l2c, l2n=l2n)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(tt*1e3, pc_ref, "k", lw=1.3, label="FDM"); ax[0].plot(tt*1e3, pc, "C1", lw=1.0, label="PINN")
ax[0].set_title(f"sonde cavite | L2={l2c*100:.1f}%"); ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("p (Pa)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(tt*1e3, pn_ref, "k", lw=1.3, label="FDM"); ax[1].plot(tt*1e3, pn, "C1", lw=1.0, label="PINN")
ax[1].set_title(f"sonde col | L2={l2n*100:.1f}%"); ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("p (Pa)"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("plots/train_pinn.png", dpi=110)
print("Figure: plots/train_pinn.png | Modele: models/pinn_marching.pth")
