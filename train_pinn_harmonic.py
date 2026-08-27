"""
PINN HARMONIQUE sur le resonateur a col OUVERT.

Change de formulation plutot que de se battre contre le temps. On suppose
    p(r,z,t) = Re{ P(r,z) * exp(i*w*t) }
ce qui transforme l'equation d'onde en equation de HELMHOLTZ :
    lap(P) + k^2 P = -F        k = w/c
Le reseau apprend le champ COMPLEXE P(r,z) : deux sorties (partie reelle,
partie imaginaire), aucune dimension temporelle.

Ce que cela supprime, par rapport a la tentative transitoire qui a echoue :
  * les 52 periodes d'oscillation  -> plus de dimension temps du tout ;
  * l'exterieur maille et sa couche absorbante -> remplaces par une condition
    d'IMPEDANCE DE RAYONNEMENT a la bouche (meme modele de piston bafle que
    le volet frequentiel) ;
  * la source ponctuelle intense a 4 mm -> remplacee par le forcage volumique
    large (1 cm) place dans la cavite, comme dans etude_helmholtz.ipynb.

Restent : la geometrie en L (col + cavite), les parois rigides, et le champ
sous-longueur d'onde -- tout ce qu'un MLP lisse sait representer.

Conditions aux limites
  * parois rigides            : dP/dn = 0
  * bouche (z=0, r<=R_col)    : dP/dz = alpha * P,  alpha = i*w*rho/Z_r
                                Z_r = rho*c*[ (ka)^2/2 + i*8*ka/(3*pi) ]
                                (piston bafle, developpement basse frequence)

Verification : solveur FDM complexe independant, ecrit dans ce meme fichier.
Sortie : models/pinn_harmonic.pth, data/pinn_harmonic.npz, plots/pinn_harmonic.png

Env : PH_F (frequence Hz, 209.84) PH_ITERS (6000) PH_NCOL (3000) PH_HID (96)
      PH_LR (2e-3) PH_SEED (0) PH_TAG ("")
"""
import os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
SEED = int(os.environ.get("PH_SEED", 0))
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(4)

# --- physique / geometrie (identiques au volet frequentiel) ---
C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
FREQ = float(os.environ.get("PH_F", 209.84))
W = 2*np.pi*FREQ; K2 = (W/C)**2
# source volumique large, dans la cavite (comme solve_config du volet 1)
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5*H_CAV, 0.01, 1.0e4
# impedance de rayonnement du piston bafle
ka = W/C*R_NECK
Zr = RHO*C*(0.5*ka**2 + 1j*(8/(3*np.pi))*ka)
ALPHA = 1j*W*RHO/Zr

ITERS = int(os.environ.get("PH_ITERS", 6000))
NCOL  = int(os.environ.get("PH_NCOL", 3000))
HID   = int(os.environ.get("PH_HID", 96))
LR    = float(os.environ.get("PH_LR", 2e-3))
TAG   = os.environ.get("PH_TAG", "")

print(f"f = {FREQ} Hz | k = {W/C:.3f} 1/m | lambda = {2*np.pi*C/W:.2f} m | ka = {ka:.4f}")
print(f"alpha = {ALPHA:.4g}  (condition de Robin a la bouche)")


# ==========================================================================
# 1) REFERENCE : solveur FDM complexe independant
# ==========================================================================
def fdm_reference(h=5e-4):
    Nr = int(round(R_CAV/h))+1; Nz = int(round(Z_TOP/h))+1
    r = np.arange(Nr)*h; z = np.arange(Nz)*h
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    fluid = ((ZZ < L_NECK-1e-12) & (RR <= R_NECK+1e-12)) | ((ZZ >= L_NECK-1e-12) & (RR <= R_CAV+1e-12))
    mouth = fluid & (np.abs(ZZ) < 1e-12) & (RR <= R_NECK+1e-12)
    idx = lambda i, j: i + j*Nr
    N = Nr*Nz; ih2 = 1.0/h**2
    rows, cols, dat = [], [], []
    b = np.zeros(N, complex)
    F = SRC_A*np.exp(-(RR**2 + (ZZ-SRC_Z)**2)/(2*SRC_W**2))
    for i in range(Nr):
        for j in range(Nz):
            ci = idx(i, j)
            if not fluid[i, j]:
                rows += [ci]; cols += [ci]; dat += [1.0]; continue
            diag = K2 + 0j
            # --- radial ---
            if i == 0:
                diag += -4*ih2
                rows += [ci]; cols += [idx(1, j)]; dat += [4*ih2]
            else:
                lc = ih2*(1-0.5/i); rc = ih2*(1+0.5/i); diag += -2*ih2
                if fluid[i-1, j]: rows += [ci]; cols += [idx(i-1, j)]; dat += [lc]
                else:             diag += lc
                if i+1 < Nr and fluid[i+1, j]: rows += [ci]; cols += [idx(i+1, j)]; dat += [rc]
                else:                          diag += rc
            # --- axial ---
            if mouth[i, j]:
                # noeud fantome elimine par dP/dz = alpha*P
                diag += -2*ih2 - 2*ALPHA/h
                rows += [ci]; cols += [idx(i, j+1)]; dat += [2*ih2]
            else:
                diag += -2*ih2
                for jj in (j-1, j+1):
                    if 0 <= jj < Nz and fluid[i, jj]:
                        rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                    else:
                        diag += ih2
            rows += [ci]; cols += [ci]; dat += [diag]
            b[ci] = -F[i, j]
    A = sp.coo_matrix((dat, (rows, cols)), shape=(N, N), dtype=complex).tocsr()
    P = spsolve(A, b).reshape((Nz, Nr)).T
    return r, z, fluid, np.where(fluid, P, np.nan)

t0 = time.time()
r_ref, z_ref, fluid_ref, P_ref = fdm_reference()
SCALE = float(np.nanmax(np.abs(P_ref)))
print(f"reference FDM : grille {fluid_ref.shape}, {fluid_ref.sum()} noeuds, {time.time()-t0:.1f} s")
print(f"               max|P| = {SCALE:.4f} Pa")


# ==========================================================================
# 2) LE RESEAU : champ complexe P = Pr + i*Pi
# ==========================================================================
class CplxNet(nn.Module):
    def __init__(self, hidden=96, layers=5):
        super().__init__()
        net = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers-1): net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*net)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05); self.net[-1].bias.zero_()
    def forward(self, x): return self.net(x)

model = CplxNet(HID)

def Tn(a, g=False):
    return torch.tensor(a, dtype=torch.float32).reshape(-1, 1).requires_grad_(bool(g))

def P_of(r, z):
    o = model(torch.cat([r/R_CAV, z/Z_TOP], 1))
    return SCALE*o[:, 0:1], SCALE*o[:, 1:2]

def sample_interior(n):
    rs, zs = [], []; need = n
    while need > 0:
        m = int(need*2.2)+64
        rr = np.random.uniform(0, R_CAV, m); zz = np.random.uniform(0, Z_TOP, m)
        k = ((zz < L_NECK) & (rr <= R_NECK)) | ((zz >= L_NECK) & (rr <= R_CAV))
        rs.append(rr[k]); zs.append(zz[k]); need = n - sum(len(a) for a in rs)
    return np.concatenate(rs)[:n], np.concatenate(zs)[:n]

def lap_of(Pc, r, z):
    pr = torch.autograd.grad(Pc, r, torch.ones_like(Pc), create_graph=True)[0]
    pz = torch.autograd.grad(Pc, z, torch.ones_like(Pc), create_graph=True)[0]
    prr = torch.autograd.grad(pr, r, torch.ones_like(pr), create_graph=True)[0]
    pzz = torch.autograd.grad(pz, z, torch.ones_like(pz), create_graph=True)[0]
    inv = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    return prr + torch.where(r < 1e-6, prr, pr*inv) + pzz, pr, pz

NORM = K2*SCALE          # echelle attendue des termes de l'equation

# ==========================================================================
# CONTRAINTE INTEGRALE EXACTE (fixe l'amplitude)
# ==========================================================================
# En integrant lap(P) + k^2 P = -F sur tout le domaine :
#   * les parois rigides ne contribuent pas (dP/dn = 0) ;
#   * seule la bouche contribue, via la condition de Robin dP/dz = alpha*P
#     (normale sortante = -z, donc dP/dn = -alpha*P).
# Il reste l'identite EXACTE, complexe :
#
#       k^2 * Integrale_volume(P)  -  alpha * Integrale_bouche(P)  =  -Integrale_volume(F)
#
# Sans elle, le residu local ne fixe pas l'amplitude : le terme laplacien est
# ~42x plus sensible que k^2*P dans le gradient, si bien que l'optimiseur
# equilibre lap(P) ~ -F localement avec une amplitude minuscule, au lieu de
# la vraie branche k^2*P ~ -F.
AREA = R_NECK*L_NECK + R_CAV*H_CAV        # aire du plan meridien
def _J_volume():
    nr, nz = 600, 1200
    rr = np.linspace(0, R_CAV, nr); zz = np.linspace(0, Z_TOP, nz)
    RR, ZZ = np.meshgrid(rr, zz, indexing="ij")
    dom = ((ZZ < L_NECK) & (RR <= R_NECK)) | ((ZZ >= L_NECK) & (RR <= R_CAV))
    F = SRC_A*np.exp(-(RR**2 + (ZZ-SRC_Z)**2)/(2*SRC_W**2))
    # integrale de F*r sur le plan (le 2*pi se simplifie des deux cotes)
    return float(np.trapz(np.trapz(F*RR*dom, zz, axis=1), rr))
J_VOL = _J_volume()
print(f"integrale de la source : J = {J_VOL:.4f}  (ponderee par r)")

def loss_integral(n_v=4000, n_s=800):
    # --- integrale de volume : Monte-Carlo pondere par r ---
    rv, zv = sample_interior(n_v)
    Pr, Pi = P_of(Tn(rv), Tn(zv))
    rw = Tn(rv)
    IVr = (Pr*rw).mean()*AREA
    IVi = (Pi*rw).mean()*AREA
    # --- integrale sur la bouche : r de 0 a R_col, en z = 0 ---
    rs = np.random.uniform(0, R_NECK, n_s)
    Qr, Qi = P_of(Tn(rs), Tn(np.zeros(n_s)))
    rsw = Tn(rs)
    ISr = (Qr*rsw).mean()*R_NECK
    ISi = (Qi*rsw).mean()*R_NECK
    ar, ai = float(ALPHA.real), float(ALPHA.imag)
    # k^2*IV - alpha*IS + J = 0   (partie reelle et partie imaginaire)
    er = K2*IVr - (ar*ISr - ai*ISi) + J_VOL
    ei = K2*IVi - (ar*ISi + ai*ISr)
    ref = K2*SCALE*AREA*R_CAV
    return (er/ref)**2 + (ei/ref)**2

def loss_pde():
    rr, zz = sample_interior(NCOL)
    r = Tn(rr, 1); z = Tn(zz, 1)
    Pr, Pi = P_of(r, z)
    lr_, _, _ = lap_of(Pr, r, z)
    li_, _, _ = lap_of(Pi, r, z)
    F = SRC_A*torch.exp(-(r**2 + (z-SRC_Z)**2)/(2*SRC_W**2))
    res_r = (lr_ + K2*Pr + F)/NORM
    res_i = (li_ + K2*Pi)/NORM
    return torch.mean(res_r**2 + res_i**2)

def loss_bc():
    m = NCOL//5
    # --- parois rigides ---
    segs = [(np.full(m, R_NECK), np.random.uniform(0, L_NECK, m), 1., 0.),
            (np.random.uniform(R_NECK, R_CAV, m), np.full(m, L_NECK), 0., 1.),
            (np.full(m, R_CAV), np.random.uniform(L_NECK, Z_TOP, m), 1., 0.),
            (np.random.uniform(0, R_CAV, m), np.full(m, Z_TOP), 0., 1.)]
    rw = np.concatenate([s[0] for s in segs]); zw = np.concatenate([s[1] for s in segs])
    nx = np.concatenate([np.full(m, s[2]) for s in segs])
    ny = np.concatenate([np.full(m, s[3]) for s in segs])
    r = Tn(rw, 1); z = Tn(zw, 1)
    Pr, Pi = P_of(r, z)
    _, prr, pzr = lap_of(Pr, r, z)
    _, pri, pzi = lap_of(Pi, r, z)
    nxt = Tn(nx); nyt = Tn(ny)
    gscale = SCALE/R_NECK
    lw = torch.mean(((nxt*prr + nyt*pzr)/gscale)**2 + ((nxt*pri + nyt*pzi)/gscale)**2)

    # --- bouche : dP/dz = alpha * P  (Robin, complexe) ---
    rm = np.random.uniform(0, R_NECK, m); zm = np.zeros(m)
    r2 = Tn(rm, 1); z2 = Tn(zm, 1)
    Qr, Qi = P_of(r2, z2)
    _, _, qzr = lap_of(Qr, r2, z2)
    _, _, qzi = lap_of(Qi, r2, z2)
    ar, ai = float(ALPHA.real), float(ALPHA.imag)
    er = (qzr - (ar*Qr - ai*Qi))/gscale
    ei = (qzi - (ar*Qi + ai*Qr))/gscale
    return lw, torch.mean(er**2 + ei**2)

# --- comparaison au FDM sur une grille commune ---
RRr, ZZr = np.meshgrid(r_ref, z_ref, indexing="ij")
mask = fluid_ref
def compare():
    with torch.no_grad():
        Pr, Pi = P_of(Tn(RRr[mask]), Tn(ZZr[mask]))
    Pn = Pr.numpy().ravel() + 1j*Pi.numpy().ravel()
    Pr_ = P_ref[mask]
    return float(np.linalg.norm(Pn-Pr_)/np.linalg.norm(Pr_)), Pn

opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
print(f"\nentrainement : {ITERS} iterations, {NCOL} points")
t0 = time.time()
for it in range(1, ITERS+1):
    opt.zero_grad()
    lp = loss_pde(); lw, lm = loss_bc(); li = loss_integral()
    (lp + 5.0*lw + 5.0*lm + 20.0*li).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if it % max(1, ITERS//12) == 0 or it == 1:
        e, _ = compare()
        print(f"  it {it:5d} | PDE {lp.item():.3e} | integ {li.item():.2e} | bouche {lm.item():.2e} "
              f"| L2 vs FDM {e*100:6.2f} % | max|P| {np.nanmax(np.abs(_)):7.1f} | {time.time()-t0:5.0f}s")

err, Pn = compare()
print(f"\n=== VERDICT (harmonique, col ouvert) ===")
print(f"L2 relatif du champ complexe vs FDM : {err*100:.2f} %")
Pfull = np.full(mask.shape, np.nan, complex); Pfull[mask] = Pn
print(f"max|P| PINN = {np.nanmax(np.abs(Pfull)):.4f} Pa   (FDM {SCALE:.4f} Pa)")

torch.save({"state": model.state_dict(), "scale": SCALE, "hid": HID, "freq": FREQ}, f"models/pinn_harmonic{TAG}.pth")
np.savez_compressed(f"data/pinn_harmonic{TAG}.npz", r=r_ref, z=z_ref, mask=mask,
                    P_pinn=Pfull, P_fdm=P_ref, freq=FREQ, err=err, scale=SCALE)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
vm = np.nanmax(np.abs(P_ref))
for a, (D, ti) in zip(ax, [(np.abs(P_ref), "|P| FDM"), (np.abs(Pfull), "|P| PINN"),
                           (np.abs(Pfull-P_ref), "|erreur|")]):
    pm = a.pcolormesh(z_ref*1e3, r_ref*1e3, D, shading="auto", cmap="magma")
    a.set(xlabel="z (mm)", ylabel="r (mm)", title=ti); a.set_aspect("equal")
    fig.colorbar(pm, ax=a)
fig.suptitle(f"PINN harmonique a {FREQ} Hz — L2 = {err*100:.2f} %")
fig.tight_layout(); fig.savefig(f"plots/pinn_harmonic{TAG}.png", dpi=110)
print(f"Figure : plots/pinn_harmonic{TAG}.png")
