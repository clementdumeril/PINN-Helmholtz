"""
PINN PARAMETRE PAR LA FREQUENCE  :  reseau (r, z, f) -> P complexe.

Bascule de formulation apres l'echec documente du transitoire pur. Trois raisons
tiennent le choix :

 1. La resonance n'est plus une ACCUMULATION sur 12,6 periodes mais un systeme
    lineaire a chaque frequence. L'amplification d'erreur par Q -- qui exigeait
    un residu juste a ~1 % en transitoire -- disparait.
 2. L'equation reste de type ELLIPTIQUE : tous les points du domaine sont
    couples simultanement par un seul systeme lineaire, au lieu de dependre
    de leur passe. Il n'y a plus de causalite a respecter, donc plus de
    ponderation causale, et surtout plus d'accumulation d'erreur.
    (Correction : j'avais ecrit ici que les moindres carres devenaient
    "exacts et non plus approches". C'est faux -- avec l'ansatz a enveloppes,
    le residu transitoire etait DEJA affine dans les poids de sortie, donc les
    moindres carres y etaient exacts aussi. Ce n'est pas la difference.)
 3. Le terme source ne pose plus le probleme de mesure du transitoire : la
    gaussienne fait sigma = 1 cm dans une cavite de 4 x 8 cm, soit une fraction
    NOTABLE du domaine, contre 0,1 % de l'espace-temps auparavant. La
    degenerescence d'amplitude perd sa cause principale.

Geometrie et physique strictement celles de fdm_sweep.py : col + cavite,
impedance de rayonnement de piston bafle a la bouche, source gaussienne dans
la cavite. La reference est le solveur direct du meme fichier.

Env : PF_FMIN (209.84) PF_FMAX (209.84) PF_ITERS (1500) PF_CHECK (50)
      PF_NF (nb de frequences par lot, 8) PF_NCOL (1200) PF_HID (96)
      PF_LR (1e-3) PF_RCOND (1e-10, troncature SVD) PF_BETA (0.2)
      PF_NLS (2400, lot FIXE) PF_ACT (sine|tanh|silu) PF_W0 (8.0)
      PF_WINT (30.0, contrainte integrale) PF_NIQ (60) PF_SEED (0) PF_TAG ("")
"""
import os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import torch, torch.nn as nn
from torch.func import jacfwd, vmap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
SEED = int(os.environ.get("PF_SEED", 0))
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cpu"; torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)      # jacfwd construit ses tangentes en double

C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5*H_CAV, 0.01, 1.0e4

F_MIN  = float(os.environ.get("PF_FMIN", 209.84))
F_MAX  = float(os.environ.get("PF_FMAX", 209.84))
ITERS  = int(os.environ.get("PF_ITERS", 1500))
CHECK  = int(os.environ.get("PF_CHECK", 50))
N_F    = int(os.environ.get("PF_NF", 8))
N_COL  = int(os.environ.get("PF_NCOL", 1200))
HID    = int(os.environ.get("PF_HID", 96))
LR     = float(os.environ.get("PF_LR", 1e-3))
RCOND  = float(os.environ.get("PF_RCOND", 1e-10))   # troncature de la SVD
BETA   = float(os.environ.get("PF_BETA", 0.2))
TAG    = os.environ.get("PF_TAG", "")
MONO   = abs(F_MAX - F_MIN) < 1e-9
F_SPAN = max(F_MAX - F_MIN, 1e-9)
W_WALL  = float(os.environ.get("PF_WWALL", 5.0))
# A la RESONANCE, la solution de l'EDP n'est PAS unique : le probleme homogene
# A v = 0 admet le mode resonant comme solution non triviale, donc P et P + c v
# satisfont l'equation aussi bien l'une que l'autre. Ce qui rend l'amplitude
# finie -- 7351 Pa et non l'infini -- c'est UNIQUEMENT l'amortissement par
# rayonnement a la bouche. Cette condition n'est donc pas une contrainte de bord
# parmi d'autres : pres du pole, c'est ELLE qui fixe l'amplitude, et elle doit
# peser en consequence face aux 2400 points d'EDP.
W_MOUTH = float(os.environ.get("PF_WMOUTH", 5.0))

# aire du plan meridien : sert au tirage uniforme
A_NECK, A_CAV = R_NECK*L_NECK, R_CAV*H_CAV


def alpha_of(f):
    """Coefficient de Robin a la bouche : dP/dz = alpha*P (piston bafle)."""
    w = 2*np.pi*f; ka = w/C*R_NECK
    Zr = RHO*C*(0.5*ka**2 + 1j*(8/(3*np.pi))*ka)
    return 1j*w*RHO/Zr


# --------------------------------------------------------------------------
# reference : le solveur direct de fdm_sweep.py, h = 1 mm
# --------------------------------------------------------------------------
def fdm(freq, h=1e-3):
    w = 2*np.pi*freq; k2 = (w/C)**2; al = alpha_of(freq)
    Nr = int(round(R_CAV/h))+1; Nz = int(round(Z_TOP/h))+1
    r = np.arange(Nr)*h; z = np.arange(Nz)*h
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    fl = ((ZZ < L_NECK-1e-12) & (RR <= R_NECK+1e-12)) | ((ZZ >= L_NECK-1e-12) & (RR <= R_CAV+1e-12))
    mo = fl & (np.abs(ZZ) < 1e-12) & (RR <= R_NECK+1e-12)
    idx = lambda i, j: i + j*Nr
    N = Nr*Nz; ih2 = 1.0/h**2
    F = SRC_A*np.exp(-(RR**2 + (ZZ-SRC_Z)**2)/(2*SRC_W**2))
    rows, cols, dat = [], [], []; b = np.zeros(N, complex)
    for i in range(Nr):
        for j in range(Nz):
            ci = idx(i, j)
            if not fl[i, j]:
                rows += [ci]; cols += [ci]; dat += [1.0]; continue
            diag = k2 + 0j
            if i == 0:
                diag += -4*ih2; rows += [ci]; cols += [idx(1, j)]; dat += [4*ih2]
            else:
                lc = ih2*(1-0.5/i); rc = ih2*(1+0.5/i); diag += -2*ih2
                if fl[i-1, j]: rows += [ci]; cols += [idx(i-1, j)]; dat += [lc]
                else:          diag += lc
                if i+1 < Nr and fl[i+1, j]: rows += [ci]; cols += [idx(i+1, j)]; dat += [rc]
                else:                        diag += rc
            if mo[i, j]:
                diag += -2*ih2 - 2*al/h
                rows += [ci]; cols += [idx(i, j+1)]; dat += [2*ih2]
            else:
                diag += -2*ih2
                for jj in (j-1, j+1):
                    if 0 <= jj < Nz and fl[i, jj]:
                        rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                    else:
                        diag += ih2
            rows += [ci]; cols += [ci]; dat += [diag]
            b[ci] = -F[i, j]
    A = sp.coo_matrix((dat, (rows, cols)), shape=(N, N), dtype=complex).tocsr()
    return r, z, fl, spsolve(A, b).reshape((Nz, Nr)).T


t0 = time.time()
F_REF = np.unique(np.round(np.linspace(F_MIN, F_MAX, 1 if MONO else 9), 4))
REF = {}
for f in F_REF:
    r_ref, z_ref, fl_ref, P = fdm(f)
    REF[float(f)] = np.where(fl_ref, P, 0.0)
SCALE = float(max(np.abs(v).max() for v in REF.values()))
print(f"reference FDM : {len(REF)} frequence(s), {fl_ref.sum()} noeuds, {time.time()-t0:.1f} s")
print(f"               max|P| = {SCALE:.2f} Pa" + ("" if MONO else "  (sur la bande)"))
NORM = SRC_A          # echelle du residu : le forcage, connu d'avance


# --------------------------------------------------------------------------
# reseau
# --------------------------------------------------------------------------
# Activation : mesuree, pas devinee. Le banc bench_bases.py compare les bases a
# initialisation aleatoire, sans entrainement, sur deux criteres : la capacite
# (erreur de projection de la solution FDM) et le CONDITIONNEMENT de la matrice
# de physique A = lap + k^2, qui est celle qu'on resout reellement.
#
#   base            projection   kappa(A_phys)   rang
#   tanh                0,36 %        6,8e13     80/96
#   SiLU / Swish        0,57 %        2,6e16     43/96   <- pire que tanh
#   sine w0 = 5         0,16 %        1,1e4      96/96
#   sine w0 = 8         0,26 %        5,1e2      96/96   <- retenu
#   sine w0 = 10        0,35 %        1,5e2      96/96
#   sine w0 = 20        5,10 %        1,7e1      96/96   <- sur-oscille
#
# Onze ordres de grandeur de conditionnement en moins, a capacite egale ou
# meilleure. Au-dela de w0 = 12 la capacite s'effondre : le champ est
# sous-longueur d'onde (lambda = 1,63 m pour 0,12 m), imposer des oscillations
# rapides detruit la representation.
ACT = os.environ.get("PF_ACT", "sine")          # "sine" | "tanh" | "silu"
W0  = float(os.environ.get("PF_W0", 8.0))


class Sine(nn.Module):
    def __init__(self, w0): super().__init__(); self.w0 = w0
    def forward(self, x): return torch.sin(self.w0*x)


class Net(nn.Module):
    def __init__(self, hidden=96, layers=5):
        super().__init__()
        mods, d = [], 3
        for i in range(layers):
            lin = nn.Linear(d, hidden)
            if ACT == "sine":
                # initialisation SIREN : uniforme en 1/d a la premiere couche,
                # en sqrt(6/d)/w0 ensuite, pour garder les activations bornees.
                with torch.no_grad():
                    b = (1.0/d) if i == 0 else (np.sqrt(6.0/d)/W0)
                    lin.weight.uniform_(-b, b); lin.bias.uniform_(-b, b)
            act = Sine(W0) if ACT == "sine" else (nn.Tanh() if ACT == "tanh" else nn.SiLU())
            mods += [lin, act]
            d = hidden
        mods += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*mods)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05); self.net[-1].bias.zero_()
    def forward(self, x): return self.net(x)


model = Net(HID).to(DEV)
HEAD, OUT = model.net[:-1], model.net[-1]
D_TH = HID + 1


def Tn(a, g=False):
    return torch.tensor(np.asarray(a), dtype=torch.float64, device=DEV).reshape(-1, 1).requires_grad_(bool(g))


def _xin(r, z, f):
    return torch.cat([r/R_CAV, z/Z_TOP, (f - F_MIN)/F_SPAN], 1)


def P_of(r, z, f):
    o = model(_xin(r, z, f))
    return SCALE*o[:, 0:1], SCALE*o[:, 1:2]


# --------------------------------------------------------------------------
# echantillonnage : col + cavite, et N_F frequences par lot
# --------------------------------------------------------------------------
def sample_dom(n):
    k = np.random.rand(n) < A_NECK/(A_NECK + A_CAV)
    r = np.where(k, np.random.uniform(0, R_NECK, n), np.random.uniform(0, R_CAV, n))
    z = np.where(k, np.random.uniform(0, L_NECK, n), np.random.uniform(L_NECK, Z_TOP, n))
    return r, z


def sample_f(n):
    return np.full(n, F_MIN) if MONO else np.random.uniform(F_MIN, F_MAX, n)


WALLS = [
    (lambda m: (np.full(m, R_NECK), np.random.uniform(0, L_NECK, m)), 1., 0.),
    (lambda m: (np.random.uniform(R_NECK, R_CAV, m), np.full(m, L_NECK)), 0., 1.),
    (lambda m: (np.full(m, R_CAV), np.random.uniform(L_NECK, Z_TOP, m)), 1., 0.),
    (lambda m: (np.random.uniform(0, R_CAV, m), np.full(m, Z_TOP)), 0., 1.),
]


def forcing(r, z):
    return SRC_A*torch.exp(-(r**2 + (z - SRC_Z)**2)/(2*SRC_W**2))


# --------------------------------------------------------------------------
# residus (descente de gradient)
# --------------------------------------------------------------------------
def _lap(P, r, z):
    pr = torch.autograd.grad(P, r, torch.ones_like(P), create_graph=True)[0]
    pz = torch.autograd.grad(P, z, torch.ones_like(P), create_graph=True)[0]
    prr = torch.autograd.grad(pr, r, torch.ones_like(pr), create_graph=True)[0]
    pzz = torch.autograd.grad(pz, z, torch.ones_like(pz), create_graph=True)[0]
    inv = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    return prr + torch.where(r < 1e-6, prr, pr*inv) + pzz, pr, pz


def loss_pde(n=None):
    n = n or N_COL
    rn, zn = sample_dom(n); fn = sample_f(n)
    r = Tn(rn, 1); z = Tn(zn, 1); f = Tn(fn)
    Pr, Pi = P_of(r, z, f)
    k2 = (2*np.pi*f/C)**2
    lr_, _, _ = _lap(Pr, r, z)
    li_, _, _ = _lap(Pi, r, z)
    F = forcing(r, z)
    return torch.mean(((lr_ + k2*Pr + F)/NORM)**2 + ((li_ + k2*Pi)/NORM)**2)


def loss_walls(m=None):
    m = m or max(16, N_COL//8)
    rs, zs, nx, ny = [], [], [], []
    for g, a, b in WALLS:
        u, v = g(m); rs.append(u); zs.append(v)
        nx.append(np.full(m, a)); ny.append(np.full(m, b))
    r = Tn(np.concatenate(rs), 1); z = Tn(np.concatenate(zs), 1)
    f = Tn(sample_f(4*m))
    nxv = Tn(np.concatenate(nx)); nyv = Tn(np.concatenate(ny))
    Pr, Pi = P_of(r, z, f)
    out = 0.0
    for P in (Pr, Pi):
        pr = torch.autograd.grad(P, r, torch.ones_like(P), create_graph=True)[0]
        pz = torch.autograd.grad(P, z, torch.ones_like(P), create_graph=True)[0]
        out = out + torch.mean(((nxv*pr + nyv*pz)/(SCALE/R_NECK))**2)
    return out


def loss_mouth(m=None):
    m = m or max(16, N_COL//8)
    rn = np.random.uniform(0, R_NECK, m); zn = np.zeros(m); fn = sample_f(m)
    r = Tn(rn, 1); z = Tn(zn, 1); f = Tn(fn)
    Pr, Pi = P_of(r, z, f)
    pzr = torch.autograd.grad(Pr, z, torch.ones_like(Pr), create_graph=True)[0]
    pzi = torch.autograd.grad(Pi, z, torch.ones_like(Pi), create_graph=True)[0]
    al = alpha_of(fn.reshape(-1, 1))
    ar = Tn(al.real); ai = Tn(al.imag)
    er = pzr - (ar*Pr - ai*Pi)
    ei = pzi - (ar*Pi + ai*Pr)
    sc = SCALE/R_NECK
    return torch.mean((er/sc)**2 + (ei/sc)**2)


# --------------------------------------------------------------------------
# CONTRAINTE INTEGRALE EXACTE  --  la seule chose NON HOMOGENE du systeme
# --------------------------------------------------------------------------
# Constat qui commande tout : residu d'EDP, parois et condition de Robin a la
# bouche sont TOUS homogenes en P. P = 0 les satisfait exactement, et les
# renforcer ne fait que pousser plus fort vers zero (mesure : passer le poids
# de bouche de 5 a 5000 fait TOMBER l'amplitude de 3,3 a 0,7 Pa).
#
# Pire, avec une base bien conditionnee l'optimiseur atteint plus facilement la
# MAUVAISE BRANCHE : un champ de 3 Pa variant sur sigma = 1 cm donne
# lap(P) ~ 3e4, largement de quoi annuler F = 1e4. Il equilibre donc
# lap(P) ~ -F localement, a amplitude minuscule, au lieu de la vraie branche
# k^2 P ~ -F. Mesure : phys = 1e-3 avec max|P| = 3 Pa contre 7351 attendus.
#
# Seul le terme SOURCE echappe a l'homogeneite. En integrant l'equation sur le
# domaine, le laplacien se reduit au flux de bord ; les parois rigides ne
# donnent rien, et la bouche s'exprime par la condition de Robin (normale
# sortante en -z, donc dP/dn = -alpha P) :
#
#        k^2 * INT_vol P dV  -  alpha * INT_bouche P dS  +  J  =  0
#
# avec J = INT F dV, NON NUL par construction. A P = 0 le residu vaut J : la
# contrainte est violee au maximum, et elle selectionne la branche.
#
# Normalisation par J, precisement : dans train_pinn_harmonic.py la meme
# identite etait normalisee 76 fois trop haut, si bien qu'a champ nul elle ne
# pesait que 3,5e-3 malgre un poids affiche de 20. Elle etait sans dents.
# Ici, residu au champ nul = 1 exactement.
#
# Quadrature FIXE, jamais Monte-Carlo : une contrainte integrale bruitee ne
# contraint rien.
W_INT = float(os.environ.get("PF_WINT", 30.0))
N_IQ  = int(os.environ.get("PF_NIQ", 60))       # finesse de la quadrature
N_IF  = int(os.environ.get("PF_NIF", 5))        # frequences contraintes


def _quad():
    """Grille fixe ponderee par r, sur le col et la cavite, plus la bouche."""
    def grid(r1, z0, z1, nr, nz):
        rr = (np.arange(nr) + 0.5)*r1/nr
        zz = z0 + (np.arange(nz) + 0.5)*(z1 - z0)/nz
        RR, ZZ = np.meshgrid(rr, zz, indexing="ij")
        w = RR*(r1/nr)*((z1 - z0)/nz)           # le 2 pi se simplifie des deux cotes
        return RR.ravel(), ZZ.ravel(), w.ravel()
    n1 = max(8, N_IQ//3)
    r1, z1, w1 = grid(R_NECK, 0.0, L_NECK, n1, N_IQ)
    r2, z2, w2 = grid(R_CAV, L_NECK, Z_TOP, N_IQ, N_IQ)
    rv = np.concatenate([r1, r2]); zv = np.concatenate([z1, z2])
    wv = np.concatenate([w1, w2])
    rm = (np.arange(N_IQ) + 0.5)*R_NECK/N_IQ    # bouche : z = 0
    wm = rm*(R_NECK/N_IQ)
    return rv, zv, wv, rm, wm


IQ_R, IQ_Z, IQ_W, IQ_MR, IQ_MW = _quad()
J_VOL = float((SRC_A*np.exp(-(IQ_R**2 + (IQ_Z - SRC_Z)**2)/(2*SRC_W**2))*IQ_W).sum())
IQ_F = np.array([F_MIN]) if MONO else np.linspace(F_MIN, F_MAX, N_IF)
print(f"contrainte integrale | {IQ_R.size} pts de volume, {IQ_MR.size} a la bouche "
      f"| J = {J_VOL:.5f} | {IQ_F.size} frequence(s)")


def integral_loss():
    """Residu de l'identite integrale, normalise par J -> vaut 1 a champ nul."""
    out = 0.0
    for f in IQ_F:
        n = IQ_R.size; m = IQ_MR.size
        pr, pi = P_of(Tn(IQ_R), Tn(IQ_Z), Tn(np.full(n, f)))
        w = Tn(IQ_W)
        IVr = (w*pr).sum(); IVi = (w*pi).sum()
        mr, mi = P_of(Tn(IQ_MR), Tn(np.zeros(m)), Tn(np.full(m, f)))
        wm = Tn(IQ_MW)
        ISr = (wm*mr).sum(); ISi = (wm*mi).sum()
        al = alpha_of(f); ar, ai = float(al.real), float(al.imag)
        k2 = (2*np.pi*f/C)**2
        er = k2*IVr - (ar*ISr - ai*ISi) + J_VOL
        ei = k2*IVi - (ar*ISi + ai*ISr)
        out = out + (er/J_VOL)**2 + (ei/J_VOL)**2
    return out/len(IQ_F)


# --------------------------------------------------------------------------
# COUCHE DE SORTIE PAR MOINDRES CARRES  --  ici EXACTE
# --------------------------------------------------------------------------
# En harmonique l'equation est lineaire en P, et la derniere couche l'est aussi.
# Le residu est donc AFFINE dans les poids de sortie theta : min ||A.theta - b||^2
# est un moindres carres ordinaire, resolu en un coup. Contrairement au
# transitoire, aucune approximation n'intervient.
#
# theta est range en (2, D_TH) : ligne 0 -> partie reelle, ligne 1 -> imaginaire.
# L'operateur lap + k^2 est REEL, donc il ne melange pas les deux ; seule la
# condition de Robin a la bouche les couple, via alpha complexe.
def _hfun(u):
    return HEAD(torch.stack([u[0]/R_CAV, u[1]/Z_TOP, (u[2] - F_MIN)/F_SPAN]))


_jac  = vmap(jacfwd(_hfun))
_hess = vmap(jacfwd(jacfwd(_hfun)))


def _U(rn, zn, fn):
    return torch.stack([Tn(rn).squeeze(1), Tn(zn).squeeze(1), Tn(fn).squeeze(1)], 1)


def _basis(rn, zn, fn, need_lap=False):
    U = _U(rn, zn, fn)
    h = HEAD(torch.stack([U[:, 0]/R_CAV, U[:, 1]/Z_TOP, (U[:, 2] - F_MIN)/F_SPAN], 1))
    M = SCALE*torch.cat([h, torch.ones_like(h[:, :1])], 1)
    if not need_lap:
        J = _jac(U)
        return M, SCALE*torch.cat([J[:, :, 0], torch.zeros_like(h[:, :1])], 1), \
                  SCALE*torch.cat([J[:, :, 1], torch.zeros_like(h[:, :1])], 1)
    J = _jac(U); H2 = _hess(U)
    hr = J[:, :, 0]; hrr = H2[:, :, 0, 0]; hzz = H2[:, :, 1, 1]
    r = U[:, 0:1]
    inv = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    lap = hrr + torch.where(r < 1e-6, hrr, hr*inv) + hzz
    Z = torch.zeros_like(h[:, :1])
    return M, SCALE*torch.cat([lap, Z], 1)


# Jeu de collocation FIXE pour les moindres carres. A la resonance l'operateur
# est quasi singulier -- c'est exactement pourquoi l'amplitude explose -- donc
# le systeme normal est mal conditionne et theta devient hypersensible au
# TIRAGE. Mesure : a ridge = 1e-14, l'amplitude atteint 7699 Pa (4,7 % du FDM)
# puis retombe a 167 en trente iterations. On fige donc les points : theta
# redevient une fonction deterministe des features cachees.
def _fixed_set():
    st = np.random.get_state(); np.random.seed(20250824)
    m = max(16, N_LS//8)
    rs, zs, nx, ny = [], [], [], []
    for g, a, b in WALLS:
        u, v = g(m); rs.append(u); zs.append(v)
        nx.append(np.full(m, a)); ny.append(np.full(m, b))
    d = dict(dom=sample_dom(N_LS), fdom=sample_f(N_LS),
             rw=np.concatenate(rs), zw=np.concatenate(zs),
             nx=np.concatenate(nx), ny=np.concatenate(ny), fw=sample_f(4*m),
             rm=np.random.uniform(0, R_NECK, m), fm=sample_f(m), m=m)
    np.random.set_state(st)
    return d


N_LS = int(os.environ.get("PF_NLS", 2400))
FIX = _fixed_set()


def ls_step():
    Z2 = lambda X: torch.zeros_like(X)
    rows, rhs = [], []

    # --- EDP : deux blocs decouples (l'operateur est reel) ---
    rn, zn = FIX["dom"]; fn = FIX["fdom"]
    with torch.no_grad():
        M, L = _basis(rn, zn, fn, need_lap=True)
        k2 = Tn((2*np.pi*fn/C)**2)
        A = (L + k2*M)/NORM
        F = forcing(Tn(rn), Tn(zn))
        rows += [torch.cat([A, Z2(A)], 1), torch.cat([Z2(A), A], 1)]
        rhs  += [-F/NORM, torch.zeros_like(F)]

        # --- parois : flux nul sur les deux parties ---
        m = FIX["m"]
        rw, zw, fw = FIX["rw"], FIX["zw"], FIX["fw"]
        _, Mr, Mz = _basis(rw, zw, fw)
        G = (Tn(FIX["nx"])*Mr + Tn(FIX["ny"])*Mz)/(SCALE/R_NECK)
        kw = np.sqrt(W_WALL)
        rows += [kw*torch.cat([G, Z2(G)], 1), kw*torch.cat([Z2(G), G], 1)]
        rhs  += [torch.zeros(G.shape[0], 1), torch.zeros(G.shape[0], 1)]

        # --- bouche : Robin, seul endroit ou reel et imaginaire se couplent ---
        rm = FIX["rm"]; zm = np.zeros(m); fm = FIX["fm"]
        Mm, _, Mmz = _basis(rm, zm, fm)
        al = alpha_of(fm.reshape(-1, 1))
        ar, ai = Tn(al.real), Tn(al.imag)
        sc = SCALE/R_NECK; km = np.sqrt(W_MOUTH)
        rows += [km*torch.cat([(Mmz - ar*Mm)/sc, (ai*Mm)/sc], 1),
                 km*torch.cat([(-ai*Mm)/sc, (Mmz - ar*Mm)/sc], 1)]
        rhs  += [torch.zeros(m, 1), torch.zeros(m, 1)]

        # --- contrainte integrale : deux lignes par frequence, second membre
        #     NON NUL. C'est le seul bloc du systeme qui interdise P = 0.
        for f in IQ_F:
            n = IQ_R.size; m = IQ_MR.size
            Mv, _, _ = _basis(IQ_R, IQ_Z, np.full(n, f))
            Mm, _, _ = _basis(IQ_MR, np.zeros(m), np.full(m, f))
            wv = Tn(IQ_W); wm = Tn(IQ_MW)
            Iv = (wv*Mv).sum(0, keepdim=True)        # (1, D_TH)
            Is = (wm*Mm).sum(0, keepdim=True)
            al = alpha_of(f); ar, ai = float(al.real), float(al.imag)
            k2 = (2*np.pi*f/C)**2
            Rr = k2*Iv - ar*Is
            ki = np.sqrt(W_INT)/J_VOL
            rows += [ki*torch.cat([Rr,  ai*Is], 1),
                     ki*torch.cat([-ai*Is, Rr], 1)]
            rhs  += [ki*torch.full((1, 1), -J_VOL), torch.zeros(1, 1)]

        A = torch.cat(rows, 0); b = torch.cat(rhs, 0)
        # SVD TRONQUEE, pas Tikhonov. Mesure sur la base non entrainee : elle
        # represente le mode a 0,40 % de L2, mais il y faut ||theta|| = 9,3e8,
        # pour un conditionnement de 3,55e16 et un rang effectif de 57 sur 97 --
        # les features tanh sont quasi lineairement dependantes. Une penalite
        # ridge ~ ||theta||^2 ecrase donc mecaniquement la solution : elle ne
        # combattait pas la resonance, elle combattait le conditionnement.
        # La SVD tronquee ecarte les directions degenerees SANS biaiser celles
        # qu'on garde. On normalise d'abord les colonnes, ce qui retire la part
        # du mauvais conditionnement due aux seules echelles.
        cn = A.norm(dim=0).clamp(min=1e-300)
        sol = torch.linalg.lstsq(A/cn, b, rcond=RCOND, driver="gelsd").solution
        th = (sol.squeeze(-1)/cn).reshape(2, D_TH)
        if not torch.isfinite(th).all():
            return False
        cur = torch.cat([OUT.weight, OUT.bias.reshape(2, 1)], 1)
        mix = (1.0 - BETA)*cur + BETA*th
        OUT.weight.copy_(mix[:, :HID]); OUT.bias.copy_(mix[:, HID])
    return True


# --------------------------------------------------------------------------
# verification et selection
# --------------------------------------------------------------------------
FLAT = np.array([[rr, zz] for rr, zz in zip(*np.where(fl_ref))], dtype=float)
RG = r_ref[FLAT[:, 0].astype(int)]; ZG = z_ref[FLAT[:, 1].astype(int)]


def l2_at(f):
    Pref = REF[f][fl_ref]
    with torch.no_grad():
        pr, pi = P_of(Tn(RG), Tn(ZG), Tn(np.full(RG.size, f)))
    Pn = pr.numpy().ravel() + 1j*pi.numpy().ravel()
    return float(np.linalg.norm(Pn - Pref)/np.linalg.norm(Pref)), float(np.abs(Pn).max())


VAL_SEED = 4242


def phys_val():
    st = np.random.get_state(); np.random.seed(VAL_SEED)
    v = float((loss_pde() + W_WALL*loss_walls() + W_MOUTH*loss_mouth()
               + W_INT*integral_loss()).detach())
    np.random.set_state(st)
    return v


opt = torch.optim.Adam(HEAD.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
print(f"bande {F_MIN:.2f} - {F_MAX:.2f} Hz | {ITERS} iterations | "
      f"{'monofrequence' if MONO else str(N_F)+' freq/lot'} | scale = {SCALE:.1f} Pa")

best, best_state, hist = np.inf, None, []
t0 = time.time()
for it in range(1, ITERS+1):
    opt.zero_grad()
    if not ls_step():
        print(f"  it {it}: moindres carres non finis, pas ignore")
    (loss_pde() + W_WALL*loss_walls() + W_MOUTH*loss_mouth()
     + W_INT*integral_loss()).backward()
    torch.nn.utils.clip_grad_norm_(HEAD.parameters(), 1.0)
    opt.step(); sched.step()

    if it % CHECK == 0:
        phys = phys_val()
        es = [l2_at(f) for f in REF]
        e = float(np.mean([a for a, _ in es])); amp = es[len(es)//2][1]
        if phys < best:
            best = phys
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            flag = "  "
        hist.append((it, phys, e, amp))
        li = float(integral_loss().detach())
        print(f"  it {it:5d} | phys={phys:.3e}{flag}| Int={li:.2e} | L2 = {e*100:7.2f} % | "
              f"max|P| = {amp:9.2f} (FDM {SCALE:9.2f}) | {time.time()-t0:5.0f}s")

if best_state is not None:
    model.load_state_dict(best_state)
print(f"\nmodele retenu : perte physique = {best:.3e} (selection sans reference)")

res = {f: l2_at(f) for f in REF}
print("\n=== VERDICT (harmonique parametre) ===")
for f, (e, a) in res.items():
    print(f"  f = {f:8.2f} Hz | L2 = {e*100:7.2f} % | max|P| = {a:9.2f} Pa "
          f"(FDM {np.abs(REF[f]).max():9.2f})")
np.savez(f"data/pinn_freq{TAG}.npz", hist=np.array(hist),
         freqs=np.array(list(res.keys())), l2=np.array([v[0] for v in res.values()]),
         amp=np.array([v[1] for v in res.values()]),
         amp_ref=np.array([np.abs(REF[f]).max() for f in res]))
torch.save({"state": model.state_dict(), "scale": SCALE, "hid": HID,
            "fmin": F_MIN, "fmax": F_MAX}, f"models/pinn_freq{TAG}.pth")

fs = np.array(list(res.keys()))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
h = np.array(hist)
ax[0].semilogy(h[:, 0], h[:, 2]*100, "C1", lw=1.3)
ax[0].axhline(100, color="0.5", ls="--", lw=1, label="score du champ nul")
ax[0].set(xlabel="itération", ylabel="L2 (%)", title="convergence")
ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].plot(fs, [np.abs(REF[f]).max() for f in fs], "ko-", lw=1.4, ms=5, label="FDM")
ax[1].plot(fs, [res[f][1] for f in fs], "C1s--", lw=1.2, ms=4, label="PINN")
ax[1].set(xlabel="fréquence (Hz)", ylabel="max |P| (Pa)", title="réponse en fréquence")
ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"plots/pinn_freq{TAG}.png", dpi=115)
print(f"Figure : plots/pinn_freq{TAG}.png")
