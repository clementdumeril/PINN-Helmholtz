"""
TREFFTZ PAR SOUS-DOMAINE + RACCORD MODAL  (etapes 1 et 2)

Aboutissement du diagnostic : toutes les bases lisses -- tanh, sine, Gabor, et
la decomposition de domaine -- representent le champ a moins de 1 % mais
produisent un residu d'EDP de 7 000 a 1 000 000 %, parce que le laplacien
amplifie l'erreur de representation par 1/L^2 = 625. Le minimum des moindres
carres n'est donc pas pres de la verite.

Trefftz supprime ce mecanisme a la racine : chaque fonction de base satisfait
lap(phi) + k^2 phi = 0 EXACTEMENT. Le residu d'EDP ne peut plus venir de
l'approximation.

La decomposition en zones -- l'intuition de depart -- devient ici la bonne
architecture, parce qu'on choisit les modes RADIAUX adaptes a chaque zone :

    col    : J0(j'_(0,m) r / R_col)   ->  la paroi du col est exacte
    cavite : J0(j'_(0,n) r / R_cav)   ->  la paroi laterale est exacte

et la partie axiale de la cavite est choisie pour que le FOND soit exact aussi.

Il ne reste alors que quatre conditions, toutes sur des bords :
    1. bouche      z=0,   r<=R_col   : Robin, dP/dz = alpha P
    2. raccord     z=L,   r<=R_col   : continuite de P
    3. raccord     z=L,   r<=R_col   : continuite de dP/dz
    4. epaulement  z=L,   R_col<r<=R_cav : paroi rigide, dP/dz = 0

Le probleme de volume (14 721 inconnues au FDM) devient un probleme de bord
d'environ 60 inconnues. C'est le mode matching, methode classique du
resonateur de Helmholtz.

Env : TM_MN (modes de col, 14) TM_NC (modes de cavite, 22) TM_NB (points de
      collocation par bord, 90) TM_F (209.84) TM_SWEEP (0/1)
"""
import os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import jn_zeros, j0, j1, erf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5*H_CAV, 0.01, 1.0e4

M_N = int(os.environ.get("TM_MN", 14))     # modes radiaux du col
N_C = int(os.environ.get("TM_NC", 22))     # modes radiaux de la cavite
NB  = int(os.environ.get("TM_NB", 90))     # points de collocation par bord
FREQ0 = float(os.environ.get("TM_F", 209.84))

KAP = jn_zeros(1, M_N)/R_NECK              # j'_(0,m) = zeros de J1 -> dJ0/dr = 0
KAP = np.concatenate([[0.0], KAP])[:M_N]
MU = jn_zeros(1, N_C)/R_CAV
MU = np.concatenate([[0.0], MU])[:N_C]


def alpha_of(f):
    w = 2*np.pi*f; ka = w/C*R_NECK
    Zr = RHO*C*(0.5*ka**2 + 1j*(8/(3*np.pi))*ka)
    return 1j*w*RHO/Zr


# --------------------------------------------------------------------------
# SOLUTION PARTICULIERE  --  developpee dans les MODES DE LA CAVITE
# --------------------------------------------------------------------------
# Premiere tentative, ecartee : la solution de Poisson en espace libre. Elle
# porte bien la source, mais elle ne satisfait NI la paroi laterale NI le fond.
# Or chaque mode homogene a dphi/dr = 0 exactement : aucun ne peut corriger
# cette violation. Le systeme etait structurellement incapable de la rattraper,
# d'ou un champ 140x trop petit malgre un raccord descendu a 4,8 %.
#
# La construction correcte developpe la source sur la base radiale de la cavite,
#
#     F(r,z) = somme_n  f_n(z) J0(mu_n r) ,
#
# puis resout, pour chaque n, l'equation differentielle a une dimension
#
#     Y_n'' - gamma_n^2 Y_n = -f_n(z)  ,   Y_n'(H_cav) = 0 ,  Y_n(0) = 0.
#
# Comme J0(mu_n r) satisfait la paroi par construction, P_part = somme Y_n J0
# la satisfait AUSSI -- et le fond avec. Il ne reste alors vraiment que le
# raccord et la bouche.
def _radial_weights():
    """g_n : projection du profil radial de la source sur J0(mu_n r)."""
    nq = 4000
    rq = (np.arange(nq) + 0.5)*R_CAV/nq
    prof = np.exp(-rq**2/(2*SRC_W**2))
    g = []
    for mn in MU:
        Jn = j0(mn*rq)
        num = np.trapezoid(prof*Jn*rq, rq)
        den = np.trapezoid(Jn**2*rq, rq)
        g.append(SRC_A*num/den)
    return np.array(g)


G_N = _radial_weights()
NZQ = 1200
ZQ = np.linspace(0.0, H_CAV, NZQ)          # d = z - L_NECK
DZQ = ZQ[1] - ZQ[0]
SRC_Z_LOC = SRC_Z - L_NECK


def _solve_Y(k):
    """Resout les NZQ x 1 problemes aux limites, un par mode radial."""
    prof = np.exp(-(ZQ - SRC_Z_LOC)**2/(2*SRC_W**2))
    Y = np.zeros((len(MU), NZQ))
    for n, mn in enumerate(MU):
        lam = mn**2 - k**2                        # Y'' - lam Y = -f
        A = np.zeros((NZQ, NZQ)); b = -G_N[n]*prof.copy()
        A[0, 0] = 1.0; b[0] = 0.0                 # Y(0) = 0
        for i in range(1, NZQ-1):
            A[i, i-1] = 1/DZQ**2; A[i, i] = -2/DZQ**2 - lam; A[i, i+1] = 1/DZQ**2
        A[-1, -1] = 1.5/DZQ; A[-1, -2] = -2.0/DZQ; A[-1, -3] = 0.5/DZQ
        b[-1] = 0.0                               # Y'(H_cav) = 0, ordre 2
        Y[n] = np.linalg.solve(A, b)
    return Y


_YCACHE = {}


def _Y(k):
    key = round(float(k), 10)
    if key not in _YCACHE:
        _YCACHE[key] = _solve_Y(k)
    return _YCACHE[key]


def p_part(r, z, k):
    d = np.clip(z - L_NECK, 0.0, H_CAV)
    Y = _Y(k)
    out = np.zeros(np.shape(r), float)
    for n, mn in enumerate(MU):
        out = out + j0(mn*r)*np.interp(d, ZQ, Y[n])
    return out


def dp_part_dz(r, z, k):
    d = np.clip(z - L_NECK, 0.0, H_CAV)
    Y = _Y(k)
    out = np.zeros(np.shape(r), float)
    for n, mn in enumerate(MU):
        dY = np.gradient(Y[n], DZQ)
        out = out + j0(mn*r)*np.interp(d, ZQ, dY)
    return out


# --------------------------------------------------------------------------
# BASES DE TREFFTZ  --  formes exponentielles stables (jamais de cosh qui deborde)
# --------------------------------------------------------------------------
def neck_basis(r, z, k):
    """Col : J0(kappa r) * {deux solutions axiales bornees}. dP/dr = 0 en R_col."""
    cols, dcols = [], []
    for km in KAP:
        d = km**2 - k**2
        Jr = j0(km*r)
        if d < 0:
            b = np.sqrt(-d)
            for fz, dfz in ((np.cos(b*z), -b*np.sin(b*z)),
                            (np.sin(b*z),  b*np.cos(b*z))):
                cols.append(Jr*fz); dcols.append(Jr*dfz)
        else:
            g = np.sqrt(d)
            # exp(-g z) et exp(-g (L-z)) : bornees par 1 sur [0, L]
            e1 = np.exp(-g*z);              d1 = -g*e1
            e2 = np.exp(-g*(L_NECK - z));   d2 =  g*e2
            cols += [Jr*e1, Jr*e2]; dcols += [Jr*d1, Jr*d2]
    return np.stack(cols, 1), np.stack(dcols, 1)


def cav_basis(r, z, k):
    """Cavite : J0(mu r) * fonction axiale a derivee NULLE au fond z = Z_TOP.

    dP/dr = 0 en R_cav ET dP/dz = 0 en Z_TOP, tous deux exacts par construction.
    """
    dz = z - L_NECK                      # 0 a l'interface, H_CAV au fond
    cols, dcols = [], []
    for mn in MU:
        d = mn**2 - k**2
        Jr = j0(mn*r)
        if d < 0:
            b = np.sqrt(-d)
            fz = np.cos(b*(H_CAV - dz))/np.cos(b*H_CAV)
            dfz = b*np.sin(b*(H_CAV - dz))/np.cos(b*H_CAV)
        else:
            g = np.sqrt(d)
            # cosh(g(H-dz))/cosh(gH), ecrit de facon stable
            num = np.exp(-g*dz) + np.exp(-g*(2*H_CAV - dz))
            den = 1.0 + np.exp(-2*g*H_CAV)
            fz = num/den
            dfz = g*(-np.exp(-g*dz) + np.exp(-g*(2*H_CAV - dz)))/den
        cols.append(Jr*fz); dcols.append(Jr*dfz)
    return np.stack(cols, 1), np.stack(dcols, 1)


# --------------------------------------------------------------------------
# ETAPE 1 : verifier que le residu d'EDP est nul a la precision machine
# --------------------------------------------------------------------------
def verif_residu(k):
    """lap(phi) + k^2 phi, par differences finies d'ordre 2 sur la base."""
    h = 2e-5
    r = np.array([0.005, 0.02, 0.03]); z = np.array([0.06, 0.08, 0.10])
    def P(rr, zz): return cav_basis(rr, zz, k)[0]
    lap = ((P(r+h, z) - 2*P(r, z) + P(r-h, z))/h**2
           + (P(r+h, z) - P(r-h, z))/(2*h)/r[:, None]
           + (P(r, z+h) - 2*P(r, z) + P(r, z-h))/h**2)
    res = lap + k**2*P(r, z)
    ech = np.abs(k**2*P(r, z)).max()
    return np.abs(res).max()/ech


# --------------------------------------------------------------------------
# ETAPE 2 : le raccord, resolu par moindres carres
# --------------------------------------------------------------------------
def solve_match(freq, verbose=False):
    k = 2*np.pi*freq/C
    al = alpha_of(freq)
    nN = neck_basis(np.array([0.0]), np.array([0.0]), k)[0].shape[1]
    nC = cav_basis(np.array([0.0]), np.array([L_NECK]), k)[0].shape[1]

    rows, rhs = [], []

    def add(An, Ac, b):
        Z = np.zeros((An.shape[0], nC)) if Ac is None else Ac
        A_ = np.zeros((An.shape[0], nN)) if An is None else An
        rows.append(np.concatenate([A_, Z], 1)); rhs.append(b)

    # 1) bouche : dP/dz - alpha P = 0     (z = 0, r <= R_col)
    rb = (np.arange(NB) + 0.5)*R_NECK/NB
    Mb, Db = neck_basis(rb, np.zeros(NB), k)
    add(Db - al*Mb, None, np.zeros(NB, complex))

    # 2) raccord, continuite de P         (z = L, r <= R_col)
    ri = (np.arange(NB) + 0.5)*R_NECK/NB
    Mn, Dn = neck_basis(ri, np.full(NB, L_NECK), k)
    Mc, Dc = cav_basis(ri, np.full(NB, L_NECK), k)
    rows.append(np.concatenate([Mn, -Mc], 1)); rhs.append(p_part(ri, np.full(NB, L_NECK), k))

    # 3) raccord, continuite de dP/dz
    rows.append(np.concatenate([Dn, -Dc], 1)); rhs.append(dp_part_dz(ri, np.full(NB, L_NECK), k))

    # 4) epaulement : dP/dz = 0           (z = L, R_col < r <= R_cav)
    rs = R_NECK + (np.arange(NB) + 0.5)*(R_CAV - R_NECK)/NB
    _, Ds = cav_basis(rs, np.full(NB, L_NECK), k)
    rows.append(np.concatenate([np.zeros((NB, nN)), Ds], 1))
    rhs.append(-dp_part_dz(rs, np.full(NB, L_NECK), k))

    A = np.concatenate(rows, 0).astype(complex)
    b = np.concatenate(rhs).astype(complex)
    cn = np.linalg.norm(A, axis=0); cn[cn == 0] = 1.0
    sol, *_ = np.linalg.lstsq(A/cn, b, rcond=1e-13)
    sol = sol/cn
    if verbose:
        s = np.linalg.svd(A/cn, compute_uv=False)
        print(f"  systeme de raccord : {A.shape[0]} equations, {A.shape[1]} inconnues, "
              f"kappa = {s[0]/s[-1]:.2e}")
        print(f"  residu du raccord  : {np.linalg.norm(A/cn@ (sol*cn) - b)/np.linalg.norm(b)*100:.4f} %")
    return sol, nN, nC, k


def field(sol, nN, nC, k, r, z):
    """Champ reconstruit, sur une grille (masque col/cavite)."""
    out = np.zeros(r.shape, complex)
    inN = (z < L_NECK) & (r <= R_NECK)
    inC = (z >= L_NECK) & (r <= R_CAV)
    if inN.any():
        Mn, _ = neck_basis(r[inN], z[inN], k)
        out[inN] = Mn @ sol[:nN]
    if inC.any():
        Mc, _ = cav_basis(r[inC], z[inC], k)
        out[inC] = Mc @ sol[nN:] + p_part(r[inC], z[inC], k)
    return out


# --------------------------------------------------------------------------
# reference FDM
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
                    if 0 <= jj < Nz and fl[i, jj]: rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                    else:                          diag += ih2
            rows += [ci]; cols += [ci]; dat += [diag]
            b[ci] = -F[i, j]
    A = sp.coo_matrix((dat, (rows, cols)), shape=(N, N), dtype=complex).tocsr()
    return r, z, fl, spsolve(A, b).reshape((Nz, Nr)).T


if __name__ == "__main__":
    print("=" * 76)
    print("ETAPE 1  --  la base satisfait-elle l'EDP par construction ?")
    print("=" * 76)
    k0 = 2*np.pi*FREQ0/C
    rr = verif_residu(k0)
    print(f"  {M_N} modes de col, {N_C} modes de cavite")
    print(f"  residu |lap(phi) + k^2 phi| / |k^2 phi|  =  {rr:.3e}")
    print(f"  -> {'NUL a la precision machine' if rr < 1e-6 else 'NON NUL, il y a un bug'}")

    rp = np.array([0.0, 0.02, R_CAV]); zp = np.array([0.06, 0.08, 0.10])
    h = 1e-5
    dr = (p_part(np.full(3, R_CAV), zp, k0) - p_part(np.full(3, R_CAV-h), zp, k0))/h
    print(f"  paroi laterale, dP_part/dr en R_cav : {np.abs(dr).max():.3e} "
          f"(doit etre ~0 : les modes ne peuvent pas la corriger)")

    print()
    print("=" * 76)
    print("ETAPE 2  --  le raccord, et la comparaison au FDM")
    print("=" * 76)
    t0 = time.time()
    sol, nN, nC, k = solve_match(FREQ0, verbose=True)
    t_tr = time.time() - t0
    t1 = time.time(); r_f, z_f, FL, P_f = fdm(FREQ0); t_fdm = time.time() - t1

    RR, ZZ = np.meshgrid(r_f, z_f, indexing="ij")
    P_t = field(sol, nN, nC, k, RR, ZZ)
    e = np.linalg.norm((P_t - P_f)[FL])/np.linalg.norm(P_f[FL])
    print(f"\n  Trefftz : {nN + nC} inconnues, {t_tr*1000:.0f} ms")
    print(f"  FDM     : {FL.sum()} inconnues, {t_fdm*1000:.0f} ms")
    print(f"  ERREUR L2 CONTRE LE FDM : {e*100:.3f} %")
    print(f"  max|P| Trefftz = {np.abs(P_t[FL]).max():.1f} Pa   "
          f"FDM = {np.abs(P_f[FL]).max():.1f} Pa   "
          f"rapport {np.abs(P_t[FL]).max()/np.abs(P_f[FL]).max():.4f}")

    if os.environ.get("TM_SWEEP", "1") == "1":
        print()
        print("=" * 76)
        print("BALAYAGE  --  Trefftz contre FDM sur 150-300 Hz")
        print("=" * 76)
        fs = np.linspace(150, 300, 21)
        errs, at, af = [], [], []
        for f in fs:
            s2, a2, b2, k2_ = solve_match(f)
            Pt = field(s2, a2, b2, k2_, RR, ZZ)
            _, _, _, Pf = fdm(f)
            errs.append(np.linalg.norm((Pt - Pf)[FL])/np.linalg.norm(Pf[FL]))
            at.append(np.abs(Pt[FL]).max()); af.append(np.abs(Pf[FL]).max())
        errs = np.array(errs); at = np.array(at); af = np.array(af)
        for f, e_, a_, b_ in zip(fs, errs, at, af):
            print(f"  f = {f:6.1f} Hz | L2 = {e_*100:8.4f} % | "
                  f"max|P| {a_:10.1f} / {b_:10.1f}  ({a_/b_:.4f}x)")
        print(f"\n  L2 median sur la bande : {np.median(errs)*100:.4f} %"
              f"   pire : {errs.max()*100:.4f} %")
        np.savez("data/trefftz_match.npz", freqs=fs, err=errs, amp=at, amp_ref=af,
                 M_N=M_N, N_C=N_C)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].semilogy(fs, errs*100, "C0o-", lw=1.4, ms=4)
        ax[0].set(xlabel="fréquence (Hz)", ylabel="L2 (%)",
                  title=f"Trefftz + raccord : {nN+nC} inconnues")
        ax[0].grid(alpha=.3, which="both")
        ax[1].semilogy(fs, af, "ko-", lw=1.4, ms=4, label="FDM")
        ax[1].semilogy(fs, at, "C1s--", lw=1.2, ms=3.5, label="Trefftz")
        ax[1].set(xlabel="fréquence (Hz)", ylabel="max |P| (Pa)", title="réponse en fréquence")
        ax[1].legend(); ax[1].grid(alpha=.3, which="both")
        fig.tight_layout(); fig.savefig("plots/trefftz_match.png", dpi=115)
        print("  Figure : plots/trefftz_match.png")
