
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# =============== Page & Intro ===============
st.set_page_config(page_title="Mean‑Field Impulse Control – Real & Imag Utility", layout="wide")
st.title("Mean‑Field Impulse Control with Real & Imaginary Utility")

with st.expander("About this model", expanded=False):
    st.markdown("""
This Streamlit app implements a **long‑run average impulse‑control** model (à la HSZ)
with a **complex-utility** twist:
- Agents control a diffusion **X** by applying impulses: at threshold **y** the state is reset to **w**.
- The long‑run reward per unit time for a (w,y) policy is
  \\(F_p(w,y) = \\dfrac{g(y)-g(w) + p\\,(y-w) - K}{\\xi(y)-\\xi(w)}\\) where
  \\(\\xi\\) and \\(g\\) are built from the diffusion's **scale**/**speed** measures and the running reward **c(x)**.
- **Price is endogenous** via a mean‑field mapping \\(p=\\phi(z)\\) with \\(z=\\dfrac{y-w}{\\xi(y)-\\xi(w)}\\).
- We add **imaginary utility** by an effective price kicker: \\(p_{eff}=\\phi(z)+\\eta\\,\\varphi\\,\\lambda_2\\).

This app numerically constructs \\(\\xi,g\\) on a grid from user‑selected \\(\\mu(x),\\sigma(x),c(x)\\),
searches for a best‑response (w,y), and computes a fixed point in \\(z\\).
""")

# =============== Sidebar: Model choices ===============
st.sidebar.header("Model setup")

# Domain & grid
a = st.sidebar.number_input("Left boundary a", value=0.0, step=0.1)
b = st.sidebar.number_input("Right boundary b", value=2.0, step=0.1)
n_grid = st.sidebar.slider("Grid points", 200, 2000, 600, 50)

# Dynamics choice
dyn = st.sidebar.selectbox("Drift μ(x) & volatility σ(x)", 
                           ["Logistic μ, const σ", "OU μ, const σ", "Const μ, const σ"])

if dyn == "Logistic μ, const σ":
    alpha = st.sidebar.number_input("α (growth rate)", value=1.0, step=0.1)
    Kc    = st.sidebar.number_input("K_c (capacity)", value=1.0, step=0.1)
    sig0  = st.sidebar.number_input("σ (volatility)", value=0.2, step=0.05)
    def mu(x): return alpha*x*(1.0 - x/max(1e-9,Kc))
    def sig(x): return sig0 + 0.0*x
elif dyn == "OU μ, const σ":
    kappa = st.sidebar.number_input("κ (mean‑reversion)", value=1.0, step=0.1)
    xbar  = st.sidebar.number_input("x̄ (long‑run mean)", value=1.0, step=0.1)
    sig0  = st.sidebar.number_input("σ (volatility)", value=0.3, step=0.05)
    def mu(x): return kappa*(xbar - x)
    def sig(x): return sig0 + 0.0*x
else:
    mu0 = st.sidebar.number_input("μ (constant drift)", value=0.2, step=0.05)
    sig0= st.sidebar.number_input("σ (volatility)", value=0.3, step=0.05)
    def mu(x): return mu0 + 0.0*x
    def sig(x): return sig0 + 0.0*x

# Running reward
c_form = st.sidebar.selectbox("Running reward c(x)", ["Linear", "Quadratic concave", "Constant"])
if c_form == "Linear":
    c0 = st.sidebar.number_input("c0", value=0.0, step=0.1)
    c1 = st.sidebar.number_input("c1", value=1.0, step=0.1)
    def cfun(x): return c0 + c1*x
elif c_form == "Quadratic concave":
    cmax = st.sidebar.number_input("peak value", value=1.0, step=0.1)
    xm   = st.sidebar.number_input("peak location", value=1.0, step=0.1)
    width= st.sidebar.number_input("width (std‑like)", value=0.6, step=0.1)
    def cfun(x): return np.maximum(0.0, cmax - ((x-xm)/max(1e-6,width))**2 )
else:
    cconst = st.sidebar.number_input("c (const)", value=0.2, step=0.05)
    def cfun(x): return cconst + 0.0*x

# Price function φ(z) and fixed cost
st.sidebar.subheader("Market & cost")
phi_a = st.sidebar.number_input("φ intercept a (max price)", value=1.5, step=0.1)
phi_b = st.sidebar.number_input("φ slope b (downward vs z)", value=0.8, step=0.1, help="p = max(0, a - b z)")
Kfix  = st.sidebar.number_input("Impulse fixed cost K", value=0.20, step=0.05)

# Imaginary utility kicker
st.sidebar.subheader("Imaginary utility")
eta   = st.sidebar.slider("η (conversion efficiency)", 0.0, 1.0, 0.6, 0.05)
varphi= st.sidebar.slider("φ̂ (reachability)", 0.0, 1.0, 0.6, 0.05)
lam2  = st.sidebar.number_input("λ₂ (per‑unit kick)", value=0.15, step=0.05)
def phi(z): 
    return max(0.0, phi_a - phi_b*z)
def p_eff(z): 
    return phi(z) + eta*varphi*lam2

# Fixed‑point and search settings
st.sidebar.subheader("Solver settings")
max_fp_iter = st.sidebar.slider("Fixed‑point iterations", 1, 50, 15, 1)
brute_n     = st.sidebar.slider("Grid for (w,y) search (per axis)", 10, 120, 40, 5)
x0_plot     = st.sidebar.number_input("Reference x₀ for ξ,g construction", value=(a+b)/2.0)

# =============== Numerical construction of s, m, ξ, g ===============
# Grid & measures
x = np.linspace(a, b, int(n_grid))
dx = (b-a)/max(1, n_grid-1)

# Avoid division by zero
def _slope(xval):
    s2 = sig(xval)**2
    return 2.0*mu(xval)/max(1e-12, s2)

# Compute scale density s(x) = exp(-∫ 2μ/σ² dx), normalized s(x0)=1
slope_vals = np.array([_slope(xi) for xi in x])
# cumulative integral of slope from x0_plot
i0 = np.searchsorted(x, x0_plot, side='left')
# left of i0
cumL = np.cumsum(slope_vals[:i0][::-1])*dx
cumL = cumL[::-1]
# right of i0
cumR = np.cumsum(slope_vals[i0:])*dx
phi_int = np.concatenate([cumL, [0.0], cumR[1:]]) if i0>0 else np.concatenate([[0.0], cumR[1:]])
s_density = np.exp(-phi_int)

# speed density m(x) = 2/(σ² s)
m_density = 2.0/(np.maximum(1e-12, (sig(x)**2) * s_density))

# Helpers for cumulative S and M
S = np.cumsum(s_density)*dx                      # scale function up to additive const
M = np.cumsum(m_density)*dx                      # speed measure cumulative

# Functions for arbitrary x via linear interp
def interp(arr, xv):
    return np.interp(xv, x, arr)

def xi_fun(xv):
    # ξ(x) = ∫_{x0}^{x} M[a,v] dS(v) = ∫ (M(v)-M(a)) dS(v). Using cumulative sums on grid.
    # Discrete approximation: ξ(x_j) ~ sum_{k=i0..j-1} M[k]*ΔS[k], with sign for direction.
    xv = float(np.clip(xv, a, b))
    Sj = interp(S, xv)
    i = int(np.searchsorted(x, xv, side='left'))
    # ΔS * M at midpoints
    if i >= 1:
        dS = np.diff(S[:i+1])
        Mm = 0.5*(M[:i][...] + M[1:i+1][...])
        val = np.sum(Mm * dS)
    else:
        dS = np.diff(S[i:i0+1])
        Mm = 0.5*(M[i:i0][...] + M[i+1:i0+1][...])
        val = -np.sum(Mm * dS)
    # shift to make ξ(x0)=0
    return val

def g_fun(xv):
    # g(x) = ∫_{x0}^{x} ∫_{a}^{v} c(u) dM(u) dS(v)
    xv = float(np.clip(xv, a, b))
    i = int(np.searchsorted(x, xv, side='left'))
    cvals = cfun(x)
    dM = np.diff(M)
    Cm = np.concatenate([[0.0], np.cumsum(0.5*(cvals[:-1]+cvals[1:]) * dM)])
    if i >= 1:
        dS = np.diff(S[:i+1])
        Cmid = 0.5*(Cm[:i] + Cm[1:i+1])
        val = np.sum(Cmid * dS)
    else:
        dS = np.diff(S[i:i0+1])
        Cmid = 0.5*(Cm[i:i0] + Cm[i+1:i0+1])
        val = -np.sum(Cmid * dS)
    return float(val)

# Vectorize for plotting
xi_vec = np.vectorize(xi_fun)
g_vec  = np.vectorize(g_fun)

# =============== Visualize μ, σ, c, s, m, ξ, g ===============
col1, col2 = st.columns(2)
with col1:
    st.subheader("Dynamics and running reward")
    fig = plt.figure()
    # After (vectorized & dtype-safe):
    mu_vals  = np.array([mu(float(xx))  for xx in x], dtype=float)
    sig_vals = np.array([sig(float(xx)) for xx in x], dtype=float)

    # cfun is written to handle arrays in all branches, so call it vectorized:
    c_vals = cfun(x).astype(float)

    plt.plot(x, mu_vals,  label="μ(x)")
    plt.plot(x, sig_vals, label="σ(x)")
    plt.plot(x, c_vals,   label="c(x)")
    plt.legend()
    st.pyplot(fig)
with col2:
    st.subheader("Constructed measures & helpers")
    fig2 = plt.figure()
    plt.plot(x, s_density, label="s(x) (scale density)")
    plt.plot(x, m_density, label="m(x) (speed density)")
    plt.legend()
    st.pyplot(fig2)

col3, col4 = st.columns(2)
with col3:
    st.subheader("ξ(x)")
    fig3 = plt.figure()
    plt.plot(x, xi_vec(x))
    st.pyplot(fig3)
with col4:
    st.subheader("g(x)")
    fig4 = plt.figure()
    plt.plot(x, g_vec(x))
    st.pyplot(fig4)

# =============== Best response search in (w,y) and fixed point ===============
def F_value(w, y, price_eff):
    denom = xi_fun(y) - xi_fun(w)
    if denom <= 1e-12 or y <= w: 
        return -1e9, 0.0  # invalid
    z = (y - w)/denom
    val = (g_fun(y) - g_fun(w) + price_eff*(y - w) - Kfix) / denom
    return val, z

def best_response(z_guess):
    pe = p_eff(z_guess)
    W = np.linspace(a, b, brute_n)
    best = (-1e18, a, a, 0.0)
    for i in range(len(W)-1):
        for j in range(i+1, len(W)):
            w, y = W[i], W[j]
            val, zloc = F_value(w, y, pe)
            if val > best[0]:
                best = (val, w, y, zloc)
    return best  # (F*, w*, y*, z(w*,y*))

st.subheader("Fixed‑point iteration for mean‑field equilibrium")
z_hist = []
z = st.number_input("Initial guess for z", value=0.4, step=0.1)
for k in range(max_fp_iter):
    Fstar, wstar, ystar, z_new = best_response(z)
    z_hist.append(z_new)
    z = z_new

Fstar, wstar, ystar, z_star = best_response(z)
p_star = phi(z_star)
pe_star = p_eff(z_star)

colA, colB, colC, colD = st.columns(4)
colA.metric("w*", f"{wstar:.3f}")
colB.metric("y*", f"{ystar:.3f}")
colC.metric("z*", f"{z_star:.3f}")
colD.metric("p* / p_eff*", f"{p_star:.3f} / {pe_star:.3f}")

figZ = plt.figure()
plt.plot(np.arange(1, len(z_hist)+1), z_hist)
plt.xlabel("Iteration")
plt.ylabel("z")
st.pyplot(figZ)

st.markdown("### Value landscape (coarse heatmap)")
W = np.linspace(a, b, 100)
H = np.full((len(W), len(W)), np.nan)
for i in range(len(W)):
    for j in range(i+1, len(W)):
        w, y = W[i], W[j]
        val, _ = F_value(w, y, pe_star)
        H[i, j] = val
figH = plt.figure()
plt.imshow(H.T, origin="lower", extent=(a,b,a,b), aspect="auto")
plt.xlabel("w")
plt.ylabel("y")
st.pyplot(figH)

st.caption("Note: numerical quadrature on coarse grids may introduce small bias; increase grid density for smoother ξ,g.")
