"""
Streamlit MVP – Per‑Cohort Family Selector (7 Aggregation Models)
================================================================
Each fighter cohort can now pick **one of seven effectiveness families**
(Power‑Mean, CES, Cobb–Douglas, Exponential‑Sat, Logit, Quadratic, or
Hybrid).  Coefficients auto‑load from illustrative defaults but are easy
for you to hard‑code or future‑expose.

Run:
    pip install streamlit numpy matplotlib
    streamlit run oodamvp.py
"""

from math import exp, log
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

EPS = 1e-9

st.set_page_config(page_title="OODA Family Selector", layout="wide")
st.title("OODA Attrition — Per‑Cohort Family Selection")

# ───────────────── Aggregation Family Functions ─────────────────

def power_mean(O, R, D, A, alpha, beta, gamma, delta, p):
    inner = alpha*O**p + beta*R**p + gamma*D**p + delta*A**p
    return inner ** (1.0/p) if p != 0 else np.exp((alpha*log(O)+beta*log(R)+gamma*log(D)+delta*log(A))/(alpha+beta+gamma+delta))

def CES(O, R, D, A, alpha, beta, gamma, delta, rho):
    inner = alpha*O**rho + beta*R**rho + gamma*D**rho + delta*A**rho
    return inner ** (1.0/rho)

def cobb_douglas(O, R, D, A, k, alpha, beta, gamma, delta):
    return k * (O**alpha) * (R**beta) * (D**gamma) * (A**delta)

def exponential_sat(O, R, D, A, alpha, beta, gamma, delta):
    return 1.0 - exp(-(alpha*O + beta*R + gamma*D + delta*A))

def logistic(O, R, D, A, alpha, beta, gamma, delta, theta):
    z = alpha*log(O+EPS) + beta*log(R+EPS) + gamma*log(D+EPS) + delta*log(A+EPS) - theta
    return 1.0 / (1.0 + exp(-z))

def quadratic(O, R, D, A, w):
    return (
        w["w1"]*O**2 + w["w2"]*R**2 + w["w3"]*D**2 + w["w4"]*A**2 +
        w["w12"]*O*R + w["w13"]*O*D + w["w14"]*O*A +
        w["w23"]*R*D + w["w24"]*R*A + w["w34"]*D*A
    )

def hybrid_add_root(O, R, D, A, alpha, beta, gamma, delta, lam):
    return alpha*O + beta*R + gamma*D + delta*A + lam*(O*R*D*A)**0.25

FAMILY_FUNCS = {
    "power":   power_mean,
    "ces":     CES,
    "cobb":    cobb_douglas,
    "exp":     exponential_sat,
    "logit":   logistic,
    "quad":    quadratic,
    "hybrid":  hybrid_add_root,
}

FAMILY_LIST = list(FAMILY_FUNCS.keys())  # order for selectboxes

# ─────────────── Default parameter sketches (illustrative) ───────────────
GENERIC_DEFAULTS = {
    "power":  dict(alpha=0.25,beta=0.25,gamma=0.25,delta=0.25,p=1.0),
    "ces":    dict(alpha=0.25,beta=0.25,gamma=0.25,delta=0.25,rho=1.0),
    "cobb":   dict(k=1.0,alpha=0.25,beta=0.25,gamma=0.25,delta=0.25),
    "exp":    dict(alpha=1.0,beta=1.0,gamma=1.0,delta=1.0),
    "logit":  dict(alpha=1.0,beta=1.0,gamma=1.0,delta=1.0,theta=0.0),
    "quad":   dict(w1=1,w2=1,w3=1,w4=1,w12=1,w13=1,w14=1,w23=1,w24=1,w34=1),
    "hybrid": dict(alpha=0.25,beta=0.25,gamma=0.25,delta=0.25,lam=0.5),
}

COHORTS = ["Blue 5-Gen","Blue 4-Gen","Red 5-Gen","Red 4-Gen"]

DEFAULT_FAMILY = {
    "Blue 5-Gen":"cobb",
    "Blue 4-Gen":"hybrid",
    "Red 5-Gen":"ces",
    "Red 4-Gen":"exp",
}

# ───────────────────── UI helpers ─────────────────────

def shares_block(label, defaults):
    """Return O,R,D,A shares that sum to 1 (A auto-derived)."""
    st.subheader(label)
    o = st.slider(f"{label} O-share", 0.0, 1.0, defaults[0], 0.05)
    r = st.slider(f"{label} R-share", 0.0, 1.0, defaults[1], 0.05)
    d = st.slider(f"{label} D-share", 0.0, 1.0, defaults[2], 0.05)
    total = o+r+d
    if total >= 1.0:
        o, r, d = [x/(total+EPS)*0.9 for x in (o,r,d)]
        total = o+r+d
    a = 1.0 - total
    st.text(f"A-share auto‑set → {a:.2f}")
    return o,r,d,a

# compute E via selected family & param dict

def compute_E(shares, family, params):
    func = FAMILY_FUNCS[family]
    if family == "quad":
        # ensure all weights exist
        base = GENERIC_DEFAULTS["quad"].copy(); base.update(params)
        return func(*shares, base)
    return func(*shares, **params)

# ─────────────────── Sidebar Controls ───────────────────
with st.sidebar:
    st.header("Force counts")
    b5_0 = st.slider("Blue 5‑Gen",0,200,40,10)
    b4_0 = st.slider("Blue 4‑Gen",0,200,60,10)
    r5_0 = st.slider("Red 5‑Gen",0,200,30,10)
    r4_0 = st.slider("Red 4‑Gen",0,200,70,10)

    st.header("Base lethality k (kills/ftr‑sec)")
    k5_blue = st.slider("k Blue 5‑Gen",0.0,0.2,0.06,0.01)
    k4_blue = st.slider("k Blue 4‑Gen",0.0,0.2,0.03,0.01)
    k5_red  = st.slider("k Red 5‑Gen",0.0,0.2,0.05,0.01)
    k4_red  = st.slider("k Red 4‑Gen",0.0,0.2,0.02,0.01)

    st.header("Select family per cohort")
    fam_b5 = st.selectbox("Blue 5‑Gen family",FAMILY_LIST, index=FAMILY_LIST.index(DEFAULT_FAMILY["Blue 5-Gen"]))
    fam_b4 = st.selectbox("Blue 4‑Gen family",FAMILY_LIST, index=FAMILY_LIST.index(DEFAULT_FAMILY["Blue 4-Gen"]))
    fam_r5 = st.selectbox("Red 5‑Gen family",FAMILY_LIST,  index=FAMILY_LIST.index(DEFAULT_FAMILY["Red 5-Gen"]))
    fam_r4 = st.selectbox("Red 4‑Gen family",FAMILY_LIST,  index=FAMILY_LIST.index(DEFAULT_FAMILY["Red 4-Gen"]))

    st.header("O‑R‑D shares (A derives)")
    sh_b5 = shares_block("Blue 5‑Gen", (0.25,0.25,0.25))
    sh_b4 = shares_block("Blue 4‑Gen", (0.25,0.25,0.25))
    sh_r5 = shares_block("Red 5‑Gen",  (0.25,0.25,0.25))
    sh_r4 = shares_block("Red 4‑Gen",  (0.25,0.25,0.25))

    st.header("Timeline")
    horizon = st.slider("Engagement duration (s)",30,600,180,10)
    dt = 0.2

# ───────────── Calculate E for each cohort ─────────────
E_b5 = compute_E(sh_b5, fam_b5, GENERIC_DEFAULTS[fam_b5])
E_b4 = compute_E(sh_b4, fam_b4, GENERIC_DEFAULTS[fam_b4])
E_r5 = compute_E(sh_r5, fam_r5, GENERIC_DEFAULTS[fam_r5])
E_r4 = compute_E(sh_r4, fam_r4, GENERIC_DEFAULTS[fam_r4])

# ───────────── Simulation arrays ─────────────
steps = int(horizon/dt)+1
_time = np.linspace(0,horizon,steps)

b5 = np.empty(steps); b4 = np.empty(steps)
r5 = np.empty(steps); r4 = np.empty(steps)

b5[0],b4[0] = b5_0,b4_0
r5[0],r4[0] = r5_0,r4_0

for i in range(1,steps):
    tot_blue = b5[i-1]+b4[i-1]
    tot_red  = r5[i-1]+r4[i-1]

    if tot_blue==0 or tot_red==0:
        b5[i:],b4[i:],r5[i:],r4[i:] = b5[i-1],b4[i-1],r5[i-1],r4[i-1]
        break

    kills_blue = dt*(k5_blue*E_b5*b5[i-1] + k4_blue*E_b4*b4[i-1])
    kills_red  = dt*(k5_red*E_r5*r5[i-1]  + k4_red*E_r4*r4[i-1])

    # distribute proportionally
    frac_r5 = r5[i-1]/tot_red
    frac_r4 = r4[i-1]/tot_red
    frac_b5 = b5[i-1]/tot_blue
    frac_b4 = b4[i-1]/tot_blue

    r5[i] = max(r5[i-1] - kills_blue*frac_r5,0.0)
    r4[i] = max(r4[i-1] - kills_blue*frac_r4,0.0)
    b5[i] = max(b5[i-1] - kills_red*frac_b5,0.0)
    b4[i] = max(b4[i-1] - kills_red*frac_b4,0.0)

# ───────────── Plot ─────────────
fig,ax = plt.subplots()
ax.plot(_time,b5,label="Blue 5‑Gen",linewidth=2)
ax.plot(_time,b4,label="Blue 4‑Gen",linestyle="--",linewidth=2)
ax.plot(_time,r5,label="Red 5‑Gen",linestyle=":",linewidth=2)
ax.plot(_time,r4,label="Red 4‑Gen",linestyle="-.",linewidth=2)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Aircraft remaining")
ax.set_title("Attrition — Per‑Cohort Aggregation Family")
ax.legend()

st.pyplot(fig)

# ───────────── Metrics ─────────────
col1,col2 = st.columns(2)
col1.metric("Blue survivors",f"5‑Gen: {b5[-1]:.1f} | 4‑Gen: {b4[-1]:.1f}")
col2.metric("Red survivors", f"5‑Gen: {r5[-1]:.1f} | 4‑Gen: {r4[-1]:.1f}")

st.caption("Choose aggregation family for each cohort and see how doctrine shifts outcomes.  ✈️🧮 #vibecoding")
