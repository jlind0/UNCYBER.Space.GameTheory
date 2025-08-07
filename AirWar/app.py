from math import exp, log
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

"""
OODA Attrition – Six‑Cohort Duel 
=================================================================
**What the “OODA Attrition – Six-Cohort Duel” app lets you do**

1. **Set up two opposing air forces in seconds**

   * The left-hand sidebar lists six aircraft “cohorts” (e.g., *NATO 5-Gen*, *China 4-Gen*).
   * For each cohort you drag sliders to pick how many jets you have and how lethal they are (the *k* slider).
   * You also decide how that cohort invests its effort across the four O O D A phases—**Observe, Orient, Decide, Act**—using three simple sliders; the fourth value (*Act*) is filled in automatically so everything still adds up to 100 % .

2. **Choose how “quality” is calculated**

   * A drop-down menu lets you pick one of several built-in mathematical recipes (power-mean, Cobb-Douglas, CES, etc.).
   * Behind the scenes the app combines your O, R, D, A choices with the selected recipe to produce a raw effectiveness score **E** for each cohort .

3. **Convert raw effectiveness into real-world punch**

   * Because combat advantage is rarely linear, every raw **E** score is remapped through a logistic curve—controlled by two sliders (slope λ and midpoint) in the sidebar—to keep values between 0 and 1 and emphasize meaningful differences rather than tiny decimals .

4. **Watch the duel play out in real time**

   * Click *Run* (Streamlit updates automatically) and the model steps through the engagement second-by-second.
   * At each step it calculates each side’s fire-power and turns that into a “kill budget” that is allocated proportionally across the enemy’s surviving aircraft .
   * A clean line graph shows how the six cohorts attrit over the chosen time horizon, and two big counters keep score of survivors for “Side 1” (NATO + Ukraine) and “Side 2” (China + Russia) .

5. **Experiment and learn**

   * Drag any slider—the plot and survivor counts refresh instantly, letting you explore *what-if* questions in seconds.
   * Try doubling a cohort’s numbers, shifting effort from *Observe* to *Decide*, or steepening the kill-advantage scale σ to see how small qualitative edges can snowball into big numerical wins.

In short, this Streamlit app is an interactive sandbox that turns abstract OODA-loop theory into an intuitive, visual **“what happens if?”** tool for planners, analysts, or curious enthusiasts.

* [OODA Paper](https://multiplex.studio/files/OODA-Game.pdf)
* [Source Code](https://github.com/jlind0/UNCYBER.Space.GameTheory/blob/main/AirWar/app.py)
"""

# ────────────────────────────────────────────────
EPS = 1e-9
st.set_page_config(page_title="OODA – Six‑Cohort Duel", layout="wide")
st.title("OODA Attrition – NATO & Ukraine vs China & Russia (4 + 5 Gen mix)")

# ───────────────── Aggregation families ──────────────────

def power_mean(O, R, D, A, alpha, beta, gamma, delta, p):
    s = alpha * O ** p + beta * R ** p + gamma * D ** p + delta * A ** p
    return s ** (1 / p) if p != 0 else np.exp(
        (alpha * log(O) + beta * log(R) + gamma * log(D) + delta * log(A))
        / (alpha + beta + gamma + delta)
    )


def CES(O, R, D, A, alpha, beta, gamma, delta, rho):
    s = alpha * O ** rho + beta * R ** rho + gamma * D ** rho + delta * A ** rho
    return s ** (1 / rho)


def cobb(O, R, D, A, k, alpha, beta, gamma, delta):
    return k * O ** alpha * R ** beta * D ** gamma * A ** delta


def exp_sat(O, R, D, A, c, lam):
    return c * (1 - np.exp(-lam * (O + R + D + A)))


def logit(O, R, D, A, alpha, beta, gamma, delta, theta):
    z = (
        alpha * log(O + EPS)
        + beta * log(R + EPS)
        + gamma * log(D + EPS)
        + delta * log(A + EPS)
        - theta
    )
    return 1 / (1 + exp(-z))


def quad(O, R, D, A, w):
    return (
        w["w1"] * O ** 2
        + w["w2"] * R ** 2
        + w["w3"] * D ** 2
        + w["w4"] * A ** 2
        + w["w5"] * O * R
        + w["w6"] * O * D
        + w["w7"] * O * A
        + w["w8"] * R * D
        + w["w9"] * R * A
        + w["w10"] * D * A
    )


def hybrid(O, R, D, A, k, alpha, beta, gamma, delta, lam):
    return k * (O ** alpha) * (np.exp(-lam * R)) + beta * D + gamma * np.sqrt(A)


FAMILY_FUNCS = {
    "power": power_mean,
    "ces": CES,
    "cobb": cobb,
    "exp": exp_sat,
    "logit": logit,
    "quad": quad,
    "hybrid": hybrid,
}
FAMILIES = list(FAMILY_FUNCS.keys())

GENERIC = {
    "power": dict(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25, p=0.8),
    "ces": dict(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25, rho=1.5),
    "cobb": dict(k=0.7, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2),
    "exp": dict(c=1.0, lam=0.05),
    "logit": dict(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2, theta=0.1),
    "quad": dict(
        w1=0.2,
        w2=0.2,
        w3=0.2,
        w4=0.2,
        w5=0.05,
        w6=0.05,
        w7=0.05,
        w8=0.02,
        w9=0.02,
        w10=0.01,
    ),
    "hybrid": dict(k=0.6, alpha=0.3, beta=0.2, gamma=0.2, delta=0.3, lam=0.05),
}

COHORTS = [
    ("NATO 5‑Gen", "Side 1", "cobb"),
    ("NATO 4‑Gen", "Side 1", "cobb"),
    ("Ukraine 4‑Gen", "Side 1", "power"),
    ("China 5‑Gen", "Side 2", "ces"),
    ("China 4‑Gen", "Side 2", "exp"),
    ("Russia 5‑Gen", "Side 2", "logit"),
    ("Russia 4‑Gen", "Side 2", "quad"),
]

# ───────── Cohort‑specific presets ─────────
DEFAULTS = {
    "NATO 5‑Gen": dict(count=25, k=0.07, shares=(0.30, 0.30, 0.25)),
    "NATO 4‑Gen": dict(count=20, k=0.045, shares=(0.30, 0.30, 0.25)),
    "Ukraine 4‑Gen": dict(count=30, k=0.035, shares=(0.30, 0.25, 0.25)),
    "China 5‑Gen": dict(count=20, k=0.08, shares=(0.30, 0.30, 0.25)),
    "China 4‑Gen": dict(count=20, k=0.025, shares=(0.25, 0.30, 0.25)),
    "Russia 5‑Gen": dict(count=10, k=0.065, shares=(0.25, 0.30, 0.25)),
    "Russia 4‑Gen": dict(count=30, k=0.035, shares=(0.25, 0.30, 0.25)),
}

# ───────── UI helper – shares ─────────
def share_sliders(tag: str, tbl=(0.25, 0.25, 0.25), pfx: str = ""):
    """Expose O-, R-, D-sliders; A is auto-computed as 1-(O+R+D)."""
    # ignore any preset A, accept 3- or 4-tuples
    if len(tbl) < 3:
        tbl = (0.25, 0.25, 0.25)
    O_def, R_def, D_def = tbl[0], tbl[1], tbl[2]

    colO, colR, colD = st.columns(3)
    with colO:
        O = st.slider(f"{pfx}O ({tag})", 0.0, 1.0, O_def, 0.01)
    with colR:
        R = st.slider(f"{pfx}R ({tag})", 0.0, 1.0, R_def, 0.01)
    with colD:
        D = st.slider(f"{pfx}D ({tag})", 0.0, 1.0, D_def, 0.01)

    A = 1.0 - (O + R + D)
    if A < 0:
        st.warning("O + R + D exceed 1; A set to 0.")
        A = 0.0
    st.caption(f"A (auto) = {A:.2f}")
    return O, R, D, A

# ───────── Effectiveness helpers ─────────

def raw_E(shares, fam):
    params = GENERIC[fam]
    fn = FAMILY_FUNCS[fam]
    if fam == "quad":
        p = GENERIC["quad"].copy()
        p.update(params)
        return fn(*shares, p)
    return fn(*shares, **params)


def logistic_remap(val: float, lam: float = 5.0, mid: float = 0.5) -> float:
    """Bound *val* to (0,1) using a logistic centred on *mid* with slope *lam*."""
    return 1.0 / (1.0 + exp(-lam * (val - mid)))

# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("Force counts & lethality k")
    counts = {}
    kvals = {}
    famsel = {}
    shares = {}
    vuln = {}

    for tag, side, dfam in COHORTS:
        presets = DEFAULTS.get(tag, {})
        counts[tag] = st.slider(
            f"{tag} count",
            0,
            200,
            presets.get("count", 60),
            10,
        )
        kvals[tag] = st.slider(
            f"k {tag}",
            0.0,
            0.2,
            presets.get("k", 0.05),
            0.01,
        )
        famsel[tag] = st.selectbox(
            f"{tag} family", FAMILIES, FAMILIES.index(dfam), key=f"{tag}-family"
        )
        shares[tag] = share_sliders(tag, presets.get("shares", (0.25, 0.25, 0.25)))
        vuln[tag] = st.slider(f"Vulnerability {tag}",
                           0.5, 1.5, 1.0, 0.05)

    st.header("Model parameters")
    horizon = st.slider("Engagement (s)", 30, 600, 30, 10)
    dt = 0.2

    st.subheader("Logistic remap (E → Ê)")
    lam_E = st.slider("λ (slope)", 1.0, 10.0, 1.5, 0.5)
    mid_E = st.slider("mid‑point", 0.1, 1.0, 1.0, 0.05)

    st.subheader("Kill advantage scale σ")
    sigma = st.slider("σ (steepness)", 1.0, 10.0, 3.5, 0.5)

# ───────── Calculate effectiveness ─────────
# Anchor‑ratio scaling: compare every raw E to its family's baseline value
ANCHOR_SHARES = (0.25, 0.25, 0.25, 0.25)
E0 = {fam: raw_E(ANCHOR_SHARES, fam) for fam in FAMILIES}

E_raw = {t: raw_E(shares[t], famsel[t]) for t, _, _ in COHORTS}
E_scaled = {t: E_raw[t] / (E0[famsel[t]] + EPS) for t in E_raw}
E = {t: logistic_remap(E_scaled[t], lam_E, mid_E) for t in E_scaled}


# ───────── Simulation arrays ─────────
steps = int(horizon / dt) + 1
_time = np.linspace(0, horizon, steps)
state = {t: np.empty(steps) for t, _, _ in COHORTS}
for t in state:
    state[t][0] = counts[t]

side_tags = {
    "Side 1": [t for t, s, _ in COHORTS if s == "Side 1"],
    "Side 2": [t for t, s, _ in COHORTS if s == "Side 2"],
}


for i in range(1, steps):
    # carry forward last-tick counts
    for tag in state:
        state[tag][i] = state[tag][i - 1]

    # per-side fire-power  P = Σ (count × k × Ê)
    P = {
        side: sum(state[tag][i] * kvals[tag] * E[tag]
                  for tag in tags)
        for side, tags in side_tags.items()
    }

    # kill budgets this tick (both sides shoot)
    K = {side: sigma * P[side] * dt for side in P}

    # apply losses proportionally to the opposite side
    for side, foe in (("Side 1", "Side 2"), ("Side 2", "Side 1")):
        foe_total = sum(state[tag][i] * vuln[tag]
            for tag in side_tags[foe]) + EPS
        
        if foe_total == 0:
            continue
        for tag in side_tags[foe]:
            hits = K[side] * (state[tag][i] * vuln[tag]) / foe_total
            loss = hits * vuln[tag]
            state[tag][i] = max(0.0, state[tag][i] - loss)


# ───────── Plot ─────────
fig, ax = plt.subplots()
styles = ["-", "--", "-.", ":", "dashdot", "dotted", (0, (3, 1, 1, 1))]
for (tag, _, _), sty in zip(COHORTS, styles):
    ax.plot(_time, state[tag], label=tag, linestyle=sty, linewidth=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Aircraft remaining")
ax.set_title("Attrition – Six‑Cohort Duel (Logistic Δ‑based kills)")
ax.legend()
st.pyplot(fig)

# ───────── Metrics summary ─────────
side1_name = "NATO⁺UKR"
side2_name = "CHN⁺RUS"

s1_end = sum(state[t][-1] for t in side_tags["Side 1"])
s2_end = sum(state[t][-1] for t in side_tags["Side 2"])
col1, col2 = st.columns(2)
col1.metric("Side 1 survivors", f"{s1_end:.1f} ({side1_name})")
col2.metric("Side 2 survivors", f"{s2_end:.1f} ({side2_name})")

st.caption(
    "Remapped effectiveness via anchor‑ratio logistic mapping + magnitude‑advantage kills. ✈️🧮 #vibecoding"
)

