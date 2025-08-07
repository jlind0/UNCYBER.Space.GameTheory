from math import exp, log
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
with st.expander("About this app", expanded=False):
    st.markdown(
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
    )
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
DEFAULT_COHORTS = [
    ("NATO", "5-Gen"),
    ("NATO", "4-Gen"),
    ("Ukraine", "4-Gen"),
    ("China", "5-Gen"),
    ("China", "4-Gen"),
    ("Russia", "5-Gen"),
    ("Russia", "4-Gen"),
]
COHORT_DEFAULTS = {
    "NATO": {
        "5-Gen":    {"count": 25, "k": 0.07,  "orda": (0.30, 0.30, 0.25), "vuln": 0.65},
        "4-Gen":    {"count": 20, "k": 0.045, "orda": (0.30, 0.30, 0.25), "vuln": 0.95},
    },
    "Ukraine": {
        "4-Gen":    {"count": 30, "k": 0.035, "orda": (0.30, 0.25, 0.25), "vuln": 1.15},
    },
    "China": {
        "5-Gen":    {"count": 20, "k": 0.08,  "orda": (0.30, 0.30, 0.25), "vuln": 0.75},
        "4-Gen":    {"count": 20, "k": 0.025, "orda": (0.25, 0.30, 0.25), "vuln": 1.10},
    },
    "Russia": {
        "5-Gen":    {"count": 10, "k": 0.065, "orda": (0.25, 0.30, 0.25), "vuln": 0.90},
        "4-Gen":    {"count": 30, "k": 0.035, "orda": (0.25, 0.30, 0.25), "vuln": 1.25},
    },
}
AIRFRAME_DEFAULTS = {
    "5-Gen": {"count": 25, "k": 0.07, "orda": (0.30, 0.30, 0.25), "vuln": 1.00},
    "4-Gen": {"count": 60, "k": 0.05, "orda": (0.25, 0.25, 0.25), "vuln": 1.00},
}
if 'cohorts' not in st.session_state:
    st.session_state.cohorts = DEFAULT_COHORTS.copy()
if "nations" not in st.session_state:
    st.session_state.nations = {
        "NATO":    {"coalition": "Allied", "family": "cobb"},
        "Ukraine": {"coalition": "Allied", "family": "power"},
        "China":   {"coalition": "Asia",   "family": "ces"},
        "Russia":  {"coalition": "Asia",   "family": "logit"},
    }

if "airframes" not in st.session_state:
    st.session_state.airframes = {
        "5-Gen": {"count": 25, "k": 0.07, "vuln": 0.75, "orda": (0.30, 0.30, 0.25)},
        "4-Gen": {"count": 60, "k": 0.05, "vuln": 1.00, "orda": (0.25, 0.25, 0.25)},
    }

# 2) Sidebar: add nations
with st.sidebar.expander("➕ Add Nation", expanded=False):
    new_nat = st.text_input("Nation name", key="new_nation_name")
    coal   = st.text_input("Coalition",  key="new_nation_coal")
    fam    = st.selectbox("Family", ["cobb","power","ces","logit"], key="new_nation_fam")
    if st.button("Create Nation"):
        if new_nat and new_nat not in st.session_state.nations:
            st.session_state.nations[new_nat] = {"coalition": coal, "family": fam}
            st.success(f"Added nation “{new_nat}”")
        else:
            st.error("Enter a unique nation name.")

# 3) Sidebar: add airframes
with st.sidebar.expander("➕ Add Airframe", expanded=False):
    new_af     = st.text_input("Airframe label (e.g. “6-Gen”)", key="new_af_name")
    cnt        = st.number_input("Default count", min_value=0, max_value=500, value=10, step=1, key="new_af_cnt")
    k_val      = st.number_input("Default k",    min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="new_af_k")
    vuln_val   = st.number_input("Default vuln", min_value=0.0, max_value=2.0, value=1.0, step=0.05, key="new_af_vuln")
    orda_A     = st.slider("O", 0.0, 1.0, 0.3, 0.01, key="new_af_ordaA")
    orda_R     = st.slider("R", 0.0, 1.0, 0.3, 0.01, key="new_af_ordaR")
    orda_D     = st.slider("D", 0.0, 1.0, 0.25,0.01, key="new_af_ordaD")
    if st.button("Create Airframe"):
        if new_af and new_af not in st.session_state.airframes:
            # normalize ORDA if it doesn't sum to 1
            total = orda_A+orda_R+orda_D
            shares = (orda_A/total, orda_R/total, orda_D/total)
            st.session_state.airframes[new_af] = {
                "count": cnt, "k": k_val, "vuln": vuln_val, "orda": shares
            }
            st.success(f"Added airframe “{new_af}”")
        else:
            st.error("Enter a unique airframe label.")
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
if 'cohorts' not in st.session_state:
    st.session_state.cohorts = DEFAULT_COHORTS.copy()
with st.sidebar.expander("🛠 Scenario Editor", expanded=False):
    st.write("Add a new cohort (nation + airframe):")
    new_nation   = st.selectbox("Nation",   list(st.session_state.nations.keys()), key="new_cohort_nation")
    new_airframe = st.selectbox("Airframe", list(st.session_state.airframes.keys()), key="new_cohort_airframe")
    if st.button("Add Cohort"):
        candidate = (new_nation, new_airframe)
        if candidate in st.session_state.cohorts:
            st.warning(f"Cohort {candidate} already exists.")
        else:
            st.session_state.cohorts.append(candidate)
            st.success(f"Added cohort {candidate}")
            st.rerun()

for idx, (nation, airframe) in enumerate(st.session_state.cohorts):
    coal  = st.session_state.nations[nation]["coalition"]
    fam   = st.session_state.nations[nation]["family"]
    cols  = st.columns((3, 1))

    # show “Nation (coalition, family) – Airframe”
    cols[0].write(f"{nation} ({coal}, {fam}) — {airframe}")

    # delete button keyed by both nation & airframe
    if cols[1].button("Delete", key=f"delete-{nation}-{airframe}"):
        # 1) remove the cohort tuple itself
        st.session_state.cohorts.pop(idx)

        # 2) remove any widgets/session keys tied to this cohort
        prefix = f"{nation}-{airframe}"
        for k in list(st.session_state.keys()):
            if k.startswith(prefix):
                del st.session_state[k]
            st.rerun()
def _update_family(nation, selectbox_key):
    # copy the new choice into your nations dict
    st.session_state.nations[nation]["family"] = st.session_state[selectbox_key]
    # then force a rerun so everything downstream picks up the new family
    st.rerun()
# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("Force counts, lethality & families")

    # ————————————— Aggregation families (per nation) —————————————
    st.subheader("Aggregation family by nation")
    for nation in st.session_state.nations:
        # grab current, then let user override
        select_key = f"fam-{nation}"
        curr = st.session_state.nations[nation]["family"]
        new_fam = st.selectbox(
            f"Family for {nation}",
            FAMILIES,
            index=FAMILIES.index(curr),
            key=select_key,
            on_change=_update_family,
            args = (nation, select_key)
        )
        st.session_state.nations[nation]["family"] = new_fam

    # —————————— now your existing counts/k/kvals/shares/vulns loops ——————————
    counts = {}
    kvals  = {}
    famsel = {}
    shares = {}
    vulns  = {}

    for nation, airframe in st.session_state.cohorts:
        
        presets = (
            COHORT_DEFAULTS
              .get(nation, {})
              .get(airframe, AIRFRAME_DEFAULTS[airframe])
        )

        key = f"{nation}-{airframe}"
        with st.expander(key, expanded=False):
            
            counts[key] = st.slider(
                f"{nation} {airframe} count", 0, 200,
                presets["count"], 1, key=f"{key}-count"
            )
            kvals[key] = st.slider(
                f"k ({nation} {airframe})", 0.0, 0.2,
                presets["k"], 0.01, key=f"{key}-k"
            )
            shares[key] = share_sliders(
                f"{nation} {airframe}", presets["orda"], pfx=""
            )
            vulns[key] = st.slider(
                f"Vulnerability ({nation} {airframe})", 0.5, 1.5,
                presets["vuln"], 0.05, key=f"{key}-vuln"
            )

            # **new**: record which family this cohort uses
            famsel[key] = st.session_state.nations[nation]["family"]
        
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

E_raw = {
    t: raw_E(shares[t], famsel[t])
    for t in shares
}
E_scaled = {t: E_raw[t] / (E0[famsel[t]] + EPS) for t in E_raw}
E = {t: logistic_remap(E_scaled[t], lam_E, mid_E) for t in E_scaled}


# ───────── Simulation arrays ─────────
steps = int(horizon / dt) + 1
_time = np.linspace(0, horizon, steps)
# create state arrays for each cohort in the dynamic list
state = {
    f"{nation}-{airframe}": np.empty(steps)
    for nation, airframe in st.session_state.cohorts
}

for t in state:
    state[t][0] = counts[t]

side_tags = {}
for nation, airframe in st.session_state.cohorts:
    key = f"{nation}-{airframe}"
    coal = st.session_state.nations[nation]["coalition"]
    side_tags.setdefault(coal, []).append(key)


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
    sides = list(side_tags.keys())
    # apply losses proportionally to the opposite side
    for side in sides:
        for foe in sides:
            if foe == side:
                continue
            foe_total = sum(
                state[tag][i] * vulns[tag]
            for tag in side_tags[foe]
            ) + EPS
    # total enemy pressure from "foe" on "side"
            attr_rate = sum(
                E[enemy] * state[enemy][i]
                for enemy in side_tags[foe]
            )
            
            if foe_total == 0:
                continue
            for tag in side_tags[foe]:
                effective_K = K[side] * (attr_rate / (sum(state[e][i] for e in side_tags[foe]) + EPS))
                hits = hits = effective_K * (state[tag][i] * vulns[tag]) / foe_total
                loss = hits * vulns[tag]
                state[tag][i] = max(0.0, state[tag][i] - loss)


# ───────── Plot ─────────
fig, ax = plt.subplots()
from itertools import cycle
styles = ["-", "--", "-.", ":", "dashdot", "dotted", (0, (3, 1, 1, 1))]
style_iter = cycle(styles)
 # Plot each cohort series; will re-run automatically when cohorts change
for nation, airframe in st.session_state.cohorts:
    key = f"{nation}-{airframe}"
    sty = next(style_iter)
    ax.plot(_time, state[key], label=key, linestyle=sty, linewidth=2)
ax.set_xlabel("Itterations")
ax.set_ylabel("Aircraft remaining")
ax.set_title("Attrition – (Logistic Δ‑based kills)")
ax.legend()
st.pyplot(fig)

# ───────── Metrics summary ─────────
side1_name = "NATO⁺UKR"
side2_name = "CHN⁺RUS"

 # compute survivors for each side dynamically
metrics = {coal: sum(state[t][-1] for t in tags) for coal, tags in side_tags.items()}
cols = st.columns(len(metrics))
for col, (coal, survivors) in zip(cols, metrics.items()):
    col.metric(f"{coal} survivors", f"{survivors:.1f}")