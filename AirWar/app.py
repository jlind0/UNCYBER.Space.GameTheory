from math import exp, log
from math import radians, sin, cos, asin, sqrt
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import folium
import copy
from streamlit_folium import st_folium
from folium.plugins import Draw, TimestampedGeoJson
import datetime
import pandas as pd
with st.expander("About this app", expanded=False):
    st.markdown(
        """
# OODA Attrition – State-of-the-Art Overview  

This Streamlit dashboard lets you **prototype, stress-test, and compare competing air-combat doctrines** by combining:

* **Modern decision-cycle theory** (OODA)  
* **State-of-the-art aggregation functions** drawn from economics, reliability engineering, and cognition research  
* **Interactive Bayesian sliders** for effort allocation and lethality

The result is a *rapid-fire lab* where planners can see how small shifts in doctrine or resource mix ripple through a multi-nation fight.

---

## 1  Function Families & When to Use Them

| Family | Mathematical Form | Intuition | When to assign it to a nation | Key coefficients |
|--------|-------------------|-----------|-------------------------------|------------------|
| **`power`** (generalised mean) | \\(E = \\bigl( \\alpha O^p + \\beta R^p + \\gamma D^p + \\delta A^p \\bigr)^{1/p}\\) | Smoothly blends inputs; tune *p* to move from max-like to min-like behaviour. | Nations whose *tempo* is limited by their **slowest** O/R/D/A phase (lower *p*) or whose strengths compound (higher *p*). | `alpha…delta` (phase weights), `p` (elasticity) |
| **`ces`** (Constant-Elasticity of Substitution) | Same form as `power`, but parameters are usually called *ρ*. | Widely used in combat modelling; controls how easily effort shifts between phases. | Forces that can fluidly re-task ISR assets or pilots → pick **high substitutability** (*ρ* near 0). Rigid doctrines → *ρ* farther from 0. | `alpha…delta`, `rho` |
| **`cobb`** (Cobb–Douglas) | \\(E = k\\,O^{\\alpha}R^{\\beta}D^{\\gamma}A^{\\delta}\\) | Pure multiplicative synergy; if one phase is 0, effectiveness is 0. | Highly integrated doctrines (e.g., USAF “fusion warfare”) where a single phase failure is catastrophic. | `k` (scale), `alpha…delta` |
| **`exp`** (Exponential saturation) | \\(E = c\\,[1 - e^{-\\lambda(O+R+D+A)}]\\) | Rapid gains early, diminishing returns later. | Conscript or low-tech forces: initial improvements help a lot, but plateau quickly. | `c` (cap), `lam` (rise rate) |
| **`logit`** (probabilistic trigger) | \\(E = \\bigl[1+e^{-(\\Sigma \\alpha_i \\ln P_i - \\theta)}\\bigr]^{-1}\\) | Interprets effectiveness as a **probability of seizing the initiative**. | Nations with doctrine centred on *critical decision points* (e.g., Russia’s emphasis on first-salvo advantage). | `alpha…delta`, `theta` (difficulty) |
| **`quad`** (full quadratic) | Weighted quadratic over all pairings. | Captures **synergies and trade-offs** explicitly (e.g., Observe-Orient cross-term). | Research or wargame labs exploring niche interactions; expensive but expressive. | `w1…w10` (weights) |
| **`hybrid`** (bespoke mix) | \\(E = k\\,O^{\\alpha}e^{-\\lambda R} + \\beta D + \\gamma \\sqrt{A}\\) | Semi-empirical formula merging power, decay, and surprise. | NATO-style composite doctrine: strong ISR surge (O), rapid R hamper, and decisive action spikes (A). | `k, alpha, beta, gamma, delta, lam` |

### Picking Families in Practice
1. **Doctrine survey** Tag each nation’s *training philosophy*—is it attrition-centric, manoeuvre-centric, or probability-of-kill focused?  
2. **Elasticity guess** If phases can substitute one another, favour `power`/`ces`; else use `cobb` or `logit`.  
3. **Complexity budget** For quick what-ifs, start with `cobb` or `power`. Use `quad` only when you have data to justify ten weights.  
4. **Tune coefficients** Start from the default table in *Settings → Coefficients*. Then calibrate against red-/blue-flag sortie data or Monte-Carlo runs.

---

## 2  ORDA Effort Shares and Why They Matter

The sliders labelled **O / R / D** set the *fraction of pilot & C2 effort* devoted to each phase; **A** is auto-filled to ensure \\(O+R+D+A=1\\).

* **Power-/CES-style families:** Effectiveness rises fastest when effort flows into the **most-weighted** phase; e.g., if `alpha > beta`, boosting *O* gives better marginal returns than *R*.  
* **Cobb–Douglas:** Decreasing any one share below ~0.1 sharply cuts \\(E\\). Balanced ORDA usually beats extreme specialisation.  
* **Exponential:** Returns plateau—after ~0.3 total effort per phase you get <5 percent gain. Good for modelling diminishing ISR returns.  
* **Logit:** Think *threshold*. Until the weighted log-sum crosses \\(\\theta\\), \\(E\\) is near 0; once past it, small extra effort lands outsized payoff.  
* **Hybrid:**   Observe fuels the first term; Reduce *R* (Radar deception) to avoid the negative exponent; high *A* boosts the square-root term—great for doctrines that strike after blinding radar.

### Quick-Start Heuristics
| Scenario | ORDA tweak | Expected outcome |
|----------|-----------|------------------|
| “Blind-then-strike” SEAD package | Raise *R* to 0.35, drop *D* | Hurts enemy detect track chain; `hybrid` or `logit` nations benefit most |
| Pilot training surge | Increase *D* (decision) for rookies | `power` family gains where *p* > 1; `cobb` nations still need Observe support |
| Swarm drones | Push *A* ≥ 0.5 | Only safe with `exp` (plateau) or `quad` (positive A² term) to avoid over-commit |

---

## 3  Using the Sliders

1. **Select Nation-level family** in *Sidebar → Aggregation family by nation*.  
2. **Adjust ORDA** inside each cohort expander. Watch the live **Attrition curve** reshape.  
3. **Fine-tune coefficients** under *Phase Weights*—hover for tool-tips.  
4. **Normalise E** with the *Logistic remap* (λ, midpoint) if cross-family scaling is desired.

---

### Further Reading
* Boyd, J. (1995) *Destruction and Creation* – foundational OODA essay.  
* Moffat, J. (2017) *Command and Control in Military Crises* – formal tempo models.  
* Kott, A. et al. (2020) “Game-Theoretic Combat Modelling with Heterogeneous Forces”, *Journal of Defense Modeling & Simulation*.
* OODA Game Theory Paper: [https://multiplex.studio/files/OODA-Game.pdf](https://multiplex.studio/files/OODA-Game.pdf)
* Source Code: [https://github.com/jlind0/UNCYBER.Space.GameTheory/blob/main/AirWar/app.py](https://github.com/jlind0/UNCYBER.Space.GameTheory/blob/main/AirWar/app.py)

Happy experimenting—may your model reveal the hidden corners of air-combat tempo!
        """
    )
# ────────────────────────────────────────────────
EPS = 1e-9
NONE_ZONE = "— none —"
st.set_page_config(page_title="OODA – Six‑Cohort Duel", layout="wide")
st.title("OODA Attrition – NATO & Ukraine vs China & Russia (4 + 5 Gen mix)")
# ───────── Time-series map helpers ─────────
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _lerp_latlon(p1, p2, t):
    # p = (lat, lon)
    return (_lerp(p1[0], p2[0], t), _lerp(p1[1], p2[1], t))

def _ring_point_latlon(wpts_latlon, u: float):
    """
    Interpolate along an ordered waypoint ring (lat,lon) with parameter u∈[0,1).
    If <2 waypoints, just return the single point (hover at centroid handled by caller).
    """
    if not wpts_latlon:
        return None
    if len(wpts_latlon) == 1:
        return wpts_latlon[0]
    n = len(wpts_latlon)
    total = n  # unit length per leg
    x = (u % 1.0) * total
    i = int(x) % n
    j = (i + 1) % n
    t = x - int(x)
    return _lerp_latlon(wpts_latlon[i], wpts_latlon[j], t)

def _position_at_time_sec(v: dict, t: float, cruise_kmh: float, tos_hours: float, ring_latlon):
    """
    Compute (lat, lon) for a cohort view 'v' at absolute time 't' seconds.
    Pattern: base -> transit -> on-station patrol -> return -> repeat.
    """
    base = v.get("base")          # (lat, lon) or None
    zcent = v.get("zone_centroid")# (lat, lon) or None
    if not base or not zcent:
        # If either missing, pin to whichever exists
        return base or zcent or (45.0, 34.0)

    # transit time (seconds)
    dist_km = haversine_km(base[0], base[1], zcent[0], zcent[1])
    t_transit = 3600.0 * (dist_km / max(1e-6, cruise_kmh))
    t_station = 3600.0 * max(0.0, tos_hours)
    cycle = max(1e-3, 2.0 * t_transit + t_station)
    tm = t % cycle

    # outbound
    if tm < t_transit:
        f = tm / max(1e-6, t_transit)
        return _lerp_latlon(base, zcent, f)
    # on-station patrol
    if tm < t_transit + t_station and t_station > 0:
        u = (tm - t_transit) / max(1e-6, t_station)  # 0..1
        if ring_latlon and len(ring_latlon) >= 2:
            return _ring_point_latlon(ring_latlon, u)
        # No ring → hover on centroid
        return zcent
    # inbound
    rem = tm - (t_transit + t_station)
    f = rem / max(1e-6, t_transit)
    return _lerp_latlon(zcent, base, f)

def _build_ts_geojson(view: dict, duration_min: int, dt_sec: int,
                      cruise_kmh: float, tos_hours: float) -> dict:
    """
    Build a FeatureCollection with MultiPoint geometry and matching ISO8601 timestamps.
    One feature per **allocated base** (not just per cohort), colored by coalition.
    """
    start = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    steps = max(1, int((duration_min * 60) / max(1, dt_sec)))
    fc = {"type": "FeatureCollection", "features": []}
    colors = {"Allied": "#2563eb", "Asia": "#dc2626"}

    for name, v in view.items():
        # precompute ring for patrol
        ring = []
        zname = v.get("zone")
        if zname and zname != NONE_ZONE:
            ring = [(latlon[1], latlon[0]) for latlon in st.session_state.zone_waypoints.get(zname, [])]
            ring = [(p[0], p[1]) for p in ring]  # (lat,lon)

        # choose base variants: either explicit multi-bases or the single default
        base_variants = v.get("bases_multi") or [{"bname": None, "latlon": v.get("base"), "share": 1.0}]

        for b in base_variants:
            base_latlon = b["latlon"]
            if not base_latlon:
                continue
            v2 = dict(v)
            v2["base"] = base_latlon  # override just for the path computation

            coords, times = [], []
            for k in range(steps):
                t = k * dt_sec
                lat, lon = _position_at_time_sec(v2, t, cruise_kmh, tos_hours, ring)
                coords.append([lon, lat])
                times.append((start + datetime.timedelta(seconds=t)).isoformat())

            coal = v.get("coalition", "")
            color = colors.get(coal, "#10b981")
            label = f"{name} @{b['bname']}" if b.get("bname") else name

            feature = {
                "type": "Feature",
                "geometry": {"type": "MultiPoint", "coordinates": coords},
                "properties": {
                    "times": times,
                    "style": {"color": color, "weight": 2, "opacity": 0.9},
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": color, "fillOpacity": 0.9, "stroke": True,
                        "radius": 6, "color": color,
                    },
                    "popup": label,
                },
            }
            fc["features"].append(feature)

    return fc


def seed_zones_for_attrition(fill_only: bool = True, overlap_zone: str | None = None):
    """
    Assign all cohorts to a single overlap zone so dist_km≈0 and attrition occurs.
    If overlap_zone is None, prefer Crimea-Central, then other Crimea tiles.
    """
    # Pick an overlap zone that actually exists
    if overlap_zone is None:
        for z in ("Crimea-Central", "Crimea-NE", "Crimea-NW", "Crimea-SE", "Crimea-SW"):
            if z in st.session_state.deployment_zones:
                overlap_zone = z
                break
        else:
            overlap_zone = NONE_ZONE  # falls back cleanly

    st.session_state.setdefault("cohort_zone_tasks", {})

    for name, _nation, airframe in st.session_state.cohorts:
        current = st.session_state.cohort_zone_tasks.get(name, {"zone": NONE_ZONE, "fraction": 0.0})
        if fill_only and current.get("zone") not in (None, "", NONE_ZONE):
            continue

        stealthy = any(tag in airframe for tag in ("F-35", "J-20", "Su-57"))
        frac = 0.70 if stealthy else 0.60
        st.session_state.cohort_zone_tasks[name] = {"zone": overlap_zone, "fraction": frac}

# ───────────────── Aggregation families ──────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def zone_centroid(rings):
    # simple average of outer ring
    if not rings: return (0.0, 0.0)
    outer = rings[0]
    xs = [p[0] for p in outer]; ys = [p[1] for p in outer]
    return (sum(xs)/len(xs), sum(ys)/len(ys))  # (lon, lat)
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
# ───────── Cohort defaults: specific airframes ─────────

DEFAULT_COHORTS = [
    ("Lightning-1",  "NATO",    "F-35A"),              # F-35 Lightning II
    ("Viper-1",      "NATO",    "F-16C"),              # F-16 Fighting Falcon (“Viper”)
    ("Typhoon-1",    "NATO",    "Eurofighter Typhoon"),

    ("Fulcrum-1",    "Ukraine", "MiG-29"),             # MiG-29 Fulcrum
    ("Flanker-1",    "Ukraine", "Su-27"),              # Su-27 Flanker
    ("Viper-UKR-1",  "Ukraine", "F-16C"),              # second F-16 cohort, Ukraine

    ("Dragon-1",     "China",   "J-20"),               # J-20 Mighty Dragon
    ("Firebird-1",   "China",   "J-10C"),              # J-10C Firebird / Vigorous Dragon

    ("Felon-1",      "Russia",  "Su-57"),              # Su-57 Felon
    ("Flanker-E-1",  "Russia",  "Su-35S"),             # Su-35 (Flanker-E)
]

COHORT_DEFAULTS = {
    "NATO": {
        "F-35A":              {"count": 24, "k": 0.080, "orda": (0.32, 0.28, 0.20), "vuln": 0.65},
        "F-16C":              {"count": 30, "k": 0.055, "orda": (0.30, 0.30, 0.25), "vuln": 1.00},
        "Eurofighter Typhoon":{"count": 12, "k": 0.065, "orda": (0.30, 0.28, 0.22), "vuln": 0.90},
    },
    "Ukraine": {
        "MiG-29":             {"count": 25, "k": 0.035, "orda": (0.30, 0.25, 0.25), "vuln": 1.15},
        "Su-27":              {"count": 25, "k": 0.040, "orda": (0.28, 0.26, 0.26), "vuln": 1.10},
        "F-16C":              {"count": 15, "k": 0.045, "orda": (0.30, 0.30, 0.25), "vuln": 1.05},
    },
    "China": {
        "J-20":               {"count": 18, "k": 0.080, "orda": (0.32, 0.28, 0.20), "vuln": 0.70},
        "J-10C":              {"count": 30, "k": 0.050, "orda": (0.28, 0.30, 0.22), "vuln": 1.00},
    },
    "Russia": {
        "Su-57":              {"count": 25, "k": 0.075, "orda": (0.32, 0.28, 0.20), "vuln": 0.75},
        "Su-35S":             {"count": 30, "k": 0.060, "orda": (0.30, 0.28, 0.22), "vuln": 0.85},
    },
}

# ───────── Generic fall-back for any airframe label not in COHORT_DEFAULTS ─────────
AIRFRAME_DEFAULTS = {
    "F-35A":               {"count": 24, "k": 0.080, "orda": (0.32, 0.28, 0.20), "vuln": 0.65},
    "F-16C":               {"count": 60, "k": 0.045, "orda": (0.30, 0.30, 0.25), "vuln": 1.00},
    "Eurofighter Typhoon": {"count": 40, "k": 0.055, "orda": (0.30, 0.28, 0.22), "vuln": 0.90},
    "MiG-29":              {"count": 28, "k": 0.035, "orda": (0.30, 0.25, 0.25), "vuln": 1.15},
    "Su-27":               {"count": 24, "k": 0.040, "orda": (0.28, 0.26, 0.26), "vuln": 1.10},
    "J-20":                {"count": 20, "k": 0.080, "orda": (0.32, 0.28, 0.20), "vuln": 0.70},
    "J-10C":               {"count": 40, "k": 0.050, "orda": (0.28, 0.30, 0.22), "vuln": 1.00},
    "Su-57":               {"count": 12, "k": 0.075, "orda": (0.32, 0.28, 0.20), "vuln": 0.75},
    "Su-35S":              {"count": 30, "k": 0.060, "orda": (0.30, 0.28, 0.22), "vuln": 0.85},
}
NATION_BASES = {
    "NATO": {
        "Incirlik AB (TUR)":      (37.00, 35.43),   # Adana, Turkey
        "M. Kogălniceanu AB (RO)":(44.36, 28.49),   # Black-Sea coast, Romania
    },
    "Ukraine": {
        "Kulbakino Mykolaiv":     (46.44, 32.45),
        "Odesa Intl":             (46.43, 30.17),
    },
    "Russia": {
        "Belbek AB (Sevastopol)": (44.69, 33.57),
        "Saki AB (Crimea)":       (45.05, 33.59),
    },
    "China": {
        "Khmeimim Det (Syria)":   (35.41, 35.97),
    },
}
# ───────── Default ORDA-family coefficients per nation × airframe ─────────
# Keys must match nation-airframe pairs used in COHORT_DEFAULTS & DEFAULT_COHORTS
DEFAULT_COEFFS = {
    "NATO": {
        "F-35A": dict(alpha=0.32, beta=0.22, gamma=0.18, delta=0.28, k=0.6,  lam=0.05),  # hybrid
        "F-16C": dict(alpha=0.28, beta=0.18, gamma=0.22, delta=0.32, k=0.6,  lam=0.05),
        "Eurofighter Typhoon":
                     dict(alpha=0.30, beta=0.20, gamma=0.22, delta=0.28, k=0.6,  lam=0.05),
    },
    "Ukraine": {
        "MiG-29": dict(alpha=0.30, beta=0.30, gamma=0.20, delta=0.20, p=0.90),          # power
        "Su-27":  dict(alpha=0.28, beta=0.32, gamma=0.18, delta=0.22, p=0.95),
        "F-16C": dict(alpha=0.28, beta=0.32, gamma=0.18, delta=0.22, p=0.95),
    },
    "China": {
        "J-20":  dict(alpha=0.28, beta=0.28, gamma=0.22, delta=0.22, rho=1.40),         # CES
        "J-10C": dict(alpha=0.24, beta=0.24, gamma=0.28, delta=0.24, rho=1.60),
    },
    "Russia": {
        "Su-57":  dict(alpha=0.32, beta=0.32, gamma=0.18, delta=0.18, theta=0.08),      # logit
        "Su-35S": dict(alpha=0.28, beta=0.28, gamma=0.22, delta=0.22, theta=0.12),
    },
}
AIRFRAME_LOADOUTS = {
    "F-35A": {
        "AIM-120C": {"default": 4, "max": 6},
        "AIM-9X":   {"default": 2, "max": 2},
    },
    "F-16C": {
        "AIM-120C": {"default": 2, "max": 6},
        "AIM-9M":   {"default": 2, "max": 2},
    },
    "Eurofighter Typhoon": {
        "Meteor": {"default": 4, "max": 6},
        "ASRAAM": {"default": 2, "max": 4},
    },
    "MiG-29": {
        "R-27R": {"default": 2, "max": 4},
        "R-73":  {"default": 2, "max": 6},
    },
    "Su-27": {
        "R-27ER": {"default": 2, "max": 6},
        "R-73":   {"default": 2, "max": 4},
    },
    "J-20": {
        "PL-15": {"default": 4, "max": 4},
        "PL-10": {"default": 2, "max": 2},
    },
    "J-10C": {
        "PL-15": {"default": 2, "max": 4},
        "PL-8":  {"default": 2, "max": 2},
    },
    "Su-57": {
        "R-77M":  {"default": 4, "max": 4},
        "R-74M2": {"default": 2, "max": 2},
    },
    "Su-35S": {
        "R-77-1": {"default": 4, "max": 6},
        "R-73":   {"default": 2, "max": 2},
    },
}
def compute_contact_time(view: dict, side_tags: dict, cap_seconds: float) -> float:
    sides = list(side_tags.keys())
    if len(sides) < 2:
        return 0.0
    s1, s2 = sides[0], sides[1]
    def _earliest_first_on(side):
        vals = [view[n]["first_on"] for n in side_tags[side] if "first_on" in view[n]]
        return min(vals) if vals else 0.0
    t_contact = max(_earliest_first_on(s1), _earliest_first_on(s2))
    return float(min(t_contact, cap_seconds))

def latlon(p):
    """Ensure (lat, lon) ordering from tuples that might be (lon, lat)."""
    if p is None:
        return None
    # Heuristic: latitude is in [-90, 90]
    a, b = p
    if -90 <= a <= 90 and not (-90 <= b <= 90):
        return (a, b)  # (lat, lon)
    return (b, a)      # assume (lon, lat) -> (lat, lon)

def centroid_latlon_from_zone(zname: str):
    rings = st.session_state.deployment_zones.get(zname, [])
    if not rings:
        return None
    zlon, zlat = zone_centroid(rings)  # your centroid returns (lon, lat)
    return (zlat, zlon)

def auto_seed_zone_tasks(fill_only: bool = True):
    all_zone_names = set(st.session_state.deployment_zones.keys())

    # coalition preferences (filtered to what exists)
    pref_allied = [z for z in ["Crimea-NW","Crimea-NE","Crimea-Central"] if z in all_zone_names] or list(all_zone_names)
    pref_asia   = [z for z in ["Crimea-SE","Crimea-SW","Crimea-Central"] if z in all_zone_names] or list(all_zone_names)

    def _centroid_latlon(zname):
        rings = st.session_state.deployment_zones.get(zname, [])
        if not rings: return None
        lon, lat = zone_centroid(rings)  # your helper returns (lon, lat)
        return (lat, lon)

    def _nearest_zone(base_latlon, candidates):
        if not base_latlon or not candidates: return NONE_ZONE
        blat, blon = base_latlon
        best, best_d = NONE_ZONE, float("inf")
        for z in candidates:
            zl = _centroid_latlon(z)
            if not zl: continue
            d = haversine_km(blat, blon, zl[0], zl[1])
            if d < best_d: best, best_d = z, d
        return best

    # ensure dict exists
    st.session_state.setdefault("cohort_zone_tasks", {})

    for name, nation, airframe in st.session_state.cohorts:
        current = st.session_state.cohort_zone_tasks.get(name, {"zone": NONE_ZONE, "fraction": 0.0})
        if fill_only and current.get("zone") not in (None, "", NONE_ZONE):
            continue  # don't override user's choice

        coal = st.session_state.nations[nation]["coalition"]
        bases = list(st.session_state.bases.get(nation, {}).values())
        base_latlon = bases[0] if bases else None
        pool = pref_allied if coal == "Allied" else pref_asia
        zpick = _nearest_zone(base_latlon, pool)

        stealthy = any(tag in airframe for tag in ("F-35", "J-20", "Su-57"))
        frac = 0.70 if stealthy else 0.60

        st.session_state.cohort_zone_tasks[name] = {"zone": zpick, "fraction": frac}
WEAPON_LIBRARY = {
    "AIM-120C": {"range_km": 105, "pk": 0.55},
    "AIM-9X":   {"range_km":  35, "pk": 0.30},
    "AIM-9M":   {"range_km":  30, "pk": 0.25},
    "Meteor":   {"range_km": 150, "pk": 0.60},
    "ASRAAM":   {"range_km":  40, "pk": 0.32},
    "R-27R":    {"range_km":  80, "pk": 0.35},
    "R-27ER":   {"range_km": 110, "pk": 0.40},
    "R-73":     {"range_km":  30, "pk": 0.25},
    "PL-15":    {"range_km": 150, "pk": 0.55},
    "PL-10":    {"range_km":  40, "pk": 0.32},
    "PL-8":     {"range_km":  35, "pk": 0.25},
    "R-77M":    {"range_km": 160, "pk": 0.55},
    "R-77-1":   {"range_km": 110, "pk": 0.45},
    "R-74M2":   {"range_km":  40, "pk": 0.30},
}

# Keep an editable copy in session state
if "weapons" not in st.session_state:
    st.session_state.weapons = {w: stats.copy() for w, stats in WEAPON_LIBRARY.items()}
if 'cohort_params' not in st.session_state:
    st.session_state.cohort_params = {}
if 'cohorts' not in st.session_state:
    st.session_state.cohorts = DEFAULT_COHORTS.copy()
if "cohort_weapons" not in st.session_state:
    st.session_state.cohort_weapons = {}
if "nations" not in st.session_state:
    st.session_state.nations = {
        "NATO":    {"coalition": "Allied", "family": "hybrid"},
        "Ukraine": {"coalition": "Allied", "family": "power"},
        "China":   {"coalition": "Asia",   "family": "ces"},
        "Russia":  {"coalition": "Asia",   "family": "logit"},
    }
if "zone_waypoints" not in st.session_state:
    st.session_state.zone_waypoints = {}
if "airframes" not in st.session_state:
    st.session_state.airframes = AIRFRAME_DEFAULTS.copy()
if "airframe_loadouts" not in st.session_state:
    st.session_state.airframe_loadouts = copy.deepcopy(AIRFRAME_LOADOUTS)
# editable copy of bases
if "bases" not in st.session_state:
    st.session_state.bases = {n: d.copy() for n, d in NATION_BASES.items()}
# cohort → { base_name: qty }
# Init in session_state if not exists
# ───────── Deployment zones: defaults & de-duped inits ─────────
if "deployment_zones" not in st.session_state:
    st.session_state.deployment_zones = {
        # Rectangles sized for Crimea as you already had; kept as-is
        "Crimea-NW":       [[(32.8,46.2),(33.2,46.2),(33.2,45.8),(32.8,45.8)]],
        "Crimea-SW":       [[(33.0,44.8),(33.4,44.8),(33.4,44.4),(33.0,44.4)]],
        "Crimea-SE":       [[(35.0,45.2),(35.4,45.2),(35.4,44.8),(35.0,44.8)]],
        "Crimea-Central":  [[(33.8,45.6),(34.2,45.6),(34.2,45.2),(33.8,45.2)]],
        "Crimea-NE":       [[(34.8,46.2),(35.2,46.2),(35.2,45.8),(34.8,45.8)]],
        "Crimea-All":      [[(32.3,46.3),(36.6,46.3),(36.6,44.4),(32.3,44.4)]],
    }

# Ensure these exist exactly once
if "zone_waypoints" not in st.session_state:
    st.session_state.zone_waypoints = {
        # Default racetrack CAPs (lon,lat) — clockwise 4-point loops
        "Crimea-NW":      [(33.05,46.15),(33.15,46.15),(33.15,45.85),(33.05,45.85)],
        "Crimea-NE":      [(35.00,46.15),(35.15,46.15),(35.15,45.90),(35.00,45.90)],
        "Crimea-Central": [(33.90,45.55),(34.10,45.55),(34.10,45.30),(33.90,45.30)],
        "Crimea-SW":      [(33.05,44.75),(33.30,44.75),(33.30,44.55),(33.05,44.55)],
        "Crimea-SE":      [(35.05,45.15),(35.30,45.15),(35.30,44.95),(35.05,44.95)],
    }

if "zone_effects" not in st.session_state:
    # Environment/A2AD multipliers (1.0 = neutral)
    st.session_state.zone_effects = {
        "Crimea-NW":      {"E_mult": 0.95, "vuln_mult": 1.10},  # moderate SAM, some clutter
        "Crimea-NE":      {"E_mult": 0.95, "vuln_mult": 1.10},
        "Crimea-Central": {"E_mult": 0.90, "vuln_mult": 1.20},  # densest IADS
        "Crimea-SW":      {"E_mult": 0.92, "vuln_mult": 1.15},
        "Crimea-SE":      {"E_mult": 0.92, "vuln_mult": 1.15},
        "Crimea-All":     {"E_mult": 0.93, "vuln_mult": 1.15},
    }
def _infer_end_count(tag: str, N0: int) -> int:
    """
    Try to infer the ending count for `tag` from whatever the sim produced.
    Prefers absolute counts; falls back to alive fraction arrays; else N0.
    """
    ss = st.session_state
    # Absolute series of counts per cohort
    series_counts = getattr(ss, "series_counts", {})
    if isinstance(series_counts, dict) and tag in series_counts and len(series_counts[tag]) > 0:
        try:
            return int(round(float(series_counts[tag][-1])))
        except Exception:
            pass

    # Fraction alive per cohort
    series_alive_frac = getattr(ss, "series_alive_frac", {})
    if isinstance(series_alive_frac, dict) and tag in series_alive_frac and len(series_alive_frac[tag]) > 0:
        try:
            frac = float(series_alive_frac[tag][-1])
            frac = max(0.0, min(1.0, frac))
            return int(round(N0 * frac))
        except Exception:
            pass

    # Final snapshot map
    final_counts = getattr(ss, "final_counts", {})
    if isinstance(final_counts, dict) and tag in final_counts:
        try:
            return int(final_counts[tag])
        except Exception:
            pass

    return int(N0)

# --- Cohort zone assignment defaults ---
# --- Cohort zone assignment defaults (auto-seeded & distance-aware) ---
def _auto_seed_zone_tasks():
    # Preferred zone sets by coalition (filtered to what's actually defined)
    all_zone_names = set(st.session_state.deployment_zones.keys())
    pref_allied = [z for z in ["Crimea-NW", "Crimea-NE", "Crimea-Central"] if z in all_zone_names] or list(all_zone_names)
    pref_asia   = [z for z in ["Crimea-SE", "Crimea-SW", "Crimea-Central"] if z in all_zone_names] or list(all_zone_names)

    def _zone_centroid_latlon(zname):
        rings = st.session_state.deployment_zones.get(zname, [])
        if not rings: 
            return None
        lon, lat = zone_centroid(rings)  # existing helper returns (lon, lat)
        return (lat, lon)

    def _nearest_zone(base_latlon, candidates):
        if not base_latlon or not candidates:
            return "— none —"
        blat, blon = base_latlon
        best = None
        best_d = float("inf")
        for z in candidates:
            zl = _zone_centroid_latlon(z)
            if not zl:
                continue
            d = haversine_km(blat, blon, zl[0], zl[1])
            if d < best_d:
                best, best_d = z, d
        return best or "— none —"

    seeded = {}
    for name, nation, airframe in st.session_state.cohorts:
        coal = st.session_state.nations[nation]["coalition"]
        bases = list(st.session_state.bases.get(nation, {}).values())
        base_latlon = bases[0] if bases else None  # (lat, lon)

        # pick coalition's preferred pool, then nearest zone to the cohort's first base
        pool = pref_allied if coal == "Allied" else pref_asia
        zpick = _nearest_zone(base_latlon, pool)

        # reasonable default on-station fraction (stealth spend more time on CAP)
        stealthy = any(tag in airframe for tag in ("F-35", "J-20", "Su-57"))
        frac = 0.70 if stealthy else 0.60

        seeded[name] = {"zone": zpick, "fraction": frac}

    st.session_state.cohort_zone_tasks = seeded




# --- Cohort base allocation defaults ---
if "cohort_base_counts" not in st.session_state:
    st.session_state.cohort_base_counts = {}
    for name, nation, airframe in st.session_state.cohorts:
        nat_bases = list(st.session_state.bases.get(nation, {}))
        default_count = (
            COHORT_DEFAULTS
            .get(nation, {})
            .get(airframe, AIRFRAME_DEFAULTS[airframe])["count"]
        )
        if nat_bases:
            # assign all aircraft to the first base by default
            st.session_state.cohort_base_counts[name] = {nat_bases[0]: default_count}
        else:
            st.session_state.cohort_base_counts[name] = {"—": default_count}

# Initialize once, or when everything is unassigned
if "cohort_zone_tasks" not in st.session_state or \
   all(v.get("zone") in (None, "", NONE_ZONE) for v in st.session_state.cohort_zone_tasks.values()):
    seed_zones_for_attrition(fill_only=False)

# 2) Sidebar: add nations
if st.sidebar.button("Auto-assign zones (attrition)"):
    seed_zones_for_attrition(fill_only=False)
    st.rerun()

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
    """Expose O-, R-, D-sliders; A is auto-computed so that O+R+D+A = 1.
       If O+R+D > 1, auto-renormalize O/R/D to keep A >= 0."""
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

    s = O + R + D
    if s > 1.0:
        # renormalize to sum=1 so A=0
        O, R, D = O / s, R / s, D / s
        st.info("O + R + D exceeded 1; auto-renormalized to keep A ≥ 0.")

    A = 1.0 - (O + R + D)
    st.caption(f"A (auto) = {A:.2f}")
    return O, R, D, A

# ───────── Effectiveness helpers ─────────

def raw_E(shares, fam, params=None):
    """Compute raw effectiveness with optional overridden params."""
    base_params = GENERIC.get(fam, {})
    if params is None:
        params = base_params
    fn = FAMILY_FUNCS[fam]
    if fam == "quad":
        # merge quad weights with overrides
        p = GENERIC["quad"].copy()
        p.update(params)
        return fn(*shares, p)
    return fn(*shares, **params)


def logistic_remap(val: float, lam: float = 5.0, mid: float = 0.5) -> float:
    """Bound *val* to (0,1) using a logistic centred on *mid* with slope *lam*."""
    return 1.0 / (1.0 + exp(-lam * (val - mid)))
# --- Weapon helpers ---
def classify_weapon(name: str, bvr_threshold_km: float, weapons: dict) -> str:
    rng = weapons.get(name, {}).get("range_km", 0.0)
    return "BVR" if rng >= bvr_threshold_km else "WVR"

def weighted(pool: dict, get_val) -> float:
    tot = sum(pool.values())
    if tot <= 1e-9:
        return 0.0
    return sum(n * float(get_val(w)) for w, n in pool.items()) / tot

def soft_gate(distance_km: float, range_km: float, softness_km: float) -> float:
    if softness_km <= 1e-6:
        return 1.0 if distance_km <= range_km else 0.0
    x = (range_km - distance_km) / softness_km
    return 1.0 / (1.0 + np.exp(-x))
NONE_ZONE = "— none —"



def build_cohort_view(
    cohorts, counts, kvals, vulns, E,
    nations, bases,
    cohort_zone_tasks,
    deployment_zones, zone_effects,
    cohort_weapons, weapons,
    bvr_threshold_km: float,
    cruise_kmh: float,
    tos_hours: float,
):
    """
    Build a per-cohort view, but carry a 'bases_multi' list so maps can draw one
    track per allocated base. Availability and first_on are computed across bases.
    """
    view = {}
    for (name, nation, _af) in cohorts:
        coal = nations[nation]["coalition"]

        # ---- read this cohort's base allocation ----
        alloc = st.session_state.cohort_base_counts.get(name, {})  # {base_name: qty}
        nat_bases = bases.get(nation, {})                          # {base_name: (lat,lon)}

        base_recs = []
        total_q = 0
        for bname, qty in alloc.items():
            if qty and bname in nat_bases:
                blat, blon = nat_bases[bname]  # (lat,lon)
                base_recs.append({"bname": bname, "latlon": (blat, blon), "qty": int(qty)})
                total_q += int(qty)

        # Fallback: if nothing allocated, use nation's first base (if any)
        if not base_recs:
            bi = list(nat_bases.items())
            if bi:
                bname, (blat, blon) = bi[0]
                base_recs = [{"bname": bname, "latlon": (blat, blon), "qty": counts.get(name, 0)}]
                total_q = base_recs[0]["qty"]

        # ---- zone selection / centroid ----
        task = cohort_zone_tasks.get(name, {"zone": "— none —", "fraction": 0.0})
        zone = task.get("zone", "— none —")
        frac = float(task.get("fraction", 0.0))
        zcent = None
        if zone and zone != "— none —":
            rings = deployment_zones.get(zone, [])
            zlon, zlat = zone_centroid(rings)  # (lon, lat)
            zcent = (zlat, zlon)

        # ---- first_on (earliest from any allocated base) ----
        t_on = 0.0
        if zcent and base_recs:
            t_on = min(
                3600.0 * (haversine_km(b["latlon"][0], b["latlon"][1], zcent[0], zcent[1]) / max(1e-6, cruise_kmh))
                for b in base_recs
            )

        # ---- zone multipliers ----
        zm, vm = 1.0, 1.0
        if zone and zone != "— none —":
            eff = zone_effects.get(zone, {})
            zm = float(eff.get("E_mult", 1.0))
            vm = float(eff.get("vuln_mult", 1.0))

        # ---- availability aggregated over bases (weight by qty share) ----
        avail = 1.0
        if zcent and base_recs and total_q > 0:
            on_station_ratio = 0.0
            for b in base_recs:
                w = b["qty"] / total_q
                dist_km = haversine_km(b["latlon"][0], b["latlon"][1], zcent[0], zcent[1])
                t_transit_h = dist_km / max(1e-6, cruise_kmh)
                cycle_h = max(1e-6, 2.0 * t_transit_h + tos_hours)
                on_station_ratio += w * (tos_hours / cycle_h)
            avail = (1.0 - frac) + frac * on_station_ratio

        # ---- weapons / BVR-WVR buckets (unchanged) ----
        per_ac = cohort_weapons.get(name, {})
        pool_bvr, pool_wvr = {}, {}
        for w, qty_per_ac in per_ac.items():
            total = counts[name] * max(0, int(qty_per_ac))
            if total <= 0:
                continue
            bucket = classify_weapon(w, bvr_threshold_km, weapons)
            (pool_bvr if bucket == "BVR" else pool_wvr)[w] = total

        bvr_pk  = weighted(pool_bvr, lambda w: weapons.get(w, {}).get("pk", 0.0))
        bvr_rng = weighted(pool_bvr, lambda w: weapons.get(w, {}).get("range_km", 0.0))
        wvr_pk  = weighted(pool_wvr, lambda w: weapons.get(w, {}).get("pk", 0.0))
        wvr_rng = weighted(pool_wvr, lambda w: weapons.get(w, {}).get("range_km", 0.0))

        # pick a primary base (nearest to zone) as a reasonable single-base default
        primary_base = None
        if base_recs:
            if zcent:
                primary_base = min(
                    (b["latlon"] for b in base_recs),
                    key=lambda p: haversine_km(p[0], p[1], zcent[0], zcent[1])
                )
            else:
                primary_base = base_recs[0]["latlon"]

        view[name] = dict(
            coalition = coal,
            nation = nation,
            base = primary_base,                      # kept for backwards compat
            bases_multi = [
                {"bname": b["bname"], "latlon": b["latlon"], "share": (b["qty"]/total_q if total_q > 0 else 0.0)}
                for b in base_recs
            ],
            zone = zone,
            zone_centroid = zcent,
            first_on = t_on,
            avail = avail,
            E_eff = E[name] * zm,
            vuln_eff = vulns[name] * vm,
            k = kvals[name],
            N = counts[name],
            pool_bvr = pool_bvr, pool_wvr = pool_wvr,
            bvr_pk = bvr_pk, bvr_rng = bvr_rng,
            wvr_pk = wvr_pk, wvr_rng = wvr_rng,
        )
    return view

def simulate_fast(
    cohorts, view, side_tags, 
    horizon, dt,
    eng_rate_per_ac_sec, salvo_size,
    bvr_opportunity, bvr_geom_soft_km,
    reload_enabled, reload_rate_per_ac_sec,
):
    steps = int(horizon/dt) + 1
    T = np.linspace(0.0, horizon, steps)
    state = {c[0]: np.empty(steps) for c in cohorts}
    for (name, _, _) in cohorts:
        state[name][0] = view[name]["N"]

    # contact time = later of the two sides’ earliest first_on
    sides = list(side_tags.keys())
    if len(sides) >= 2:
        s1, s2 = sides[0], sides[1]
        t_contact = max(
            min([view[n]["first_on"] for n in side_tags[s1]]) if side_tags[s1] else 0.0,
            min([view[n]["first_on"] for n in side_tags[s2]]) if side_tags[s2] else 0.0,
        )
    else:
        t_contact = 0.0
    start_idx = max(1, min(int(t_contact/dt), steps-1))

    # flat until contact
    for i in range(1, start_idx):
        for (name,_,_) in cohorts:
            state[name][i] = state[name][i-1]

    quiet_clock = 0.0
    early_stop = st.session_state.get("early_stop_on_quiet", True)
    for i in range(start_idx, steps):
        # carry forward
        for (name,_,_) in cohorts:
            state[name][i] = state[name][i-1]

        # expected kills generated by each coalition this tick
        kills_by_side = {s: 0.0 for s in side_tags}

        # 1) for each coalition, compute total expected kills from BVR and WVR
        for side in side_tags:
            # (a) total on-station shooters (weighted by availability) and their “Êk”
            onstation = 0.0
            ek_sum = 0.0
            # shot opportunities and ammo pools per bucket
            bvr_shots_need = 0.0
            wvr_shots_need = 0.0
            # distances for geometry gating (use centroid vs centroid if both have zones)
            # build one representative distance to opponents (median of distances)
            dists = []
            for me in side_tags[side]:
                v = view[me]
                onstation += state[me][i] * v["avail"]
                ek_sum    += state[me][i] * v["E_eff"] * v["k"]
                # pairwise distances to foe cohorts with zones
                for foe_side, foes in side_tags.items():
                    if foe_side == side: 
                        continue
                    for foe in foes:
                        vz = view[foe]["zone_centroid"]
                        vm = v["zone_centroid"]
                        if vm and vz:
                            dists.append(haversine_km(vm[0], vm[1], vz[0], vz[1]))
            dist_km = (min(dists) if dists else 0.0)
            if "closure_kmh" in st.session_state:
                t_since_contact = max(0.0, (i - start_idx) * dt)  # seconds
                dist_km = max(0.0, dist_km - st.session_state["closure_kmh"] * (t_since_contact / 3600.0))

            # shot opportunities this tick
            shot_ops = onstation * eng_rate_per_ac_sec * dt
            # BVR geometric gate (0..1) using missiles-weighted BVR range across this side
            # Compute a side BVR range proxy (weighted average over all cohorts that have BVR ammo)
            bvr_rngs = []
            for me in side_tags[side]:
                if sum(view[me]["pool_bvr"].values()) > 0:
                    bvr_rngs.append(view[me]["bvr_rng"])
            side_bvr_rng = float(np.mean(bvr_rngs)) if bvr_rngs else 0.0

            geom_gate = soft_gate(dist_km, side_bvr_rng, bvr_geom_soft_km) if side_bvr_rng > 0 else 0.0
            bvr_share = bvr_opportunity * geom_gate
            wvr_share = 1.0 - bvr_share

            # requested shots by bucket (before ammo caps & salvo size)
            bvr_shots_need = shot_ops * bvr_share * max(0.0, salvo_size)
            wvr_shots_need = shot_ops * wvr_share * max(0.0, salvo_size)
            bvr_fired_by = {me: 0.0 for me in side_tags[side]}
            wvr_fired_by = {me: 0.0 for me in side_tags[side]}
            # (b) satisfy shots from ammo across cohorts
            # BVR
            bvr_fired = 0.0
            for me in side_tags[side]:
                pool = view[me]["pool_bvr"]
                have = sum(pool.values())
                if have <= 0 or bvr_shots_need <= 1e-12: 
                    continue
                fire = min(have, bvr_shots_need)
                bvr_fired += fire
                bvr_fired_by[me] += fire
                # consume proportionally by weapon share
                if have > 0:
                    for w in list(pool.keys()):
                        take = fire * (pool[w] / have)
                        pool[w] = max(0.0, pool[w] - take)
                    # refresh weighted pk if we fired something
                    view[me]["bvr_pk"] = weighted(pool, lambda w: st.session_state.weapons[w]["pk"]) if sum(pool.values())>0 else 0.0
                bvr_shots_need -= fire
                if bvr_shots_need <= 0: break

            # WVR
            wvr_fired = 0.0
            for me in side_tags[side]:
                pool = view[me]["pool_wvr"]
                have = sum(pool.values())
                if have <= 0 or wvr_shots_need <= 1e-12:
                    continue
                fire = min(have, wvr_shots_need)
                wvr_fired += fire
                wvr_fired_by[me] += fire
                if have > 0:
                    for w in list(pool.keys()):
                        take = fire * (pool[w] / have)
                        pool[w] = max(0.0, pool[w] - take)
                    view[me]["wvr_pk"] = weighted(pool, lambda w: st.session_state.weapons[w]["pk"]) if sum(pool.values())>0 else 0.0
                wvr_shots_need -= fire
                if wvr_shots_need <= 0: break

            # (c) expected kills = missiles * pk * (side-level Êk / shooters)  (simple coupling)
            # ek_sum already includes state * E_eff * k; normalize by onstation to avoid double-count
            kills_by_side[side] = 0.0
            for me in side_tags[side]:
                exp_hits_me = (
                    bvr_fired_by[me] * max(0.0, float(view[me]["bvr_pk"])) +
                    wvr_fired_by[me] * max(0.0, float(view[me]["wvr_pk"]))
                )
                ek_me = max(0.0, float(view[me]["E_eff"])) * max(0.0, float(view[me]["k"]))
                kills_by_side[side] += exp_hits_me * ek_me

            # (d) reloads for off-station share
            if reload_enabled and reload_rate_per_ac_sec > 0:
                offstation = 0.0
                for me in side_tags[side]:
                    offstation += state[me][i] * (1.0 - view[me]["avail"])
                reloads = offstation * reload_rate_per_ac_sec * dt
                # push reloads into each cohort, proportionally to empty capacity (simple model)
                # here we just add to the largest bucket deficit first
                if reloads > 1e-12:
                    for me in side_tags[side]:
                        # split half/half BVR/WVR by current mix (could be smarter)
                        view[me]["pool_bvr"]["_RELOAD"] = view[me]["pool_bvr"].get("_RELOAD", 0.0) + 0.5*reloads/len(side_tags[side])
                        view[me]["pool_wvr"]["_RELOAD"] = view[me]["pool_wvr"].get("_RELOAD", 0.0) + 0.5*reloads/len(side_tags[side])

        # 2) distribute losses to the opposing coalitions (proportional)
        for side in side_tags:
            for foe_side in side_tags:
                if foe_side == side:
                    continue
                K = max(0.0, float(kills_by_side[side]))
                if K <= 1e-12:
                    continue
                # Hard cap: can't kill more aircraft than exist on that foe this tick.
                target_cap = sum(state[tag][i] for tag in side_tags[foe_side])
                if target_cap <= 1e-12:
                    continue
                K = min(K, target_cap)
                # Weight targets by availability & vulnerability (once).
                weights = {tag: state[tag][i] * float(view[tag]["vuln_eff"]) for tag in side_tags[foe_side]}
                W = sum(weights.values())
                if W <= 1e-12:
                    continue
                for tag, w in weights.items():
                    loss = K * (w / W)
                    state[tag][i] = max(0.0, state[tag][i] - loss)

        # 3) early-out if quiet for 30s after contact
        if early_stop and sum(kills_by_side.values()) <= 1e-9:
            quiet_clock += dt
            if quiet_clock >= 30.0:
                for j in range(i+1, steps):
                    for (name,_,_) in cohorts:
                        state[name][j] = state[name][j-1]
                break
        else:
            quiet_clock = 0.0

    return T, state

if 'cohorts' not in st.session_state:
    st.session_state.cohorts = DEFAULT_COHORTS.copy()
with st.sidebar.expander("🛠 Scenario Editor", expanded=False):
    st.write("Add a new cohort:")
    new_name    = st.text_input("Cohort call-sign", key="new_cohort_name")
    new_nation   = st.selectbox("Nation",   list(st.session_state.nations.keys()), key="new_cohort_nation")
    new_airframe = st.selectbox("Airframe", list(st.session_state.airframes.keys()), key="new_cohort_airframe")
    if st.button("Add Cohort"):
        candidate = (new_name, new_nation, new_airframe)
        if not new_name:
            st.error("Give the cohort a unique name.")
        elif any(c[0] == new_name for c in st.session_state.cohorts):
            st.error("That call-sign is already used.")
        else:
            st.session_state.cohorts.append(candidate)
            # seed base allocation: all jets at the first listed base
            bases_for_nat = list(st.session_state.bases.get(new_nation, {}))
            if bases_for_nat:
                st.session_state.cohort_base_counts[new_name] = {
                    bases_for_nat[0]: COHORT_DEFAULTS.get(new_nation, {})
                        .get(new_airframe, AIRFRAME_DEFAULTS[new_airframe])["count"]
                }
            st.success(f"Added cohort {candidate}")
            st.rerun()
def _update_family(nation, selectbox_key):
    # 1) Update the nation’s family
    new_family = st.session_state[selectbox_key]
    st.session_state.nations[nation]["family"] = new_family

    # 2) For each cohort of that nation, clear out old sliders and reset params
    for name, coh_nation, _ in st.session_state.cohorts:
        if coh_nation != nation:
            continue

        key = name

        # Determine the new default coefficients for this nation+airframe
        default = (
            GENERIC.get(new_family, {})).copy()

        # Remove any stale slider state for this cohort’s ORDA params
        prefix = f"{key}-orda-param-"
        for sess_key in list(st.session_state.keys()):
            if sess_key.startswith(prefix):
                del st.session_state[sess_key]

        # Overwrite the cohort’s params with the new defaults
        st.session_state.cohort_params[key] = default

    # 3) Rerun so all sliders rebuild with the new defaults
    st.rerun()
# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("Force counts, lethality & families")

        # ————————————— Air-bases by nation —————————————
    with st.expander("Air-bases", expanded=False):
        for nat in sorted(st.session_state.bases):
            with st.expander(nat, expanded=False):
                for bname, (lat, lon) in list(st.session_state.bases[nat].items()):
                    st.text_input("Name", bname, key=f"bname-{nat}-{bname}", disabled=True)
                    e_lat_key = f"blat-{nat}-{bname}"
                    e_long_key = f"blon-{nat}-{bname}"
                    
                    st.session_state.setdefault(e_lat_key, lat)
                    st.session_state.setdefault(e_long_key, lon)
                    me = folium.Map(location=[st.session_state[e_lat_key], st.session_state[e_long_key ]], zoom_start=10)
                    folium.Marker(
                        [st.session_state[e_lat_key], st.session_state[e_long_key]],
                        popup=f"{bname}",
                        icon=folium.Icon(icon="map-pin", prefix="fa")
                    ).add_to(me)
                    folium.LatLngPopup().add_to(me)
                    oute = st_folium(me, height=250, width=400, key=f"mape-{nat}-{bname}")

                    # If the user clicks this run → write to session_state THEN rerun
                    if oute and oute.get("last_clicked"):
                        st.session_state[e_lat_key] = oute["last_clicked"]["lat"]
                        st.session_state[e_long_key ] = oute["last_clicked"]["lng"]
                    col2, col3 = st.columns((4, 4))
                    with col2:
                        st.number_input("Lat", value=lat, key=e_lat_key)
                    with col3:
                        st.number_input("Lon", value=lon, key=e_long_key )
                with st.expander("Add Air Base", expanded=False):
                    # add a new base for this nation
                    new_bn = st.text_input("New base name", key=f"newbase-{nat}")
                    # ⬇️ map-picker section
                    st.caption("Click map or type numbers ↓")
                    lat_key = f"newlat-{nat}"
                    lon_key = f"newlon-{nat}"

                    # ------------------------------------------------------------------
                    # 1️⃣  If user clicked on the map in the previous run, the coords are
                    #     already in st.session_state[lat_key / lon_key].  Otherwise set 0.
                    # ------------------------------------------------------------------
                    st.session_state.setdefault(lat_key, 0.0)
                    st.session_state.setdefault(lon_key, 0.0)

                    # ------------------------------------------------------------------
                    # 2️⃣  Show the map first and capture a click
                    # ------------------------------------------------------------------
                    m = folium.Map(location=[44.5, 34.0], zoom_start=5)
                    folium.LatLngPopup().add_to(m)
                    out = st_folium(m, height=250, width=400, key=f"map-{nat}")

                    # If the user clicks this run → write to session_state THEN rerun
                    if out and out.get("last_clicked"):
                        st.session_state[lat_key] = out["last_clicked"]["lat"]
                        st.session_state[lon_key] = out["last_clicked"]["lng"]

                    # ------------------------------------------------------------------
                    # 3️⃣  Now create the number_input widgets; they read from session_state
                    # ------------------------------------------------------------------
                    col_lat, col_lon = st.columns(2)
                    new_lat = col_lat.number_input("Lat°", -90.0, 90.0,
                                                key=lat_key)
                    new_lon = col_lon.number_input("Lon°", -180.0, 180.0,
                                                key=lon_key)
                        
                    if st.button("Add base", key=f"addbase-{nat}"):
                        
                        if new_bn and new_bn not in st.session_state.bases[nat]:
                            st.session_state.bases[nat][new_bn] = (new_lat, new_lon)
                            st.toast(f"Base “{new_bn}” added to {nat}")
                            st.rerun()
    with st.sidebar.expander("Deployment zones", expanded=False):
    # List & delete
        for zname, rings in list(st.session_state.deployment_zones.items()):
            st.markdown(f"**{zname}**  ({len(rings)} ring(s))")
            if st.button("Delete", key=f"del-zone-{zname}"):
                del st.session_state.deployment_zones[zname]
                st.rerun()

        # --- Effects (defaults seeded in session_state) ---
        st.markdown("### Zone effects")
        for z in st.session_state.deployment_zones.keys():
            eff = st.session_state.zone_effects.setdefault(z, {"E_mult": 1.0, "vuln_mult": 1.0})
            st.session_state.zone_effects[z]["E_mult"] = st.slider(
                f"{z} – E multiplier", 0.5, 1.5, eff["E_mult"], 0.05, key=f"{z}-Em"
            )
            st.session_state.zone_effects[z]["vuln_mult"] = st.slider(
                f"{z} – Vulnerability multiplier", 0.5, 1.5, eff["vuln_mult"], 0.05, key=f"{z}-Vm"
            )

        # --- Draw or edit a zone polygon ---
        st.markdown("### Add / edit zone")
        zname = st.text_input("Zone name", key="new_zone_name")
        m = folium.Map(location=[45.0, 34.0], zoom_start=5)
        Draw(
            export=False,
            draw_options={
                "polyline": False, "polygon": True, "rectangle": True,
                "circle": False, "marker": False, "circlemarker": False,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)
        out = st_folium(m, height=300, width=420, key="zones-map", returned_objects=[])

        features = []
        if out and out.get("all_drawings"):
            features = out["all_drawings"]["features"]
        elif out and out.get("last_active_drawing"):
            features = [out["last_active_drawing"]]

        rings = []
        for f in features:
            geom = f.get("geometry", {})
            if geom.get("type") in ("Polygon", "Rectangle"):
                for ring in geom["coordinates"]:
                    rings.append([(pt[0], pt[1]) for pt in ring])  # (lon,lat)

        if st.button("Save zone"):
            if not zname:
                st.warning("Name your zone.")
            elif not rings:
                st.warning("Draw at least one ring (polygon or rectangle).")
            else:
                st.session_state.deployment_zones[zname] = rings
                # seed neutral effects & empty CAP if brand new
                st.session_state.zone_effects.setdefault(zname, {"E_mult": 1.0, "vuln_mult": 1.0})
                st.session_state.zone_waypoints.setdefault(zname, [])
                st.success(f"Saved zone “{zname}” with {len(rings)} ring(s).")
                st.rerun()

        txt = st.text_area(
            "Manual rings editor (JSON list of rings: [[(lon,lat), ...], ...])",
            value="", placeholder="[[[33.0,44.8],[33.4,44.8],[33.4,44.4],[33.0,44.4]]]"
        )
        if st.button("Apply manual"):
            import json
            try:
                parsed = json.loads(txt)
                if zname:
                    st.session_state.deployment_zones[zname] = [
                        [(float(x), float(y)) for x, y in ring] for ring in parsed
                    ]
                    st.session_state.zone_effects.setdefault(zname, {"E_mult": 1.0, "vuln_mult": 1.0})
                    st.session_state.zone_waypoints.setdefault(zname, [])
                    st.success("Applied manual rings.")
                    st.rerun()
                else:
                    st.warning("Provide a zone name.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

            # --- Waypoints editor (ordered CAP route) ---
        st.markdown("### Patrol waypoints")
        zsel = st.selectbox(
            "Select zone",
            ["—"] + list(st.session_state.deployment_zones.keys()),
            key="zsel-zones"   # ← unique key
        )
        if zsel != "—":
            st.session_state.zone_waypoints.setdefault(zsel, [])
            m2 = folium.Map(location=[45.0, 34.0], zoom_start=6)
            folium.LatLngPopup().add_to(m2)

            # show existing waypoints
            for i, (lon, lat) in enumerate(st.session_state.zone_waypoints[zsel], 1):
                folium.Marker([lat, lon], popup=f"{i}").add_to(m2)

            out2 = st_folium(m2, height=280, width=420, key=f"wpts-{zsel}")
            if out2 and out2.get("last_clicked"):
                lat = out2["last_clicked"]["lat"]
                lon = out2["last_clicked"]["lng"]
                st.session_state.zone_waypoints[zsel].append((lon, lat))
                st.rerun()

            c1, c2 = st.columns(2)
            if c1.button("Clear last", key=f"wpt-pop-{zsel}"):
                if st.session_state.zone_waypoints[zsel]:
                    st.session_state.zone_waypoints[zsel].pop()
                    st.rerun()
            if c2.button("Clear all", key=f"wpt-clear-{zsel}"):
                st.session_state.zone_waypoints[zsel] = []
                st.rerun()
    # ————————————— Weapon catalogue (shared stats) —————————————
    with st.expander("Weapon characteristics", expanded=False):
        for w in sorted(st.session_state.weapons):
            with st.expander(w, expanded=False):
                spec = st.session_state.weapons[w]
                r_key = f"{w}-rng"; p_key = f"{w}-pk"
                spec["range_km"] = st.number_input(
                    f"{w} range (km)", 10, 300, spec["range_km"], 1, key=r_key
                )
                spec["pk"] = st.slider(
                    f"{w} Pₖ", 0.05, 0.95, spec["pk"], 0.01, key=p_key
                )

        # ————————————— Airframe default load-outs —————————————
    with st.expander("Airframe characteristics", expanded=False):
        for af in sorted(st.session_state.airframe_loadouts):
            with st.expander(af, expanded=False):
                load = st.session_state.airframe_loadouts[af]
                # edit existing weapons
                for w in list(load):
                    max_key = f"{af}-{w}-max"
                    q_key   = f"{af}-{w}-qty"

                    # Max first → defines the slider cap for qty
                    new_max = st.number_input(
                        f"{w} MAX", 1, 12, load[w]["max"], 1, key=max_key
                    )
                    load[w]["max"] = new_max

                    new_q = st.number_input(
                       f"{w} default", 0, new_max,
                       min(load[w]["default"], new_max), 1, key=q_key
                    )
                    if new_q == 0:
                        del load[w]               # delete weapon from this air-frame
                        # propagate deletion to cohorts
                        for ckey, cweps in st.session_state.cohort_weapons.items():
                            _, _, c_af = ckey.partition("-")
                            if c_af == af:
                                cweps.pop(w, None)
                    else:
                        load[w]["default"] = new_q
                # add another weapon
                with st.expander("Add Weapon", expanded=False):
                    add_w = st.selectbox(
                        "Add weapon", ["— select —"] + sorted(st.session_state.weapons),
                        key=f"{af}-addw"
                    )
                    add_max = st.number_input("MAX / a/c", 1, 12, 2, 1, key=f"{af}-addmax")
                    add_q   = st.number_input("Qty", 1, add_max, 2, 1, key=f"{af}-addq")
                    if st.button("Add", key=f"{af}-addbtn"):
                        if add_w == "— select —":
                            st.warning("Pick a weapon first.")
                        else:
                            # 1) add to the current air-frame
                            load[add_w] = {"default": min(add_q, add_max), "max": add_max}

                            # 2) propagate to _every_ other air-frame that ALREADY carries this weapon
                            for other_af, other_load in st.session_state.airframe_loadouts.items():
                                if other_af == af:
                                    continue
                                if add_w in other_load:
                                    other_load[add_w] = add_q
                                                    # 3) ensure all current cohorts flying those air-frames get the weapon
                            for ckey, cweps in st.session_state.cohort_weapons.items():
                                # cohort key format: "Nation-Airframe"
                                _, _, c_af = ckey.partition("-")
                                if add_w in st.session_state.airframe_loadouts.get(c_af, {}):
                                    # add only if not already present (honour manual changes)
                                    cweps.setdefault(add_w, min(add_q, add_max))
                            st.toast(f"{add_w} set to qty {add_q} for all applicable air-frames")
                            st.rerun()
        # ————————————— Aggregation families (per nation) —————————————
    with st.expander("Aggregation family by nation", expanded=False):
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
    with st.expander("Cohorts", expanded=False):
        for name, nation, airframe in st.session_state.cohorts:
            
            presets = (
                COHORT_DEFAULTS
                .get(nation, {})
                .get(airframe, AIRFRAME_DEFAULTS[airframe])
            )

            key = name
                # clone current default load-out on first appearance
            if key not in st.session_state.cohort_weapons:
                st.session_state.cohort_weapons[key] = {
                    w: spec["default"]
                    for w, spec in st.session_state.airframe_loadouts.get(airframe, {}).items()
                }
            with st.expander("Deployment", expanded=False):
                zopt = ["— none —"] + list(st.session_state.deployment_zones.keys())
                curr = st.session_state.cohort_zone_tasks.get(key, {"zone": "— none —", "fraction": 0.0})
                zpick = st.selectbox("Zone", zopt, index=zopt.index(curr["zone"]) if curr["zone"] in zopt else 0, key=f"{key}-zone")
                zfrac = st.slider("% assigned on-station", 0, 100, int(curr["fraction"]*100), 5, key=f"{key}-zfrac")
                st.session_state.cohort_zone_tasks[key] = {"zone": zpick, "fraction": zfrac/100.0}

            with st.expander(f"{key} ({nation}: {airframe})", expanded=False):
                if st.button("Delete", key=f"delete-{name}"):
                # 1) remove the cohort tuple itself
                    for ix, (iname, ination, iairframe) in enumerate(st.session_state.cohorts):
                        if(iname == name):
                            st.session_state.cohorts.pop(ix)

                    # 2) remove any widgets/session keys tied to this cohort
                    prefix = f"{name}"
                    for k in list(st.session_state.keys()):
                        if k.startswith(prefix):
                            del st.session_state[k]
                    st.session_state.cohort_base_counts.pop(key, None)
                    st.rerun()
                counts[key] = st.slider(
                    f"{name} count", 0, 200,
                    presets["count"], 1, key=f"{key}-count"
                )
                with st.expander("Model Variables", expanded=False):
                    kvals[key] = st.slider(
                        f"k ({name})", 0.0, 0.2,
                        presets["k"], 0.01, key=f"{key}-k"
                    )
                    shares[key] = share_sliders(
                        f"{name}", presets["orda"], pfx=""
                    )
                    vulns[key] = st.slider(
                        f"Vulnerability ({name})", 0.5, 1.5,
                        presets["vuln"], 0.05, key=f"{key}-vuln"
                    )
                    famsel[key] = st.session_state.nations[nation]["family"]
            # seed default coefficients for this cohort (nation+airframe)
                    default = DEFAULT_COEFFS.get(nation, {}).get(airframe, GENERIC.get(famsel[key], {}))
                    st.session_state.cohort_params.setdefault(key, default.copy())
                    # dynamic per-cohort ORDA coefficient overrides based on selected family
                    fam = famsel[key]
                    base_params = GENERIC.get(fam, {})
                    # ensure storage exists
                    st.session_state.cohort_params.setdefault(key, {})
                    for param, val in list(st.session_state.cohort_params[key].items()):
                        max_val = max(1.0, val * 2)
                        slider_key = f"{key}-orda-param-{param}"
                        st.session_state.cohort_params[key][param] = st.slider(
                            f"{param.capitalize()}",
                            0.0,
                            max_val,
                            val,
                            max_val/100,
                            key=slider_key,
                        )
                            # ————————————— Base selector —————————————
                            # ————————————— Base allocation —————————————
                nat_bases = list(st.session_state.bases.get(nation, {}))
                if key not in st.session_state.cohort_base_counts:
                    # first render → put everything at first base
                    st.session_state.cohort_base_counts[key] = {
                        nat_bases[0] if nat_bases else "—": counts[key]
                    }

                with st.expander("Base Allocation", expanded=False):
                    total_assigned = 0
                    for b in nat_bases:
                        alloc_key = f"{key}-base-{b}"
                        current   = st.session_state.cohort_base_counts[key].get(b, 0)
                        new_qty = st.slider(
                            f"{b}", 0, counts[key], current, 1, key=alloc_key
                        )
                        st.session_state.cohort_base_counts[key][b] = new_qty
                        total_assigned += new_qty

                    # warn if allocations don’t match cohort size
                    if total_assigned != counts[key]:
                        st.warning(f"{total_assigned} / {counts[key]} aircraft assigned")
                    # ————————————— Cohort weapon quantities —————————————
                with st.expander("Weapon Systems", expanded=False):
                    for w in list(st.session_state.cohort_weapons[key]):
                        spec = st.session_state.weapons.get(w, {"range_km":"?", "pk":0})
                        q_key = f"{key}-{w}-qty"
                        qty = st.slider(
                            f"{w} qty", 0, st.session_state.airframe_loadouts[airframe][w]["max"],
                            st.session_state.cohort_weapons[key][w], 1, key=q_key
                        )
                        if qty == 0:
                            del st.session_state.cohort_weapons[key][w]
                        else:
                            st.session_state.cohort_weapons[key][w] = qty
                        st.caption(f" Range {spec['range_km']} km | Pₖ {spec['pk']:.2f}")
        with st.expander("Quick-start deployment", expanded=False):
            if st.button("Auto-assign nearest zones and sensible on-station %"):
                if "cohort_zone_tasks" in st.session_state:
                    st.session_state.pop("cohort_zone_tasks")
                # re-seed
                if "_auto_seed_zone_tasks" in globals():
                    _auto_seed_zone_tasks()
                st.success("Zones assigned based on coalition and nearest base.")
                st.rerun()

                
    
    st.header("Global Model parameters")
    horizon = st.slider("Engagement (s)", 100000, 1000000, 100000, 50000)
    dt = 0.2

    st.subheader("Logistic remap (E → Ê)")
    lam_E = st.slider("λ (slope)", 1.0, 10.0, 1.5, 0.5)
    mid_E = st.slider("mid‑point", 0.1, 1.0, 1.0, 0.05)

    st.subheader("Kill advantage scale σ")
    sigma = st.slider("σ (steepness)", 1.0, 10.0, 3.5, 0.5)
    st.subheader("Weapons & engagement knobs")

    # Engagement tempo (how often shooters take a shot opportunity)
    eng_rate_per_ac_min = st.slider("Engagements per on-station aircraft per minute",
                                    0.0, 2.0, 0.10, 0.05)
    salvo_size = st.slider("Missiles per engagement (salvo size)",
                        0.0, 8.0, 2.0, 0.5)

    # BVR vs WVR split
    bvr_range_threshold_km = st.slider("BVR range threshold (km)",
                                    50, 120, 70, 5)
    bvr_opportunity = st.slider("BVR opportunity share (0=WVR only, 1=BVR only)",
                                0.0, 1.0, 0.60, 0.05)

    # Reloads (rearm while off-station)
    reload_enabled = st.checkbox("Enable reloads (rearm while off-station)", value=True)
    reload_rate_per_ac_min = st.slider("Reload rate (missiles per off-station aircraft per minute)",
                                    0.0, 4.0, 0.50, 0.10)
    # How sharply distance gates BVR (km of transition around the range)
    bvr_geom_soft_km = st.slider("BVR geometry softness (km)",
                             5, 80, 25, 5)
    closure_kmh = st.number_input("Aggressive closure (km/h)", 0, 1500, 300, 10, key="knob_closure")
    st.session_state["closure_kmh"] = closure_kmh

    tos_hours = st.sidebar.slider("Time on Station (hours)", 0.5, 12.0, 2.0, 0.5)
    st.checkbox("Early-stop after 30 s of no shots", value=True, key="early_stop_on_quiet")
    # convenience
    eng_rate_per_ac_sec = eng_rate_per_ac_min / 60.0
    reload_rate_per_ac_sec = reload_rate_per_ac_min / 60.0
    with st.sidebar.expander("Weapons & Engagement knobs", expanded=False):
        st.caption("These affect contact gating, shot cadence, and ammo consumption.")
        bvr_threshold_km = st.number_input("BVR threshold (km)", 50, 200, 80, 5, key="knob_bvr_thr")
        salvo_size       = st.number_input("Missiles per shot-op (salvo size)", 1, 6, 2, 1, key="knob_salvo")
        eng_rate_per_ac  = st.number_input("Shot opportunities per a/c per second", 0.0, 1.0, 0.04, 0.01, key="knob_eng_rate")
        bvr_geom_soft_km = st.number_input("Geometry softness (km)", 1, 100, 30, 1, key="knob_soft")
        bvr_opportunity  = st.slider("Max BVR share (0..1)", 0.0, 1.0, 0.8, 0.05, key="knob_bvr_share")

        st.divider()
        st.caption("Transit/availability & contact timing")
        cruise_kmh = st.number_input("Cruise speed (km/h)", 300, 1200, 800, 10, key="knob_cruise")
        tos_hours  = st.number_input("Time-on-station per sortie (hours)", 0.1, 3.0, 0.5, 0.1, key="knob_tos")
        contact_cap_sec = st.number_input("Cap contact time to ≤ this many seconds", 10, 3600, 240, 10, key="knob_tcap")

        st.divider()
        reload_enabled = st.checkbox("Enable reloading off-station", value=False, key="knob_reload_en")
        reload_rate_per_ac = st.number_input("Reload rate (missiles/a/c/sec off-station)", 0.0, 1.0, 0.02, 0.01, key="knob_reload_rate")



# ───────── Calculate effectiveness ─────────
# Anchor‑ratio scaling: compare every raw E to its family's baseline value
ANCHOR_SHARES = (0.25, 0.25, 0.25, 0.25)
E0 = {fam: raw_E(ANCHOR_SHARES, fam) for fam in FAMILIES}
TOS = tos_hours

E_raw = {
    t: raw_E(shares[t], famsel[t], st.session_state.cohort_params.get(t, {}))
    for t in shares
}
E_scaled = {t: E_raw[t] / (E0[famsel[t]] + EPS) for t in E_raw}
E = {t: logistic_remap(E_scaled[t], lam_E, mid_E) for t in E_scaled}

# ───────── Fast-forward helpers: centroids & first contact time ─────────
CRUISE_KMH = 800.0  # keep in one place
SEC_PER_HR = 3600.0

# 1) Cohort → assigned zone centroid (lat,lon) or None
cohort_zone_centroid = {}
for cname, nation, _af in st.session_state.cohorts:
    task = st.session_state.cohort_zone_tasks.get(cname, {"zone": "— none —"})
    z = task.get("zone")
    if z and z != "— none —":
        rings = st.session_state.deployment_zones.get(z, [])
        zlon, zlat = zone_centroid(rings)               # (lon, lat)
        cohort_zone_centroid[cname] = (zlat, zlon)      # store (lat, lon)
    else:
        cohort_zone_centroid[cname] = None

# 2) Cohort → first on-station time (seconds) if it has a zone; else 0
cohort_first_onstation_sec = {}
for cname, nation, _af in st.session_state.cohorts:
    cent = cohort_zone_centroid[cname]
    if cent is None:
        cohort_first_onstation_sec[cname] = 0.0
        continue
    bases_for_nat = list(st.session_state.bases.get(nation, {}).items())
    if not bases_for_nat:
        cohort_first_onstation_sec[cname] = 0.0
        continue
    # distance from FIRST base (matches your availability logic)
    _, (blat, blon) = bases_for_nat[0]                  # (lat, lon)
    zlat, zlon = cent
    dist_km = haversine_km(blat, blon, zlat, zlon)
    t_transit_hr = dist_km / max(1e-6, CRUISE_KMH)
    cohort_first_onstation_sec[cname] = t_transit_hr * SEC_PER_HR

# 3) Coalition → earliest on-station; t_contact = max(earliest_allied, earliest_opfor)
side_tags = {}
for (name, nation, _airframe) in st.session_state.cohorts:
    coal = st.session_state.nations.get(nation, {}).get("coalition")
    if coal:
        side_tags.setdefault(coal, []).append(name)

view = build_cohort_view(
        cohorts = st.session_state.cohorts,
        counts = counts,
        kvals = kvals,
        vulns = vulns,
        E = E,
        nations = st.session_state.nations,
        bases = st.session_state.bases,
        cohort_zone_tasks = st.session_state.cohort_zone_tasks,
        deployment_zones = st.session_state.deployment_zones,
        zone_effects = st.session_state.zone_effects,
        cohort_weapons = st.session_state.cohort_weapons,
        weapons = st.session_state.weapons,
        bvr_threshold_km = bvr_threshold_km,
        cruise_kmh = CRUISE_KMH,
        tos_hours = TOS,
    )
    
# Run fast sim
# side_tags is still built from cohorts + nations (unchanged)
T, state = simulate_fast(
    cohorts = st.session_state.cohorts,
    view = view,
    side_tags = side_tags,
    horizon = horizon,
    dt = dt,
    eng_rate_per_ac_sec = eng_rate_per_ac_sec,
    salvo_size = salvo_size,
    bvr_opportunity = bvr_opportunity,
    bvr_geom_soft_km = bvr_geom_soft_km,
    reload_enabled = reload_enabled,
    reload_rate_per_ac_sec = reload_rate_per_ac_sec,
)
st.session_state.series_counts = {
    tag: state[tag].astype(float).tolist() for tag in state
}
st.session_state.series_alive_frac = {
    tag: (state[tag] / max(1.0, float(view[tag]["N"]))).astype(float).tolist()
    for tag in state
}
st.session_state.final_counts = {
    tag: int(round(float(state[tag][-1]))) for tag in state
}
def compute_contact_time(view: dict) -> float:
    """
    Earliest engagement time (sec): later of the two coalitions'
    earliest on-station cohorts. Uses view[name]["first_on"].
    """
    # Find the fastest-to-station cohort per side
    min_ready_by_side = {}
    for name, info in view.items():
        if info.get("zone_centroid") is None:
            continue  # not assigned to a zone → ignore
        side = info["coalition"]
        t_on = float(info.get("first_on", 0.0))
        if side not in min_ready_by_side or t_on < min_ready_by_side[side]:
            min_ready_by_side[side] = t_on

    # Need at least two opposing sides to define contact time
    if len(min_ready_by_side) < 2:
        return 0.0

    # Engagement can start once both sides have at least one cohort on-station
    return max(min_ready_by_side.values())

# Compute contact time based on zone assignments & speeds
t_contact_sec = compute_contact_time(view)  # your helper

# Clip it to sim horizon
t_contact_sec = min(t_contact_sec, float(st.session_state.get("horizon", 30)))

# Bound to horizon
t_contact_sec = min(t_contact_sec, float(st.session_state.get("horizon", 30)) )

# --- Weapons pools (total missiles) & pk mix per cohort ---
cohort_ammo_total = {}   # {cohort: float missiles remaining}
cohort_pk_weighted = {}  # {cohort: weighted average pk of remaining stock}

def _recompute_pk_weighted(pool: dict) -> float:
    # pool is {weapon: remaining_count}
    total = sum(pool.values())
    if total <= 0:
        return 0.0
    s = 0.0
    for w, n in pool.items():
        pk = st.session_state.weapons.get(w, {}).get("pk", 0.0)
        s += n * pk
    return s / total

# build per-weapon pools so pk can evolve as one type depletes
cohort_weapon_pool = {}  # {cohort: {weapon: remaining_count}}
for name, nation, airframe in st.session_state.cohorts:
    key = name
    per_ac = st.session_state.cohort_weapons.get(key, {})
    pool = {w: counts[key] * qty for w, qty in per_ac.items()}  # total missiles
    cohort_weapon_pool[key] = pool
    cohort_ammo_total[key]  = sum(pool.values())
    cohort_pk_weighted[key] = _recompute_pk_weighted(pool)

# --- Helpers for geometry-aware BVR gating ---
def _weighted_pk(pool_dict: dict) -> float:
    total = sum(pool_dict.values())
    if total <= EPS: return 0.0
    s = 0.0
    for w, n in pool_dict.items():
        pk = st.session_state.weapons.get(w, {}).get("pk", 0.0)
        s += n * pk
    return s / total

def _weighted_range(pool_dict: dict) -> float:
    total = sum(pool_dict.values())
    if total <= EPS: return 0.0
    s = 0.0
    for w, n in pool_dict.items():
        rng = st.session_state.weapons.get(w, {}).get("range_km", 0.0)
        s += n * rng
    return s / total  # missiles-weighted average range

def _sigmoid01(x: float) -> float:
    # smooth step 0..1; x is unitless
    import math
    return 1.0 / (1.0 + math.exp(-x))



# ───────── Simulation arrays ─────────
# ───────── Simulation arrays ─────────
# ───────── Simulation arrays (fast-forward to contact) ─────────
steps = int(horizon / dt) + 1
_time = np.linspace(0.0, float(horizon), steps)

# state[c][0] = initial counts
state = {name: np.empty(steps) for name, _, _ in st.session_state.cohorts}
for tag in state:
    state[tag][0] = counts[tag]

# coalition dictionary already built in Patch A (side_tags)

# index to start “active” sim
start_idx = int(t_contact_sec / dt)
start_idx = max(1, min(start_idx, steps - 1))

# Pre-fill flat up to start_idx (no attrition before first contact)
for tix in range(1, start_idx):
    for tag in state:
        state[tag][tix] = state[tag][tix - 1]

# Keep constants here once
TOS = 0.5  # hours on-station per sortie (same as before)

# ========= MAIN LOOP from first contact onward =========
for i in range(start_idx, steps):
    # carry forward last tick
    for tag in state:
        state[tag][i] = state[tag][i - 1]

    # --- availability per cohort (unchanged math, just computed when needed) ---
    cohort_avail = {}
    for cname, nation, _airframe in st.session_state.cohorts:
        task = st.session_state.cohort_zone_tasks.get(cname, {"zone": "— none —", "fraction": 0.0})
        frac = task["fraction"]
        if task["zone"] and task["zone"] != "— none —":
            rings = st.session_state.deployment_zones.get(task["zone"], [])
            zlon, zlat = zone_centroid(rings)
            bases_for_nat = list(st.session_state.bases.get(nation, {}).items())
            if bases_for_nat:
                _, (blat, blon) = bases_for_nat[0]
                dist_km = haversine_km(blat, blon, zlat, zlon)
                t_transit = dist_km / CRUISE_KMH
                cycle = max(1e-6, 2 * t_transit + TOS)
                on_station_ratio = TOS / cycle
                cohort_avail[cname] = (1 - frac) + frac * on_station_ratio
            else:
                cohort_avail[cname] = 1.0
        else:
            cohort_avail[cname] = 1.0

    # --- zone multipliers (unchanged) ---
    E_eff = {}
    vuln_eff = {}
    for cname, nation, _ in st.session_state.cohorts:
        task = st.session_state.cohort_zone_tasks.get(cname, {"zone": "— none —"})
        zm, vm = 1.0, 1.0
        if task["zone"] and task["zone"] != "— none —":
            eff = st.session_state.zone_effects.get(task["zone"], {})
            zm = eff.get("E_mult", 1.0)
            vm = eff.get("vuln_mult", 1.0)
        E_eff[cname] = E[cname] * zm
        vuln_eff[cname] = vulns[cname] * vm

    # --- ammo-aware shots & kills (use your latest ammo block here) ---
    # NOTE: keep your existing “kills_side = {...} … consumption … reloads …” code here
    kills_side = {side: 0.0 for side in side_tags.keys()}
        # Early-out: if both sides produce ~no kills for several seconds, stop
    if all(k <= 1e-9 for k in kills_side.values()):
        # if we're far into the sim and nothing is happening, break
        if _time[i] - _time[start_idx] > 30.0:  # 30s of quiet after contact
            # fill the rest flat and finish
            for j in range(i + 1, steps):
                for tag in state:
                    state[tag][j] = state[tag][j - 1]
            break

    # Optional pruning: drop empty cohorts from further math (micro-opt)
    # (Keep bookkeeping simple: skip building new lists, just continue on zeros)

    # [insert your ammo firing block]
    # (no changes needed beyond where it already reads state[k], cohort_avail, E_eff, kvals, etc.)

    # --- apply losses (light vectorization while keeping your logic) ---
    # Build quick arrays for foe totals to avoid inner Python loops
    for side in side_tags.keys():
        for foe in side_tags.keys():
            if foe == side:
                continue
            # scalar sums with small comprehensions (fast in CPython for ~tens of cohorts)
            foe_total = sum(state[tag][i] * vuln_eff[tag] for tag in side_tags[foe]) + EPS
            if foe_total <= EPS:
                continue
            attr_rate = sum(E_eff[enemy] * state[enemy][i] for enemy in side_tags[foe])
            denom = sum(state[e][i] for e in side_tags[foe]) + EPS
            pressure_share = attr_rate / denom
            effective_kills = kills_side[side] * pressure_share

            share_denom = foe_total
            for tag in side_tags[foe]:
                hits = effective_kills * (state[tag][i] * vuln_eff[tag]) / share_denom
                loss = hits * vuln_eff[tag]
                state[tag][i] = max(0.0, state[tag][i] - loss)


st.markdown("### ⏱️ Time-series cohort map")
col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
with col_ts1:
    ts_minutes = st.slider("Timeline (min)", 5, 180, 60, 5, key="ts_minutes")
with col_ts2:
    ts_dt = st.slider("Δt (sec)", 5, 120, 15, 5, key="ts_dt")
with col_ts3:
    ts_speed = st.number_input("Cruise (km/h)", 200, 1400, 900, 10, key="ts_cruise")
with col_ts4:
    ts_tos = st.number_input("Time-on-station (h)", 0.0, 6.0, 1.0, 0.25, key="ts_tos")

# Build and show the animated map
ts_fc = _build_ts_geojson(view, ts_minutes, ts_dt, ts_speed, ts_tos)
center = (45.0, 34.0)
for v in view.values():
    if v.get("zone_centroid"):
        center = v["zone_centroid"]; break
m_ts = folium.Map(location=[center[0], center[1]], zoom_start=6, control_scale=True)
TimestampedGeoJson(
    data=ts_fc,
    period=f"PT{max(1, ts_dt)}S",
    add_last_point=True,
    transition_time=100,      # ms between frames
    auto_play=False,
    loop=False,
).add_to(m_ts)
st_folium(m_ts, height=480, use_container_width=True, key="ts_map")


# ───────── Plot ─────────
fig, ax = plt.subplots()
from itertools import cycle
styles = ["-", "--", "-.", ":", "dashdot", "dotted", (0, (3,1,1,1))]
style_iter = cycle(styles)

for name, nation, airframe in st.session_state.cohorts:
    ax.plot(T, state[name], label=f"{name} ({nation} - {airframe})", linestyle=next(style_iter), linewidth=2)

ax.set_xlabel("Iterations")
ax.set_ylabel("Aircraft remaining")
ax.set_title("Attrition – Ammo-limited BVR/WVR with contact-aware start")
ax.legend(fontsize="small", title_fontsize="x-small", ncol=2, handlelength=1.2, loc="upper right", framealpha=0.5)
st.pyplot(fig)

# ───────── Metrics summary ─────────
# Survivors by coalition (last time step)
survivors = {coal: sum(state[t][-1] for t in tags) for coal, tags in side_tags.items()}
# ───────── Nation & Cohort breakouts (start/end) ─────────
rows = []
for tag, v in view.items():
    N0 = int(v.get("N", 0))
    N1 = _infer_end_count(tag, N0)
    dN = N1 - N0
    loss_pct = (0.0 if N0 <= 0 else 100.0 * (N0 - N1) / N0)
    rows.append({
        "Coalition": v.get("coalition", ""),
        "Nation": v.get("nation", ""),
        "Cohort": tag,
        "Start N": N0,
        "End N": N1,
        "ΔN": dN,
        "Loss %": round(loss_pct, 1),
    })

df_break = pd.DataFrame(rows)

# Nation-level breakout
df_nation = (
    df_break
    .groupby(["Coalition", "Nation"], as_index=False)[["Start N", "End N", "ΔN"]]
    .sum()
    .assign(**{
        "Loss %": lambda d: (100.0 * (d["Start N"] - d["End N"]) / d["Start N"]).where(d["Start N"] > 0, 0.0).round(1)
    })
    .sort_values(["Coalition", "Nation"])
)

# Cohort-level breakout
df_cohort = (
    df_break
    .sort_values(["Coalition", "Nation", "Cohort"])
    .reset_index(drop=True)
)

st.markdown("### Nation breakout (start → end)")
st.dataframe(df_nation, use_container_width=True)

st.markdown("### Cohort breakout (start → end)")
st.dataframe(df_cohort, use_container_width=True)

# Optional: CSV exports
st.download_button(
    "⬇️ Download nation breakout (CSV)",
    df_nation.to_csv(index=False).encode("utf-8"),
    file_name="nation_breakout.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download cohort breakout (CSV)",
    df_cohort.to_csv(index=False).encode("utf-8"),
    file_name="cohort_breakout.csv",
    mime="text/csv",
)

if survivors and any(v > 0 for v in survivors.values()):
    cols = st.columns(max(1, len(survivors)))
    for col, (coal, val) in zip(cols, survivors.items()):
        col.metric(f"{coal} survivors", f"{val:.1f}")
else:
    st.info("No coalitions with survivorship to display yet. Check zone assignments and contact distance.")
with st.expander("Debug – assignments", expanded=False):
    st.write("Zone tasks:", st.session_state.cohort_zone_tasks)
    st.write("First-on-station (s):", {n: view[n]["first_on"] for n,_,_ in st.session_state.cohorts})
    st.write("Zones by side:", {side: [view[c]["zone"] for c in tags] for side, tags in side_tags.items()})

