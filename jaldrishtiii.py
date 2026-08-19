import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="JalDrishti",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Synthetic city network (demonstration data only)
# =========================================================
ZONES = {
    "Zone A": {"node": "N1", "population": 18400, "critical": 1.25, "base": 74, "x": 1.0, "y": 3.2},
    "Zone B": {"node": "N2", "population": 11200, "critical": 1.00, "base": 49, "x": 4.0, "y": 4.3},
    "Zone C": {"node": "N3", "population": 15600, "critical": 1.35, "base": 68, "x": 7.0, "y": 3.0},
    "Zone D": {"node": "N4", "population": 8200,  "critical": 0.90, "base": 35, "x": 4.2, "y": 1.0},
}

NODES = {
    "N1": (1.0, 3.2), "N2": (4.0, 4.3), "N3": (7.0, 3.0),
    "N4": (4.2, 1.0), "N5": (2.4, 2.0), "N6": (5.8, 2.0)
}

BASE_EDGES = [
    ("N1", "N5", 62), ("N2", "N5", 78), ("N2", "N6", 72),
    ("N3", "N6", 55), ("N4", "N5", 48), ("N4", "N6", 52),
    ("N5", "N6", 40),
]

PUMPS = {
    "P1": {"node": "N5", "capacity": 90, "zone_bias": ["Zone A", "Zone B"]},
    "P2": {"node": "N6", "capacity": 85, "zone_bias": ["Zone C", "Zone D"]},
}

ZONE_PUMP = {"Zone A": "P1", "Zone B": "P1", "Zone C": "P2", "Zone D": "P2"}

PUMP_DEGRADED_FACTOR = 0.42  # fraction of throughput retained on backup/manual mode

# =========================================================
# Core model functions (transparent, rule-based — not ML)
# =========================================================
def effective_capacity(design_capacity, blockage, silt, pump_available=True):
    return design_capacity * (1 - blockage / 100) * (1 - silt / 100) * (1 if pump_available else PUMP_DEGRADED_FACTOR)


def risk_score(rain, blockage, silt, storage_loss, pump_available, zone):
    # Demonstration scoring model: intentionally transparent, not a trained ML model.
    rain_component = min(rain / 120, 1.0) * 52
    capacity_penalty = (blockage * 0.18) + (silt * 0.12) + (storage_loss * 0.10)
    pump_penalty = 10 if not pump_available else 0
    criticality = (ZONES[zone]["critical"] - 1) * 10
    score = rain_component + capacity_penalty + pump_penalty + criticality
    return max(0, min(99, score))


def cause_diagnosis(blockage, silt, storage_loss, pump_available, rain):
    factors = {
        "Blockage": blockage * 1.0,
        "Siltation": silt * 0.85,
        "Storage loss": storage_loss * 0.75,
        "Pump failure": 45 if not pump_available else 0,
        "Rainfall loading": rain * 0.30,
    }
    top = max(factors, key=factors.get)
    confidence = min(96, 55 + factors[top] * 0.45)
    return top, confidence, factors


def recommended_action(score):
    if score >= 70:
        return "Deploy pump / emergency response"
    elif score >= 50:
        return "Dispatch drainage team"
    return "Monitor"


def priority_table(rain, blockage, silt, storage_loss, pump_status):
    rows = []
    for zone, d in ZONES.items():
        pump_id = ZONE_PUMP[zone]
        pump_available = pump_status[pump_id]
        score = risk_score(rain, blockage, silt, storage_loss, pump_available, zone)
        cap = effective_capacity(d["base"], blockage, silt, pump_available)
        cause, conf, _ = cause_diagnosis(blockage, silt, storage_loss, pump_available, rain)
        impact = math.log1p(d["population"]) * 5
        critical_bonus = d["critical"] * 18
        scarcity_bonus = max(0, 45 - cap) * 0.55
        priority = score * 0.72 + impact + critical_bonus + scarcity_bonus
        rows.append({
            "Zone": zone,
            "Pump": pump_id,
            "Risk": round(score, 1),
            "Capacity": round(cap, 1),
            "Cause": cause,
            "Confidence": round(conf),
            "Priority": round(priority, 1),
            "Action": recommended_action(score),
            "Population": d["population"],
        })
    df = pd.DataFrame(rows).sort_values("Priority", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


# =========================================================
# Max-flow / network bottleneck engine
# =========================================================
def build_flow_graph(pump_status, blockage, silt):
    """SOURCE -> zone nodes -> drainage edges -> pump nodes -> SINK."""
    edges = []
    for zone, d in ZONES.items():
        edges.append(("SOURCE", d["node"], d["base"]))
    for u, v, c in BASE_EDGES:
        edges.append((u, v, c))
    for pid, p in PUMPS.items():
        cap = effective_capacity(p["capacity"], blockage, silt, pump_status[pid])
        edges.append((p["node"], "SINK", cap))
    return edges


def max_flow_with_mincut(edges, source="SOURCE", sink="SINK"):
    nodes = set()
    for u, v, c in edges:
        nodes.add(u); nodes.add(v)
    graph = {n: {} for n in nodes}
    for u, v, c in edges:
        graph[u][v] = graph[u].get(v, 0) + c
        graph[v][u] = graph[v].get(u, 0)  # residual back-edge starts at 0

    flow = 0
    while True:
        parent = {source: None}
        queue = [source]
        for u in queue:
            if u == sink:
                break
            for v, cap in graph[u].items():
                if cap > 1e-9 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break
        bottleneck = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, graph[u][v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= bottleneck
            graph[v][u] += bottleneck
            v = u
        flow += bottleneck

    # min-cut: nodes reachable from source in residual graph
    reachable = {source}
    queue = [source]
    for u in queue:
        for v, cap in graph[u].items():
            if cap > 1e-9 and v not in reachable:
                reachable.add(v)
                queue.append(v)

    cut_edges = []
    for u, v, c in edges:
        if u in reachable and v not in reachable:
            cut_edges.append((u, v, c))
    return flow, cut_edges


# =========================================================
# Visualization
# =========================================================
def draw_network(df, pump_status, cut_edges):
    cut_set = {(u, v) for u, v, c in cut_edges} | {(v, u) for u, v, c in cut_edges}
    fig, ax = plt.subplots(figsize=(9, 5.4))

    for u, v, c in BASE_EDGES:
        x1, y1 = NODES[u]; x2, y2 = NODES[v]
        is_bottleneck = (u, v) in cut_set
        ax.plot([x1, x2], [y1, y2],
                linewidth=max(1.4, c / 30),
                alpha=0.9 if is_bottleneck else 0.4,
                color="crimson" if is_bottleneck else "steelblue",
                linestyle="-" if not is_bottleneck else "--")
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, f"{c}", fontsize=8, ha="center")

    risk_lookup = dict(zip(df["Zone"], df["Risk"]))
    for zone, d in ZONES.items():
        x, y = d["x"], d["y"]
        r = risk_lookup[zone]
        marker = "🔴" if r >= 70 else ("🟡" if r >= 50 else "🟢")
        ax.scatter(x, y, s=850, alpha=0.15)
        ax.scatter(x, y, s=260, edgecolors="black", linewidths=1.2)
        ax.text(x, y, marker, fontsize=14, ha="center", va="center")
        ax.text(x, y - 0.43, f"{zone}\nRisk {r:.0f}", fontsize=9, ha="center", weight="bold")

    for pid, p in PUMPS.items():
        x, y = NODES[p["node"]]
        available = pump_status[pid]
        ax.scatter(x, y + 0.35, s=240, marker="s", edgecolors="black",
                   color="seagreen" if available else "lightgray",
                   alpha=1 if available else 0.6)
        ax.text(x, y + 0.35, "P", ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x, y + 0.67, f"{pid} {'ONLINE' if available else 'OFFLINE'}",
                ha="center", fontsize=8, weight="bold",
                color="black" if available else "crimson")

    ax.set_title("JalDrishti — Synthetic Drainage Network (dashed red = current bottleneck)", fontsize=12, weight="bold")
    ax.set_xlim(0, 8); ax.set_ylim(0, 5.2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


# =========================================================
# UI styling
# =========================================================
st.markdown("""
<style>
.main-title {font-size: 2.3rem; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 1.05rem; opacity: .75; margin-top: 0;}
.small {font-size: .85rem; opacity: .72;}
.up {color:#1b7a1b; font-weight:700;}
.down {color:#b3261e; font-weight:700;}
.flat {color:#555; font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌧️ JalDrishti</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-assisted urban stormwater decision intelligence — '
    'WHERE it will flood → WHY → WHAT to do → WHO gets resources first</div>',
    unsafe_allow_html=True,
)

# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.header("🎛️ Scenario Controls")
    rain = st.slider("Rainfall intensity (mm)", 20, 150, 95, 5)
    blockage = st.slider("Blockage factor (%)", 0, 80, 28, 2)
    silt = st.slider("Siltation factor (%)", 0, 60, 18, 2)
    storage_loss = st.slider("Storage loss (%)", 0, 60, 12, 2)

    st.divider()
    st.subheader("⚙️ Pump network event")
    if "pump_status" not in st.session_state:
        st.session_state.pump_status = {"P1": True, "P2": True}

    c1, c2 = st.columns(2)
    if c1.button("🔴 Fail P1", use_container_width=True):
        st.session_state.pump_status["P1"] = False
    if c2.button("🔴 Fail P2", use_container_width=True):
        st.session_state.pump_status["P2"] = False
    if st.button("🟢 Restore all pumps", use_container_width=True):
        st.session_state.pump_status = {"P1": True, "P2": True}

    st.divider()
    st.caption(
        "Prototype note: the risk score and cause diagnosis are a transparent, "
        "rule-based demonstration model — not a trained ML model. No fabricated "
        "accuracy figures are shown. Swap in trained inference once verified "
        "labelled data are available."
    )

pump_status = st.session_state.pump_status
any_pump_down = not all(pump_status.values())

df = priority_table(rain, blockage, silt, storage_loss, pump_status)
baseline_df = priority_table(rain, blockage, silt, storage_loss, {"P1": True, "P2": True})

edges_now = build_flow_graph(pump_status, blockage, silt)
edges_baseline = build_flow_graph({"P1": True, "P2": True}, blockage, silt)
flow_now, cut_now = max_flow_with_mincut(edges_now)
flow_baseline, _ = max_flow_with_mincut(edges_baseline)

# =========================================================
# KPI row
# =========================================================
top = df.iloc[0]
cols = st.columns(5)
cols[0].metric("Highest Risk", f"{top['Risk']:.0f}/99", top["Zone"])
cols[1].metric("Top Priority", f"#{int(top['Rank'])}", top["Zone"])
cols[2].metric("Network Max Flow", f"{flow_now:.0f}",
                f"{flow_now - flow_baseline:+.0f} vs normal ops")
cols[3].metric("Pump P1", "ONLINE" if pump_status["P1"] else "OFFLINE")
cols[4].metric("Pump P2", "ONLINE" if pump_status["P2"] else "OFFLINE")

if any_pump_down:
    down = [p for p, ok in pump_status.items() if not ok]
    st.error(f"⚠️ {', '.join(down)} OFFLINE — network capacity has changed. "
             f"JalDrishti has automatically recalculated the priority queue and bottleneck.")
else:
    st.success("🟢 All pumps online. Priority queue reflects normal operating conditions.")

st.caption("Human-in-the-loop: JalDrishti produces a recommendation for the control-room authority. It does not autonomously actuate pumps or infrastructure.")

st.divider()

# =========================================================
# Tabs: WHERE / WHY / WHAT / WHO FIRST / NETWORK / DEMO
# =========================================================
tab_where, tab_why, tab_what_who, tab_network, tab_demo = st.tabs(
    ["📍 WHERE", "🔎 WHY", "✅ WHAT / 🚦 WHO FIRST", "🌐 Network Resilience", "🎬 Judge Demo"]
)

with tab_where:
    st.subheader("📍 WHERE will flooding concentrate?")
    left, right = st.columns([1.55, 1])
    with left:
        st.pyplot(draw_network(df, pump_status, cut_now), use_container_width=True)
    with right:
        st.markdown("**Hotspot ranking (predicted risk)**")
        hotspot = df[["Rank", "Zone", "Risk", "Capacity", "Population"]].copy()
        hotspot["Risk"] = hotspot["Risk"].map(lambda x: f"{x:.0f}")
        hotspot["Capacity"] = hotspot["Capacity"].map(lambda x: f"{x:.1f}")
        st.dataframe(hotspot, hide_index=True, use_container_width=True)
        st.caption("🔴 Risk ≥ 70  🟡 Risk 50–69  🟢 Risk < 50. Dashed red edge on the map = current network bottleneck (see Network Resilience tab).")

with tab_why:
    st.subheader(f"🔎 WHY is {top['Zone']} the top concern?")
    cause, conf, factors = cause_diagnosis(blockage, silt, storage_loss, pump_status[ZONE_PUMP[top["Zone"]]], rain)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown(f"""
        - **Zone:** {top['Zone']}
        - **Risk score:** {top['Risk']:.0f}/99
        - **Effective capacity:** {top['Capacity']:.1f}
        - **Leading cause:** {cause}
        - **Diagnosis confidence:** {conf:.0f}% (illustrative, evidence-weighted — not a calibrated ML probability)
        """)
    with c2:
        cause_df = pd.DataFrame({"Factor": list(factors.keys()), "Evidence score": list(factors.values())})
        st.bar_chart(cause_df.set_index("Factor"))
    st.caption("Cause diagnosis is a transparent evidence-weighting rule, so the control room can see exactly why a factor was blamed — not a black-box prediction.")

with tab_what_who:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("✅ WHAT to do")
        for _, r in df.iterrows():
            st.markdown(f"**{r['Zone']}** — {r['Action']}  \n"
                        f"<span class='small'>Risk {r['Risk']:.0f} · Cause: {r['Cause']}</span>",
                        unsafe_allow_html=True)
    with c2:
        st.subheader("🚦 WHO gets resources first")
        display = df[["Rank", "Zone", "Priority", "Risk", "Action"]].copy()
        display["Priority"] = display["Priority"].map(lambda x: f"{x:.1f}")
        display["Risk"] = display["Risk"].map(lambda x: f"{x:.0f}")
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption("Priority = 0.72 × Risk + population-impact term + zone criticality weight + capacity-scarcity bonus. Fully explainable — every term is visible, nothing is hidden inside a model.")

    st.divider()
    st.subheader("⏱️ Before vs After — priority queue under the current pump state")
    comp = baseline_df[["Zone", "Rank", "Priority"]].rename(columns={"Rank": "Rank (before)", "Priority": "Priority (before)"})
    comp = comp.merge(
        df[["Zone", "Rank", "Priority"]].rename(columns={"Rank": "Rank (after)", "Priority": "Priority (after)"}),
        on="Zone"
    )

    def rank_change(row):
        delta = row["Rank (before)"] - row["Rank (after)"]
        if delta > 0:
            return f"<span class='up'>▲ up {delta}</span>"
        elif delta < 0:
            return f"<span class='down'>▼ down {abs(delta)}</span>"
        return "<span class='flat'>— no change</span>"

    comp["Movement"] = comp.apply(rank_change, axis=1)
    comp = comp.sort_values("Rank (after)")
    st.write(
        comp.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )
    if any_pump_down:
        st.info("This table is the core proof point: a pump failure changed network capacity, and JalDrishti automatically re-ranked which zone gets attention first — no manual recalculation needed.")
    else:
        st.caption("Currently identical because all pumps are online. Trigger a pump failure in the sidebar to see the queue reorder.")

with tab_network:
    st.subheader("🌐 Network bottleneck / max-flow demonstration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max flow — normal ops", f"{flow_baseline:.0f}")
    c2.metric("Max flow — current scenario", f"{flow_now:.0f}")
    c3.metric("Capacity lost", f"{flow_baseline - flow_now:.0f}",
              delta_color="inverse")
    st.markdown("**Current bottleneck (min-cut) edges** — the constraint limiting total city drainage throughput:")
    if cut_now:
        cut_df = pd.DataFrame(
            [{"From": u, "To": v, "Capacity": c} for u, v, c in cut_now if u not in ("SOURCE",) and v not in ("SINK",)]
        )
        if cut_df.empty:
            cut_df = pd.DataFrame([{"From": u, "To": v, "Capacity": c} for u, v, c in cut_now])
        st.dataframe(cut_df, hide_index=True, use_container_width=True)
    else:
        st.write("No cut edges found.")
    st.caption(
        "Modeled as a flow network: each zone feeds drainage volume into shared pipe segments, "
        "which converge on pump stations P1/P2 that discharge to the outfall. Max-flow / min-cut "
        "identifies the true system-wide constraint — which may not be the same as the highest-risk zone."
    )

with tab_demo:
    st.subheader("🎬 30-second Judge Demo")
    st.markdown("""
    **Suggested narration:**
    1. "We start with a heavy-rain scenario across four zones."
    2. "JalDrishti predicts **WHERE** flooding will concentrate, and **WHY** — blockage, siltation, storage loss, or pump failure."
    3. "It recommends **WHAT** to do per zone, and **WHO** should get pumps and drainage teams first, with a fully explainable priority score."
    4. "The authority has limited pumps, so now I simulate a real failure — click **Fail P1**."
    5. "The network recalculates instantly: risk, priority ranking, and the system bottleneck all update — see the Before/After table and the dashed red edge on the map."
    6. "That's the difference between a static alert system and a decision-intelligence system: it re-plans automatically when the ground truth changes, but the human authority still makes the call."
    """)
    if any_pump_down:
        st.success("🔥 DEMO STATE: failure injected → network recalculated → priority queue reordered → bottleneck shifted. Ready to show the panel.")
    else:
        st.warning("For the live demo, click **Fail P1** (or **Fail P2**) in the sidebar and walk through the WHO FIRST tab's Before/After table.")

st.divider()
st.caption("JalDrishti • SIH 2026 prototype • Synthetic demonstration data • No fabricated real-world results or ML accuracy claims")