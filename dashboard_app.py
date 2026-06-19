import base64
import html
import json
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from confidence_engine import compute_confidence_from_evidence
from data_models import EvidenceClaim
from evidence_engine import enrich_ranking, enrich_relevance, enrich_with_stance
from graph_engine import apply_temporal_weighting, build_evidence_graph, graph_confidence_adjustment
from pipeline import refine_queries
from query_engine import generate_search_queries
from reddit_source import build_evidence_stream
from state_store import TemporalEvidenceStore
from temporal_engine import compute_temporal_trend


APP_TITLE = "Crisis Evidence Intelligence Dashboard"
APP_SUBTITLE = "Search Reddit, measure evidence quality, and explain the claim-level decision in one place."

SAMPLE_PROMPTS = [
    "The Philippines Earthquake is causing huge casualties",
    "The wildfire near Riverside is spreading quickly and evacuations are underway",
    "Flood warnings have been issued for the northern communities",
    "The water supply has been contaminated in the city",
    "The major highway is closed after the accident",
    "Rail service has been suspended across the region",
    "A severe storm is causing damage and power outages",
    "There was a shooting downtown and multiple injuries were reported",
]


CRISIS_KEYWORDS = {
    "fire": {"family": "Wildfire / Fire", "signals": ["fire", "smoke", "containment", "evacuation", "burn"]},
    "wildfire": {"family": "Wildfire / Fire", "signals": ["fire", "smoke", "containment", "evacuation", "burn"]},
    "earthquake": {"family": "Earthquake", "signals": ["earthquake", "quake", "aftershock", "tsunami", "damage"]},
    "quake": {"family": "Earthquake", "signals": ["earthquake", "quake", "aftershock", "tsunami", "damage"]},
    "flood": {"family": "Flood / Water", "signals": ["flood", "river", "water", "overflow", "inundation"]},
    "storm": {"family": "Severe Storm", "signals": ["storm", "wind", "power outage", "damage", "warning"]},
    "hurricane": {"family": "Severe Storm", "signals": ["storm", "wind", "power outage", "damage", "warning"]},
    "typhoon": {"family": "Severe Storm", "signals": ["storm", "wind", "power outage", "damage", "warning"]},
    "cyclone": {"family": "Severe Storm", "signals": ["storm", "wind", "power outage", "damage", "warning"]},
    "accident": {"family": "Transport / Accident", "signals": ["accident", "closure", "traffic", "injury", "delay"]},
    "shooting": {"family": "Public Safety / Shooting", "signals": ["shooting", "police", "injury", "casualties", "lockdown"]},
    "water": {"family": "Water / Health", "signals": ["water", "contamination", "boil", "advisory", "illness"]},
    "outbreak": {"family": "Health / Outbreak", "signals": ["outbreak", "cases", "illness", "hospital", "advisory"]},
    "landslide": {"family": "Landslide", "signals": ["landslide", "road closed", "damage", "rescue", "casualties"]},
}


def ensure_session_defaults():
    defaults = {
        "history": [],
        "analysis": None,
        "input_value": SAMPLE_PROMPTS[0],
        "live_active": False,
        "live_interval": 15,
        "live_claim": SAMPLE_PROMPTS[0],
        "live_evidence": [],
        "live_seen": set(),
        "live_query_stats": [],
        "live_analysis": None,
        "live_refreshes": 0,
        "live_last_refresh": None,
        "live_history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_live_stream(claim: str):
    st.session_state.live_claim = claim
    st.session_state.live_evidence = []
    st.session_state.live_seen = set()
    st.session_state.live_query_stats = []
    st.session_state.live_analysis = None
    st.session_state.live_refreshes = 0
    st.session_state.live_last_refresh = None
    st.session_state.live_history = []


def poll_reddit_for_claim(claim: str):
    raw_queries = generate_search_queries(claim)
    queries = refine_queries(raw_queries)

    new_items = []
    query_stats = []

    for query in queries:
        try:
            batch = build_evidence_stream(query)
            query_stats.append({"query": query, "status": "ok", "items": len(batch)})

            for item in batch:
                key = item.text.strip().lower()
                if key not in st.session_state.live_seen:
                    st.session_state.live_seen.add(key)
                    new_items.append(item)

        except Exception as exc:
            query_stats.append({"query": query, "status": f"error: {exc}", "items": 0})

    st.session_state.live_query_stats = query_stats
    if new_items:
        st.session_state.live_evidence.extend(new_items)

    st.session_state.live_refreshes += 1
    st.session_state.live_last_refresh = datetime.now()

    return queries, new_items, query_stats


def score_evidence_collection(user_input: str, evidence: list[EvidenceClaim]):
    scored = list(evidence)

    if scored:
        scored = enrich_relevance(user_input, scored)
        scored = [item for item in scored if item.relevance_score >= 0.35]
        if scored:
            scored = enrich_with_stance(user_input, scored)
            scored = enrich_ranking(user_input, scored)
            scored.sort(key=lambda x: getattr(x, "ranking_score", 0.0), reverse=True)

    store = TemporalEvidenceStore()
    store.add_batch(scored)
    all_evidence = store.get_all()

    if scored:
        base_confidence, explanation = compute_confidence_from_evidence(user_input, scored)
    else:
        base_confidence, explanation = 0.5, "No evidence available → neutral confidence"

    graph = build_evidence_graph(all_evidence)
    temporal_signal = compute_temporal_trend(all_evidence)
    graph = apply_temporal_weighting(graph, temporal_signal)
    graph_signal = graph_confidence_adjustment(graph)
    final_confidence = max(0.0, min(1.0, base_confidence + (graph_signal * 0.2)))

    store.add_confidence_snapshot(final_confidence)
    crisis_layer = detect_crisis_layer(user_input)

    return {
        "user_input": user_input,
        "evidence": scored,
        "all_evidence": all_evidence,
        "graph": graph,
        "timeline": store.get_timeline(),
        "base_confidence": base_confidence,
        "graph_signal": graph_signal,
        "final_confidence": final_confidence,
        "explanation": explanation,
        "query_stats": [],
        "stance_counts": Counter(getattr(item, "stance", "NEUTRAL") for item in scored),
        "subreddit_counts": Counter(getattr(item, "subreddit", "unknown") for item in scored),
        "source_counts": Counter(getattr(item, "source_type", "unknown") for item in scored),
        "crisis_layer": crisis_layer,
    }


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(34, 197, 94, 0.10), transparent 24%),
                radial-gradient(circle at 90% 10%, rgba(14, 165, 233, 0.10), transparent 22%),
                linear-gradient(180deg, #071018 0%, #09131c 45%, #05080d 100%);
            color: #e5eef7;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1520px;
        }

        .hero {
            padding: 1.3rem 1.4rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(11, 18, 29, 0.92), rgba(8, 14, 22, 0.82));
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin-bottom: 0.15rem;
            font-size: 2.2rem;
            color: #f4f8fb;
            letter-spacing: -0.03em;
        }

        .hero p {
            margin-top: 0.35rem;
            color: rgba(229, 238, 247, 0.78);
            font-size: 1.02rem;
            line-height: 1.6;
        }

        .pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            margin: 0.15rem 0.25rem 0 0;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.09);
            color: #dbe7f1;
            font-size: 0.85rem;
        }

        .card {
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 20px;
            background: rgba(10, 16, 25, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 38px rgba(0, 0, 0, 0.22);
        }

        .card h3 {
            margin-top: 0;
            margin-bottom: 0.35rem;
            color: #f6fbff;
        }

        .muted {
            color: rgba(229, 238, 247, 0.72);
        }

        .verdict-badge {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.02em;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
        }

        .verdict-true {
            background: rgba(34, 197, 94, 0.16);
            color: #86efac;
            border: 1px solid rgba(34, 197, 94, 0.38);
        }

        .verdict-mixed {
            background: rgba(251, 191, 36, 0.16);
            color: #fde68a;
            border: 1px solid rgba(251, 191, 36, 0.38);
        }

        .verdict-false {
            background: rgba(248, 113, 113, 0.15);
            color: #fecaca;
            border: 1px solid rgba(248, 113, 113, 0.32);
        }

        .section-title {
            margin-top: 0.25rem;
            margin-bottom: 0.5rem;
            color: #f7fbff;
            font-size: 1.15rem;
        }

        .small-note {
            color: rgba(229, 238, 247, 0.65);
            font-size: 0.88rem;
            line-height: 1.4;
        }

        div[data-testid="stMetricValue"] {
            color: #f6fbff;
        }

        div[data-testid="stMetricLabel"] {
            color: rgba(229, 238, 247, 0.82);
        }

        div[data-testid="stTabs"] button {
            font-weight: 600;
        }

        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def verdict_from_score(score: float) -> tuple[str, str]:
    if score >= 0.75:
        return "Likely true", "verdict-true"
    if score >= 0.55:
        return "Mixed evidence", "verdict-mixed"
    if score >= 0.35:
        return "Likely false", "verdict-false"
    return "Very likely false", "verdict-false"


def detect_crisis_layer(user_input: str):
    text = user_input.lower()
    matches = []

    for keyword, meta in CRISIS_KEYWORDS.items():
        if keyword in text:
            matches.append(meta)

    if matches:
        family = matches[0]["family"]
        signals = sorted({signal for meta in matches for signal in meta["signals"]})
    else:
        family = "General Crisis / Mixed"
        signals = ["evidence", "reports", "update", "warning", "damage"]

    return {
        "family": family,
        "signals": signals,
        "is_specific": bool(matches),
    }


def build_contradiction_graph_html(graph: nx.Graph) -> Path | None:
    contradiction_graph = nx.Graph()

    for node_id, data in graph.nodes(data=True):
        contradiction_graph.add_node(node_id, **data)

    for u, v, data in graph.edges(data=True):
        if data.get("type") == "CONTRADICT":
            contradiction_graph.add_edge(u, v, **data)

    if contradiction_graph.number_of_edges() == 0:
        return None

    return make_graph_html(contradiction_graph)


def make_graph_html(graph: nx.Graph, filename: str = "graph_latest.html") -> Path:
    net = Network(
        height="760px",
        width="100%",
        bgcolor="#0b1118",
        font_color="white",
        directed=False,
    )

    for node_id, data in graph.nodes(data=True):
        stance = data.get("stance", "NEUTRAL")
        if stance == "SUPPORT":
            color = "#22c55e"
        elif stance == "CONTRADICT":
            color = "#ef4444"
        else:
            color = "#94a3b8"

        net.add_node(
            node_id,
            label=(data.get("text", "")[:28] or node_id),
            title=f"""
Text: {html.escape(data.get('text', ''))}
Stance: {stance}
Relevance: {data.get('relevance', 0.0):.2f}
Confidence: {data.get('confidence', 0.0):.2f}
Subreddit: {html.escape(data.get('subreddit', 'unknown'))}
""",
            color=color,
            size=8 + data.get("relevance", 0.0) * 20,
        )

    for u, v, data in graph.edges(data=True):
        edge_type = data.get("type", "SUPPORT")
        color = "#22c55e" if edge_type == "SUPPORT" else "#ef4444"
        net.add_edge(u, v, value=data.get("weight", 1.0), color=color)

    net.force_atlas_2based()

    temp_path = Path(tempfile.gettempdir()) / filename
    net.write_html(str(temp_path))
    return temp_path


def summarize_evidence_rows(evidence: list[EvidenceClaim]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(evidence, start=1):
        rows.append(
            {
                "rank": idx,
                "source_type": getattr(item, "source_type", "unknown"),
                "subreddit": getattr(item, "subreddit", "unknown"),
                "stance": getattr(item, "stance", "NEUTRAL"),
                "relevance": round(getattr(item, "relevance_score", 0.0), 3),
                "stance_confidence": round(getattr(item, "stance_confidence", 0.0), 3),
                "ranking_score": round(getattr(item, "ranking_score", 0.0), 3),
                "parent_id": getattr(item, "parent_id", None) or "",
                "text": getattr(item, "text", ""),
            }
        )

    return pd.DataFrame(rows)


def group_threads(evidence: list[EvidenceClaim]):
    posts = [item for item in evidence if getattr(item, "source_type", "") == "reddit_post"]
    comments = [item for item in evidence if getattr(item, "source_type", "") == "reddit_comment"]

    posts_by_id = {item.source_id: item for item in posts}
    comments_by_parent = defaultdict(list)

    for comment in comments:
        parent_id = getattr(comment, "parent_id", None)
        if parent_id:
            comments_by_parent[parent_id].append(comment)

    for parent_id, items in comments_by_parent.items():
        items.sort(key=lambda x: getattr(x, "ranking_score", 0.0), reverse=True)

    ordered_posts = sorted(posts, key=lambda x: getattr(x, "ranking_score", 0.0), reverse=True)
    return ordered_posts, posts_by_id, comments_by_parent


def build_analysis(user_input: str):
    start = time.perf_counter()

    raw_queries = generate_search_queries(user_input)
    queries = refine_queries(raw_queries)

    evidence: list[EvidenceClaim] = []
    query_stats = []

    for query in queries:
        try:
            batch = build_evidence_stream(query)
            query_stats.append({"query": query, "status": "ok", "items": len(batch)})
            if batch:
                evidence.extend(batch)
        except Exception as exc:
            query_stats.append({"query": query, "status": f"error: {exc}", "items": 0})

    deduped = {}
    for item in evidence:
        key = item.text.strip().lower()
        if key not in deduped:
            deduped[key] = item
    evidence = list(deduped.values())

    scored = score_evidence_collection(user_input, evidence)

    evidence = scored["evidence"]
    all_evidence = scored["all_evidence"]
    graph = scored["graph"]
    timeline = scored["timeline"]
    base_confidence = scored["base_confidence"]
    graph_signal = scored["graph_signal"]
    final_confidence = scored["final_confidence"]
    explanation = scored["explanation"]

    counts = scored["stance_counts"]
    subreddit_counts = scored["subreddit_counts"]
    source_counts = scored["source_counts"]
    crisis_layer = scored["crisis_layer"]
    analysis_seconds = time.perf_counter() - start

    verdict, verdict_class = verdict_from_score(final_confidence)

    return {
        "user_input": user_input,
        "queries": queries,
        "query_stats": query_stats,
        "evidence": evidence,
        "all_evidence": all_evidence,
        "graph": graph,
        "timeline": timeline,
        "stance_counts": counts,
        "subreddit_counts": subreddit_counts,
        "source_counts": source_counts,
        "crisis_layer": crisis_layer,
        "base_confidence": base_confidence,
        "graph_signal": graph_signal,
        "final_confidence": final_confidence,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "explanation": explanation,
        "analysis_seconds": analysis_seconds,
    }


def render_live_stream_panel():
    st.markdown("#### Real-time streaming (live Reddit ingestion loop)")
    st.caption("This panel polls Reddit on a timer, dedupes new items, and continuously re-scores the claim.")

    with st.container():
        live_claim = st.text_area(
            "Live claim or topic",
            value=st.session_state.live_claim,
            height=90,
            key="live_claim_input",
        )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        interval = col_a.slider("Refresh every (seconds)", min_value=5, max_value=60, value=int(st.session_state.live_interval), step=5)
        start_clicked = col_b.button("Start live stream", width="stretch")
        stop_clicked = col_c.button("Stop live stream", width="stretch")

        st.session_state.live_interval = int(interval)

        if start_clicked:
            reset_live_stream(live_claim.strip() or st.session_state.input_value)
            st.session_state.live_active = True
            st.rerun()

        if stop_clicked:
            st.session_state.live_active = False

        if live_claim.strip() and live_claim.strip() != st.session_state.live_claim:
            reset_live_stream(live_claim.strip())

    if st.session_state.live_active and st.session_state.live_claim.strip():

        @st.fragment(run_every=st.session_state.live_interval)
        def live_fragment():
            queries, new_items, query_stats = poll_reddit_for_claim(st.session_state.live_claim)
            analysis = score_evidence_collection(st.session_state.live_claim, st.session_state.live_evidence)
            st.session_state.live_analysis = analysis
            st.session_state.live_history.append(
                {
                    "timestamp": datetime.now(),
                    "confidence": analysis["final_confidence"],
                }
            )

            top = st.columns(4)
            top[0].metric("Live items", len(st.session_state.live_evidence))
            top[1].metric("New this poll", len(new_items))
            top[2].metric("Current trust", f"{analysis['final_confidence']:.3f}")
            top[3].metric("Polls", st.session_state.live_refreshes)

            st.progress(analysis["final_confidence"])

            verdict, verdict_class = verdict_from_score(analysis["final_confidence"])
            st.markdown(
                f"""
                <div class="card" style="margin-top: 0.8rem;">
                    <span class="verdict-badge {verdict_class}">{verdict}</span>
                    <h3 class="section-title">Current Live Confidence: {analysis['final_confidence']:.3f}</h3>
                    <div class="small-note">Last refresh: {st.session_state.live_last_refresh}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            left, right = st.columns([1.1, 0.9])
            with left:
                st.markdown("##### Latest poll status")
                status_df = pd.DataFrame(query_stats)
                if status_df.empty:
                    st.info("No queries have been polled yet.")
                else:
                    st.dataframe(status_df, width="stretch", height=180)

                st.markdown("##### Live explanation")
                st.markdown(analysis["explanation"].replace("\n", "<br>"), unsafe_allow_html=True)

            with right:
                st.markdown("##### Recent live evidence")
                live_df = summarize_evidence_rows(st.session_state.live_evidence[:12])
                if live_df.empty:
                    st.info("Waiting for the first live batch.")
                else:
                    st.dataframe(
                        live_df[["rank", "source_type", "subreddit", "stance", "relevance", "stance_confidence", "ranking_score", "text"]],
                        width="stretch",
                        height=360,
                    )

            st.markdown("##### Live evidence graph")
            if analysis["graph"].number_of_nodes() > 1:
                graph_path = make_graph_html(analysis["graph"], filename="live_belief_graph.html")
                st.iframe(graph_path, width="stretch", height=620)
            else:
                st.info("Not enough live structure for a graph yet.")

            st.markdown("##### Live contradiction network")
            contradiction_html = build_contradiction_graph_html(analysis["graph"])
            if contradiction_html:
                st.iframe(contradiction_html, width="stretch", height=520)
            else:
                st.info("No contradiction edges have formed yet in live mode.")

            st.markdown("##### Live confidence timeline")
            if st.session_state.live_history:
                st.plotly_chart(build_timeline_figure(st.session_state.live_history), width="stretch")
            else:
                st.info("The live timeline will appear after the first poll.")

            st.caption("Live mode keeps polling the same claims, dedupes identical Reddit items, and updates the confidence score as new evidence arrives.")

        live_fragment()
    else:
        st.markdown(
            """
            <div class="card" style="margin-top: 0.85rem;">
                <h3 class="section-title">Live mode is paused</h3>
                <div class="small-note">
                    Click <strong>Start live stream</strong> to begin a real-time Reddit ingestion loop.
                    The app will poll on the selected interval and keep appending new evidence items.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_timeline_figure(history):
    fig = go.Figure()

    if history:
        fig.add_trace(
            go.Scatter(
                x=[item["timestamp"] for item in history],
                y=[item["confidence"] for item in history],
                mode="lines+markers",
                line=dict(color="#22d3ee", width=3),
                marker=dict(size=8, color="#f59e0b"),
                fill="tozeroy",
                fillcolor="rgba(34, 211, 238, 0.08)",
            )
        )

    fig.update_layout(
        title="Confidence Over Time",
        title_font=dict(color="#f8fafc"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7f1"),
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="Timestamp",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def build_source_chart(source_counts):
    labels = list(source_counts.keys())
    values = list(source_counts.values())
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=["#22d3ee", "#a78bfa", "#34d399", "#f59e0b"],
            )
        ]
    )
    fig.update_layout(
        title="Source Mix",
        title_font=dict(color="#f8fafc"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7f1"),
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def build_subreddit_chart(subreddit_counts):
    items = sorted(subreddit_counts.items(), key=lambda item: item[1], reverse=True)[:12]
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color="#34d399",
            )
        ]
    )
    fig.update_layout(
        title="Subreddit Activity",
        title_font=dict(color="#f8fafc"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7f1"),
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_tickangle=-35,
    )
    return fig


def chip(text: str) -> str:
    return f'<span class="pill">{html.escape(text)}</span>'


inject_css()
st.set_page_config(page_title=APP_TITLE, page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

ensure_session_defaults()

with st.sidebar:
    st.markdown("### Crisis Control Panel")
    sample = st.selectbox("Sample crisis claim", SAMPLE_PROMPTS, index=0)
    if st.button("Load sample into prompt"):
        st.session_state.input_value = sample
        st.rerun()

    st.markdown(
        """
        <div class="card">
            <div class="section-title">How it works</div>
            <div class="small-note">
                The dashboard searches Reddit, filters and ranks evidence, measures stance and relevance,
                then explains the final score using the strongest supporting and contradicting claims.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Run a claim through the full crisis evidence pipeline.")

st.markdown(
    f"""
    <div class="hero">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
        <div>
            {chip('Reddit evidence')} {chip('Claim-level confidence')} {chip('Comments included')} {chip('Graph + timeline')}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("analysis_form", clear_on_submit=False):
    claim = st.text_area(
        "Enter a claim or topic",
        value=st.session_state.input_value,
        height=110,
        placeholder="Example: The Philippines Earthquake is causing huge casualties",
    )
    analyze = st.form_submit_button("Analyze claim", width="stretch")

if analyze and claim.strip():
    with st.spinner("Searching Reddit, ranking evidence, and computing the final confidence score..."):
        st.session_state.analysis = build_analysis(claim.strip())
        st.session_state.input_value = claim.strip()
        st.session_state.history.append(
            {
                "timestamp": datetime.now(),
                "claim": claim.strip(),
                "confidence": st.session_state.analysis["final_confidence"],
            }
        )

analysis = st.session_state.analysis

if not analysis:
    st.markdown(
        """
        <div class="card">
            <h3 class="section-title">Ready when you are</h3>
            <p class="muted">
                Use the form above to analyze any crisis claim. You’ll get a confidence score,
                a narrative explanation, a graph view, source mix, and the Reddit comments that help
                explain the decision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

final_confidence = analysis["final_confidence"]
verdict, verdict_class = verdict_from_score(final_confidence)

# Top-level metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trust Score", f"{final_confidence:.3f}")
c2.metric("Base Confidence", f"{analysis['base_confidence']:.3f}")
c3.metric("Graph Signal", f"{analysis['graph_signal']:.3f}")
c4.metric("Relevant Evidence", len(analysis["evidence"]))
c5.metric("Analysis Time", f"{analysis['analysis_seconds']:.1f}s")

st.progress(final_confidence)

st.markdown(
    f"""
    <div class="card">
        <span class="verdict-badge {verdict_class}">{verdict}</span>
        <h3 class="section-title">Claim Accuracy / Trust Score: {final_confidence:.3f}</h3>
        <div class="small-note">{html.escape(analysis['user_input'])}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

query_pills = " ".join(chip(query) for query in analysis["queries"])
st.markdown(
    f"""
    <div class="card" style="margin-top: 0.8rem;">
        <div class="section-title">Search Queries Used</div>
        <div>{query_pills}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

support_count = analysis["stance_counts"].get("SUPPORT", 0)
contradict_count = analysis["stance_counts"].get("CONTRADICT", 0)
neutral_count = analysis["stance_counts"].get("NEUTRAL", 0)

c6, c7, c8 = st.columns(3)
c6.metric("Support Items", support_count)
c7.metric("Contradict Items", contradict_count)
c8.metric("Neutral Items", neutral_count)

# tabs
summary_tab, evidence_tab, threads_tab, live_tab, graph_tab, signals_tab, timeline_tab, export_tab = st.tabs(
    ["Summary", "Evidence", "Threads", "Live Stream", "Graph", "Signals", "Timeline", "Export"]
)

with summary_tab:
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Decision Rationale")
        st.markdown(analysis["explanation"].replace("\n", "<br>"), unsafe_allow_html=True)

        crisis_layer = analysis.get("crisis_layer", {"family": "General Crisis / Mixed", "signals": [], "is_specific": False})

        st.markdown(
            f"""
            <div class="card" style="margin-top: 0.85rem;">
                <h3 class="section-title">Crisis Detection Layer</h3>
                <div class="muted">Detected family: <strong>{html.escape(crisis_layer['family'])}</strong></div>
                <div class="small-note" style="margin-top: 0.4rem;">
                    Key signals: {', '.join(html.escape(signal) for signal in crisis_layer['signals'])}
                </div>
                <div class="small-note" style="margin-top: 0.4rem;">
                    Crisis-specific match: {'Yes' if crisis_layer['is_specific'] else 'No'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card" style="margin-top: 0.85rem;">
                <h3 class="section-title">What this means</h3>
                <div class="muted">
                    The score combines evidence relevance, claim-specific contradiction checks,
                    source reliability, and graph pressure. For crisis claims, casualty-related posts
                    get stricter treatment than generic event mentions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### Signal Snapshot")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=final_confidence,
                number={"valueformat": ".3f"},
                gauge={
                    "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "white"},
                    "bar": {"color": "#22d3ee"},
                    "steps": [
                        {"range": [0, 0.35], "color": "#3b0a12"},
                        {"range": [0.35, 0.55], "color": "#3f2a06"},
                        {"range": [0.55, 0.75], "color": "#183b23"},
                        {"range": [0.75, 1.0], "color": "#08311c"},
                    ],
                    "threshold": {"line": {"color": "#f59e0b", "width": 4}, "thickness": 0.75, "value": final_confidence},
                },
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#dbe7f1"},
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            f"""
            <div class="card">
                <h3 class="section-title">Verdict</h3>
                <div class="muted">{verdict}</div>
                <div class="small-note" style="margin-top: 0.5rem;">
                    Graph pressure: {analysis['graph_signal']:.3f} | Base confidence: {analysis['base_confidence']:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with evidence_tab:
    st.markdown("#### Ranked Evidence Explorer")
    evidence_df = summarize_evidence_rows(analysis["evidence"])

    if evidence_df.empty:
        st.info("No evidence survived the relevance filter.")
    else:
        col_a, col_b, col_c = st.columns(3)
        stance_filter = col_a.multiselect("Filter stance", sorted(evidence_df["stance"].unique()), default=sorted(evidence_df["stance"].unique()))
        source_filter = col_b.multiselect("Filter source type", sorted(evidence_df["source_type"].unique()), default=sorted(evidence_df["source_type"].unique()))
        subreddit_filter = col_c.text_input("Filter subreddit", value="")

        filtered_df = evidence_df[
            evidence_df["stance"].isin(stance_filter)
            & evidence_df["source_type"].isin(source_filter)
        ].copy()
        if subreddit_filter.strip():
            filtered_df = filtered_df[filtered_df["subreddit"].str.contains(subreddit_filter.strip(), case=False, na=False)]

        st.dataframe(
            filtered_df[["rank", "source_type", "subreddit", "stance", "relevance", "stance_confidence", "ranking_score", "text"]],
            width="stretch",
            height=520,
        )

with threads_tab:
    st.markdown("#### Posts and their top comments")
    posts, posts_by_id, comments_by_parent = group_threads(analysis["evidence"])

    if not posts:
        st.info("No Reddit posts to display.")
    else:
        for post in posts[:10]:
            comments = comments_by_parent.get(post.source_id, [])
            header = f"{getattr(post, 'stance', 'NEUTRAL')} | {getattr(post, 'subreddit', 'unknown')} | score {getattr(post, 'ranking_score', 0.0):.3f}"
            with st.expander(f"{header} — {post.text[:120]}"):
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="small-note">Post relevance: {getattr(post, 'relevance_score', 0.0):.3f} | Stance confidence: {getattr(post, 'stance_confidence', 0.0):.3f}</div>
                        <p>{html.escape(post.text)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if not comments:
                    st.caption("No retained comments for this post.")
                else:
                    st.caption(f"Top {min(5, len(comments))} comments retained for this thread")
                    for comment in comments[:5]:
                        st.markdown(
                            f"""
                            <div class="card" style="margin-top: 0.5rem; background: rgba(255,255,255,0.03);">
                                <div class="small-note">{getattr(comment, 'stance', 'NEUTRAL')} | {getattr(comment, 'subreddit', 'unknown')} | score {getattr(comment, 'ranking_score', 0.0):.3f}</div>
                                <p style="margin-bottom: 0;">{html.escape(comment.text)}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

with live_tab:
    render_live_stream_panel()

with graph_tab:
    st.markdown("#### Reasoning Graph")
    graph_html = make_graph_html(analysis["graph"])
    contradiction_html = build_contradiction_graph_html(analysis["graph"])

    graph_col, contradiction_col = st.columns(2)
    with graph_col:
        st.markdown("##### Evolving belief graph")
        st.iframe(graph_html, width="stretch", height=680)
    with contradiction_col:
        st.markdown("##### Contradiction network")
        if contradiction_html:
            st.iframe(contradiction_html, width="stretch", height=680)
        else:
            st.info("No contradiction edges detected yet.")

    st.caption(f"Nodes: {analysis['graph'].number_of_nodes()} | Edges: {analysis['graph'].number_of_edges()}")

with signals_tab:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_source_chart(analysis["source_counts"]), width="stretch")
        st.plotly_chart(build_subreddit_chart(analysis["subreddit_counts"]), width="stretch")
    with right:
        stance_df = pd.DataFrame(
            {
                "stance": list(analysis["stance_counts"].keys()),
                "count": list(analysis["stance_counts"].values()),
            }
        )
        if not stance_df.empty:
            st.plotly_chart(
                go.Figure(
                    data=[
                        go.Pie(
                            labels=stance_df["stance"],
                            values=stance_df["count"],
                            hole=0.5,
                            marker=dict(colors=["#22c55e", "#ef4444", "#94a3b8"]),
                        )
                    ]
                ).update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#dbe7f1"),
                    title="Stance Mix",
                    title_font=dict(color="#f8fafc"),
                    height=360,
                    margin=dict(l=10, r=10, t=45, b=10),
                ),
                width="stretch",
            )

    st.markdown("##### Temporal memory")
    if analysis["timeline"]:
        timeline_df = pd.DataFrame(analysis["timeline"])
        st.dataframe(timeline_df, width="stretch", height=180)
    else:
        st.info("No temporal memory entries yet.")

with timeline_tab:
    if not st.session_state.history:
        st.info("Run at least one analysis to populate the timeline.")
    else:
        st.plotly_chart(build_timeline_figure(st.session_state.history), width="stretch")
        timeline_df = pd.DataFrame(st.session_state.history)
        st.dataframe(timeline_df, width="stretch")

with export_tab:
    summary = {
        "claim": analysis["user_input"],
        "final_confidence": analysis["final_confidence"],
        "base_confidence": analysis["base_confidence"],
        "graph_signal": analysis["graph_signal"],
        "verdict": analysis["verdict"],
        "queries": analysis["queries"],
        "stance_counts": dict(analysis["stance_counts"]),
        "subreddit_counts": dict(analysis["subreddit_counts"]),
        "source_counts": dict(analysis["source_counts"]),
        "analysis_seconds": analysis["analysis_seconds"],
        "evidence": [
            {
                "text": item.text,
                "source_type": item.source_type,
                "subreddit": item.subreddit,
                "stance": item.stance,
                "relevance_score": item.relevance_score,
                "stance_confidence": getattr(item, "stance_confidence", 0.0),
                "ranking_score": getattr(item, "ranking_score", 0.0),
                "parent_id": getattr(item, "parent_id", None),
            }
            for item in analysis["evidence"]
        ],
    }

    st.download_button(
        "Download JSON report",
        data=json.dumps(summary, indent=2, default=str),
        file_name="crisis_evidence_report.json",
        mime="application/json",
        width="stretch",
    )

    st.markdown(
        """
        <div class="card" style="margin-top: 0.8rem;">
            <h3 class="section-title">Operational Notes</h3>
            <div class="small-note">
                The dashboard keeps the CLI pipeline intact and runs the same evidence scoring logic.
                Use the sample prompts on the left to test different crisis families such as earthquakes,
                floods, wildfires, transport disruptions, water contamination, and public safety incidents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
