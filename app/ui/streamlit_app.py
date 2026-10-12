from __future__ import annotations

import httpx
import streamlit as st

st.set_page_config(page_title="GraphRAG Hybrid Engine", page_icon="🕸️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

DEFAULT_API = "http://localhost:8000"
API_URL = st.sidebar.text_input("API base URL", value=DEFAULT_API)

CHIP_CSS = """
<style>
.hero {
    background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #7c3aed 100%);
    padding: 1.4rem 2rem; border-radius: 14px; margin-bottom: 1rem;
}
.hero h1 { color: #fff; font-size: 1.65rem; margin: 0; }
.hero p { color: #cbd5e1; margin: .35rem 0 0 0; font-size: .95rem; }
.chip { display:inline-block; padding:.18rem .7rem; border-radius:999px;
        font-size:.74rem; font-weight:600; margin:.15rem .3rem .15rem 0; color:#fff; }
.chip-chunk { background:#2563eb; } .chip-graph { background:#7c3aed; }
.chip-summary { background:#0891b2; } .chip-web { background:#d97706; }
.answer-card { border:1px solid #334155; border-left:5px solid #8b5cf6;
               border-radius:10px; padding:1rem 1.2rem; background:#16213b; color:#e2e8f0; }
.score-pill { color:#94a3b8; font-size:.72rem; }
div[data-testid="stMetric"] { background:#16213b; border:1px solid #334155;
                              border-radius:10px; padding:.6rem .8rem; }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color:#e2e8f0 !important; }
.stChatMessage { border-radius: 12px; }
</style>
"""
st.markdown(CHIP_CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='hero'><h1>🕸️ GraphRAG &amp; Self-Corrective Hybrid Retrieval</h1>"
    "<p>Neo4j knowledge graph · Qdrant dense+sparse · RRF fusion · cross-encoder rerank · "
    "CRAG loop with web fallback · Ragas-evaluated</p></div>",
    unsafe_allow_html=True,
)


def chip(source: str) -> str:
    labels = {"chunk": "KB chunk", "graph": "graph hop", "summary": "doc summary", "web": "web"}
    label = labels.get(source, source)
    return f"<span class='chip chip-{source}'>{label}</span>"


def call_query(question: str) -> dict:
    response = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=900)
    response.raise_for_status()
    return response.json()


def render_answer(payload: dict) -> None:
    st.markdown(f"<div class='answer-card'>{payload.get('answer', '')}</div>", unsafe_allow_html=True)

    if payload.get("used_web"):
        st.warning("⚠️ Knowledge-base coverage was insufficient — the CRAG agent used a DuckDuckGo fallback.")
    else:
        st.success("✅ Answered fully from your knowledge base (no web fallback needed).")

    citations = payload.get("citations", [])
    if citations:
        top_score = max(float(c.get("score", 0)) for c in citations)
        chips = "".join(chip(c.get("source", "")) for c in citations)
        confidence = "high" if top_score >= 0.78 else ("medium" if top_score >= 0.5 else "low")
        dot = {"high": "🟢", "medium": "🟡", "low": "🟠"}[confidence]
        st.markdown(
            chips
            + f"<span class='chip' style='background:#334155'>{dot} top relevance "
            f"{top_score:.2f} · {confidence}</span>",
            unsafe_allow_html=True,
        )
        with st.expander(f"📚 Sources ({len(citations)}) — click to inspect"):
            for index, citation in enumerate(citations, start=1):
                score = citation.get("score", 0)
                st.markdown(
                    f"<span class='score-pill'>[{index}]</span> **{citation.get('title', '')}** "
                    f"&nbsp;<span class='score-pill'>score={score}</span><br>"
                    f"<code style='font-size:.72rem'>{str(citation.get('id', ''))[:110]}</code>",
                    unsafe_allow_html=True,
                )

    trace = payload.get("trace", [])
    if trace:
        left, right = st.columns([1, 1])
        with left:
            st.caption(f"⏱️ total {payload.get('latency_ms', 0):,.0f} ms · "
                       f"CRAG iterations {payload.get('iterations', 0)} · "
                       f"stop={payload.get('stop_reason', '')}")
            render_latency_chart(trace)
        with right:
            with st.expander("🔬 Pipeline trace detail"):
                rows = [
                    {
                        "stage": step["stage"],
                        "ms": step.get("ms", 0),
                        "detail": str(step.get("detail", {}))[:160],
                    }
                    for step in trace
                ]
                st.dataframe(rows, width="stretch", hide_index=True)


def render_latency_chart(trace: list[dict]) -> None:
    try:
        import altair as alt
        import pandas as pd

        rows = [
            {"stage": step["stage"], "ms": step.get("ms", 0)}
            for step in trace
            if step.get("ms", 0) > 0
        ]
        if not rows:
            return
        frame = pd.DataFrame(rows)
        chart = (
            alt.Chart(frame)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("ms:Q", title="latency (ms)"),
                y=alt.Y("stage:N", sort="-x", title=None),
                color=alt.Color("ms:Q", scale=alt.Scale(scheme="purples"), legend=None),
                tooltip=["stage", "ms"],
            )
            .properties(height=230)
        )
        st.altair_chart(chart, width="stretch")
    except Exception:
        chart_data = {step["stage"]: step.get("ms", 0) for step in trace}
        st.bar_chart(chart_data, height=220)


def render_subgraph() -> None:
    with st.expander("🧠 Explore knowledge-graph neighborhood", expanded=False):
        entity = st.text_input("Entity name (e.g. Kestrel Metals, INC-1042)", value="", key="entity_input")
        depth = st.slider("Traversal hops", 1, 2, 1, key="depth_slider")
        if entity and st.button("Fetch subgraph", type="primary"):
            try:
                data = httpx.get(f"{API_URL}/subgraph/{entity}", params={"depth": depth}, timeout=30).json()
            except Exception as exc:
                st.error(f"Subgraph fetch failed: {exc}")
                return
            edges = data.get("edges", [])
            if not edges:
                st.info("No RELATED edges found for that entity yet.")
                return
            try:
                from pyvis.network import Network

                net = Network(height="480px", directed=True, bgcolor="#0f172a", font_color="#e2e8f0",
                              select_menu=False, filter_menu=False)
                net.force_atlas_2based(gravity=-60, spring_length=120)
                degrees: dict[str, int] = {}
                for edge in edges:
                    degrees[edge["source"]] = degrees.get(edge["source"], 0) + 1
                    degrees[edge["target"]] = degrees.get(edge["target"], 0) + 1
                center = data.get("center", entity)
                for node in sorted(degrees):
                    is_center = node.lower() == center.lower()
                    net.add_node(
                        node,
                        label=node,
                        size=34 if is_center else max(14, 12 + 3 * degrees[node]),
                        color="#f59e0b" if is_center else "#38bdf8",
                        shape="star" if is_center else "dot",
                    )
                for edge in edges:
                    net.add_edge(edge["source"], edge["target"], label=edge["relation"],
                                 color="#94a3b8", arrows="to")
                html_path = "/tmp/graphrag_subgraph.html"
                net.save_graph(html_path)
                from streamlit.components import v1

                v1.html(open(html_path).read(), height=500)
            except ImportError:
                st.dataframe(edges)


def documents_panel() -> None:
    with st.sidebar.expander("🗂️ My documents", expanded=False):
        try:
            payload = httpx.get(f"{API_URL}/documents", timeout=20).json()
            documents = payload.get("documents", [])
        except Exception:
            documents = []
        if not documents:
            st.caption("No documents indexed yet.")
            return
        for document in documents:
            col_main, col_del = st.columns([5, 1])
            col_main.markdown(
                f"**{document['title'][:30]}**  \n"
                f"<span style='font-size:.72rem;color:#94a3b8'>"
                f"{document['chunks']} chunks · {document['entities']} entities</span>",
                unsafe_allow_html=True,
            )
            col_del.button(
                "🗑",
                key=f"del_{document['id']}",
                on_click=delete_document,
                args=(document["id"],),
            )


def delete_document(doc_id: str) -> None:
    try:
        httpx.delete(f"{API_URL}/documents/{doc_id}", timeout=60)
        st.toast(f"Deleted {doc_id}")
    except Exception as exc:
        st.error(f"Delete failed: {exc}")


def ingest_panel() -> None:
    with st.sidebar.expander("📥 Ingest documents", expanded=False):
        st.caption("Option 1 — upload files from your computer")
        uploaded = st.file_uploader(
            "PDF / Markdown / TXT",
            type=["pdf", "md", "txt", "markdown"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("⬆️ Upload & index", type="primary", use_container_width=True):
            for uploaded_file in uploaded:
                try:
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        payload = httpx.post(
                            f"{API_URL}/ingest/upload",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                            timeout=3600,
                        ).json()
                    document = payload.get("document", {})
                    st.success(
                        f"✅ {uploaded_file.name}: {document.get('chunks', '?')} chunks · "
                        f"{document.get('entities', '?')} entities · "
                        f"{document.get('relations', '?')} relations"
                    )
                except Exception as exc:
                    st.error(f"{uploaded_file.name}: {exc}")

        st.divider()
        st.caption("Option 2 — index a folder already on this machine")
        folder = st.text_input("Folder/file path", value="data/sample_corpus")
        if st.button("Run ingestion (sync)", use_container_width=True):
            try:
                with st.spinner("Extracting entities and indexing..."):
                    payload = httpx.post(
                        f"{API_URL}/ingest", json={"path": folder, "sync": True}, timeout=7200
                    ).json()
                documents = payload.get("documents", [])
                st.success(f"Ingested {len(documents)} document(s)")
                for document in documents:
                    st.write(f"- **{document['title']}**: {document['chunks']} chunks · "
                             f"{document['entities']} entities · {document['relations']} relations")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")


with st.sidebar:
    st.header("⚙️ Cluster")
    try:
        stats = httpx.get(f"{API_URL}/stats", timeout=15).json()
        health = httpx.get(f"{API_URL}/health", timeout=15).json()
        graph_stats = stats.get("graph", {}) or {}
        col_a, col_b = st.columns(2)
        col_a.metric("Documents", graph_stats.get("documents", "?"))
        col_b.metric("Chunks", graph_stats.get("chunks", "?"))
        col_c, col_d = st.columns(2)
        col_c.metric("Entities", graph_stats.get("entities", "?"))
        col_d.metric("Relations", graph_stats.get("relations", "?"))
        dots = " ".join("🟢" if health.get(k) else "🔴" for k in ("neo4j", "qdrant", "redis", "llm"))
        st.caption(f"{dots} &nbsp; neo4j · qdrant · redis · llm")
    except Exception:
        st.error("API unreachable — start it first (see README runbook).")
    ingest_panel()
    documents_panel()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

SAMPLE_QUESTIONS = [
    "Which supplier's machining defect caused the AtlasArm actuator overheating incident, "
    "and which batch was affected?",
    "Who should arrange an RMA for a failed VoltCore 4000 battery, and what is the SLA?",
    "Does the firmware fix for the NimbusOne telemetry dropouts satisfy SEC-14?",
]
st.caption("Try one of these multi-hop questions:")
cols = st.columns(3)
for column, question in zip(cols, SAMPLE_QUESTIONS, strict=False):
    column.button(question.split(",")[0][:34] + "...", key=f"sample_{hash(question)}",
                  on_click=lambda q=question: st.session_state.update(pending=q))

if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.pending = prompt

pending = st.session_state.pop("pending", None)
if pending:
    st.session_state.messages.append({"role": "user", "content": pending})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(pending)

    with st.chat_message("assistant", avatar="🕸️"):
        with st.spinner("dense + sparse + graph retrieval → RRF → rerank → grade → generate"):
            try:
                payload = call_query(pending)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.stop()
        render_answer(payload)
        render_subgraph()
        st.session_state.messages.append({"role": "assistant", "content": payload.get("answer", "")})

for message in st.session_state.messages[:-1]:
    avatar = "🙋" if message["role"] == "user" else "🕸️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
