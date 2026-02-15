import os
import json
import glob
import pandas as pd
import streamlit as st

LOG_DIR = "logs"

st.set_page_config(layout="wide", page_title="DietCheat Trace Dashboard")

@st.cache_data(show_spinner=False)
def load_jsonl_files(log_dir: str):
    paths = sorted(glob.glob(os.path.join(log_dir, "trace_*.jsonl")))
    records = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records, paths

def flatten_record(rec):
    summary = rec.get("summary") or {}
    return {
        "request_id": rec.get("request_id"),
        "ts_start": rec.get("ts_start"),
        "total_ms": rec.get("total_ms"),
        "intent": rec.get("intent"),
        "final_mode": rec.get("final_mode"),
        "raw_query": rec.get("raw_query"),
        "rag_domain": summary.get("rag_domain"),
        "retry_count": summary.get("retry_count"),
        "top_score": summary.get("top_score"),
    }

records, paths = load_jsonl_files(LOG_DIR)

st.sidebar.header("Logs")
st.sidebar.write(f"Files: {len(paths)}")
st.sidebar.write(f"Records: {len(records)}")

if not records:
    st.error("No trace JSONL files found in ./logs")
    st.stop()

df = pd.DataFrame([flatten_record(r) for r in records])

# Filters
intent_vals = ["ALL"] + sorted([x for x in df["intent"].dropna().unique().tolist()])
mode_vals = ["ALL"] + sorted([x for x in df["final_mode"].dropna().unique().tolist()])
domain_vals = ["ALL"] + sorted([x for x in df["rag_domain"].dropna().unique().tolist()])

st.sidebar.header("Filters")
intent_f = st.sidebar.selectbox("Intent", intent_vals, index=0)
mode_f = st.sidebar.selectbox("Final mode", mode_vals, index=0)
domain_f = st.sidebar.selectbox("RAG domain", domain_vals, index=0)
q_search = st.sidebar.text_input("Search (query/answer)", value="").strip()

fdf = df.copy()

if intent_f != "ALL":
    fdf = fdf[fdf["intent"] == intent_f]
if mode_f != "ALL":
    fdf = fdf[fdf["final_mode"] == mode_f]
if domain_f != "ALL":
    fdf = fdf[fdf["rag_domain"] == domain_f]

# For answer text search we need to scan records (keep simple)
if q_search:
    q_lower = q_search.lower()
    matched_ids = []
    for r in records:
        rq = (r.get("raw_query") or "").lower()
        # find writer output inside events if present
        ans = ""
        for ev in (r.get("events") or []):
            if ev.get("node") == "writer":
                # writer doesn't store answer text, so fallback to meta if you add it later
                pass
        # we can only search query for now unless you add final_answer to meta
        if q_lower in rq:
            matched_ids.append(r.get("request_id"))
    fdf = fdf[fdf["request_id"].isin(matched_ids)]

st.title("DietCheat Trace Dashboard")
st.dataframe(
    fdf.sort_values(by="ts_start", ascending=False),
    use_container_width=True,
    height=420
)

selected = st.selectbox(
    "Select request_id to inspect",
    options=fdf.sort_values(by="ts_start", ascending=False)["request_id"].tolist()
)

rec = next((r for r in records if r.get("request_id") == selected), None)
if rec is None:
    st.error("Record not found.")
    st.stop()

st.subheader("Request overview")
left, right = st.columns(2)
with left:
    st.json({k: rec.get(k) for k in ["request_id", "ts_start", "ts_end", "total_ms", "intent", "final_mode", "raw_query", "summary"]})
with right:
    st.write("Events count:", len(rec.get("events") or []))
    # Optional: show node list
    st.write("Nodes:", [e.get("node") for e in (rec.get("events") or [])])

st.subheader("Trace events")
for i, ev in enumerate(rec.get("events") or []):
    node = ev.get("node")
    extra = ev.get("extra") or {}
    with st.expander(f"{i+1}. {node}  |  {ev.get('duration_ms')} ms", expanded=(i == 0)):
        st.json(ev)

        # Convenience views for heavy nodes
        if node == "rag_retrieve":
            st.markdown("### chunks_preview")
            st.json(extra.get("chunks_preview", []), expanded=False)

            st.markdown("### chunks_full (for RAGAS)")
            st.json(extra.get("chunks_full", []), expanded=False)

        if node == "math_tool":
            rd = extra.get("resolver_debug") or {}
            st.markdown("### Resolver debug")
            st.json(rd, expanded=True)
