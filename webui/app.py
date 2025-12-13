import os
from pathlib import Path

import streamlit as st

import scripts.rag_ask as rag_ask
from scripts.rag_ask import answer_question, DEFAULT_ASSISTANT_NAME
from scripts.feedback_store import append_feedback


APP_TITLE = "dwFixIT WebUI"


def default_config_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "scripts" / "rag_config.json")


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Config")
    config_path = st.text_input("rag_config.json", value=default_config_path())
    index_dir_override = st.text_input("Index dir override (optional)", value="")
    col_a, col_b = st.columns(2)
    with col_a:
        force_vision = st.checkbox("Force vision", value=False)
    with col_b:
        disable_vision = st.checkbox("Disable vision", value=False)
    show_debug = st.checkbox("Show retrieval debug", value=False)

st.caption("Local RAG over your `docs/` PDFs via Ollama.")

col_a, col_b = st.columns([1, 2])
with col_a:
    model = st.text_input("Model", value="", placeholder="JT2020, JT2720, FX30, ...")
with col_b:
    request_type = st.selectbox("Request type", ["Parts lookup", "Service question"], index=0)

part_desc = ""
service_question = ""
if request_type == "Parts lookup":
    part_desc = st.text_input("Short part description", value="", placeholder="air filter, oil filter, joystick, ...")
    service_question = st.text_area(
        "Service question (optional)",
        value="",
        placeholder="Optional extra context (symptoms, what you’re trying to do, serial break notes, etc.)",
        height=90,
    )
else:
    service_question = st.text_area(
        "Service question",
        value="",
        placeholder="Describe the issue (symptoms, codes, what changed, what you already tried).",
        height=140,
    )

structured_question_lines = []
if model.strip():
    structured_question_lines.append(f"Model: {model.strip()}")
if request_type == "Parts lookup" and part_desc.strip():
    structured_question_lines.append(f"Short part description: {part_desc.strip()}")
if service_question.strip():
    structured_question_lines.append(f"Service question: {service_question.strip()}")
structured_question_lines.append("Question: Provide the best answer using the manuals; if multiple variants fit, list part numbers by variant.")
question = "\n".join(structured_question_lines).strip()

col1, col2 = st.columns([1, 4])
with col1:
    ask = st.button("Ask", type="primary", use_container_width=True)
with col2:
    st.write("")

if ask:
    if request_type == "Parts lookup" and not model.strip():
        st.error("Enter a model (at least what the customer told you).")
    elif request_type == "Parts lookup" and not part_desc.strip() and not service_question.strip():
        st.error("Enter a short part description (or a service question).")
    elif request_type == "Service question" and not service_question.strip():
        st.error("Enter a service question.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = answer_question(
                    question=question,
                    config_path=config_path or None,
                    index_dir_override=(index_dir_override or None),
                    force_vision=force_vision,
                    disable_vision=disable_vision,
                )
            except Exception as e:
                st.error(str(e))
            else:
                st.session_state["last_result"] = result
                st.session_state["last_query"] = {
                    "model": model.strip(),
                    "request_type": request_type,
                    "part": part_desc.strip(),
                    "service_question": service_question.strip(),
                    "question": question,
                }

result = st.session_state.get("last_result")
last_query = st.session_state.get("last_query") or {}
if result:
    st.markdown(result.get("answer", ""))
    variant_parts = result.get("variant_parts") or {}
    if variant_parts:
        with st.expander("Parts found (auto)", expanded=False):
            for variant, part_numbers in variant_parts.items():
                if not part_numbers:
                    continue
                st.write(f"**{variant}**")
                st.write(", ".join(f"`{pn}`" for pn in part_numbers))

    sources = result.get("sources") or []
    if sources:
        with st.expander("Sources", expanded=False):
            for i, p in enumerate(sources, start=1):
                st.write(f"[{i}] `{p}`")

    if show_debug:
        with st.expander("Retrieval debug", expanded=False):
            st.json(result.get("hits", []))

    with st.expander("Feedback (improves future answers)", expanded=False):
        rating = st.radio("Was the answer correct?", ["Correct", "Partially correct", "Incorrect"], horizontal=True)
        corrected_variant = ""
        corrected_part_numbers = ""
        if rating != "Correct":
            variants = list((variant_parts or {}).keys())
            corrected_variant = st.selectbox(
                "Which variant should this apply to? (optional)",
                options=[""] + variants + ["Other / type below"],
                index=0,
            )
            if corrected_variant == "Other / type below":
                corrected_variant = st.text_input("Variant label", value="", placeholder="JT2020 MACH 1 TIER 3, JT2020 MACH 1 (LEGACY), ...")
            corrected_part_numbers = st.text_input(
                "Correct part numbers (comma-separated)",
                value="",
                placeholder="194-484, 194-830, 194-577",
            )

        note = st.text_area("Notes (optional)", value="", height=80)
        if st.button("Save feedback"):
            record = {
                "model": last_query.get("model") or "",
                "part": last_query.get("part") or "",
                "request_type": last_query.get("request_type") or "",
                "service_question": last_query.get("service_question") or "",
                "question": last_query.get("question") or "",
                "answer": result.get("answer") or "",
                "sources": sources,
                "hits": result.get("hits") or [],
                "variant_parts": variant_parts,
                "rating": rating.lower().replace(" ", "_"),
            }
            if rating != "Correct" and corrected_part_numbers.strip():
                pns = [p.strip() for p in corrected_part_numbers.split(",") if p.strip()]
                corr_vp = {}
                if corrected_variant.strip():
                    corr_vp[corrected_variant.strip()] = pns
                else:
                    corr_vp["Unspecified variant"] = pns
                record["corrected"] = {"variant_parts": corr_vp}
            if note.strip():
                record["note"] = note.strip()

            out_path = append_feedback(record)
            rag_ask.load_feedback_index.cache_clear()
            st.success(f"Saved to `{out_path}`")

st.divider()
st.caption(f"Assistant: `{DEFAULT_ASSISTANT_NAME}` (set via `scripts/rag_config.json`).")
