import os
from pathlib import Path
import json

import streamlit as st

import scripts.rag_ask as rag_ask
from scripts.rag_ask import answer_question, DEFAULT_ASSISTANT_NAME
from scripts.feedback_store import append_feedback


APP_TITLE = "dwFixIT WebUI"


def default_config_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "scripts" / "rag_config.json")


def load_config_quiet(config_path: str) -> dict:
    try:
        p = Path(config_path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def guess_ollama_server_log_path() -> Path | None:
    # Windows default: %LOCALAPPDATA%\Ollama\server.log
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        p = Path(local_appdata) / "Ollama" / "server.log"
        if p.exists():
            return p

    # Fallbacks (best-effort for non-Windows).
    home = Path.home()
    for cand in (
        home / ".ollama" / "server.log",
        home / ".ollama" / "logs" / "server.log",
    ):
        if cand.exists():
            return cand
    return None


def tail_text_file(path: Path, *, max_lines: int = 200, max_bytes: int = 250_000) -> str:
    # Efficient-ish tail implementation to avoid reading multi-MB logs into memory/UI.
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, max_bytes)
            f.seek(-read_size, os.SEEK_END)
            data = f.read(read_size)
    except Exception as e:
        return f"Failed to read: {path}\n{e}"

    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Mode")
    ui_mode = st.selectbox("UI mode", ["Chat", "Form (advanced)"], index=0)

    st.subheader("Config")
    config_path = st.text_input("rag_config.json", value=default_config_path())
    cfg = load_config_quiet(config_path)
    index_dir_override = st.text_input("Index dir override (optional)", value="")
    col_a, col_b = st.columns(2)
    with col_a:
        force_vision = st.checkbox("Force vision", value=False)
    with col_b:
        disable_vision = st.checkbox("Disable vision", value=False)
    show_debug = st.checkbox("Show retrieval debug", value=False)

    st.divider()
    st.subheader("Ollama Logs")
    show_ollama_logs = st.checkbox("Show Ollama server log tail", value=False)
    default_log = (cfg.get("ollama_server_log_path") or "").strip()
    if not default_log:
        guessed = guess_ollama_server_log_path()
        default_log = str(guessed) if guessed else ""
    ollama_log_path = st.text_input("server.log path", value=default_log)
    tail_lines = int(cfg.get("ollama_server_log_tail_lines") or 200)
    tail_lines = st.number_input("Tail lines", min_value=50, max_value=5000, value=tail_lines, step=50)
    refresh_logs = st.button("Refresh logs", use_container_width=True)

st.caption("Local RAG over your `docs/` PDFs via Ollama.")

def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    if t.startswith(("how ", "why ", "what ", "where ", "when ", "does ", "do ", "can ", "is ", "are ")):
        return True
    return False


def _contains_structured_fields(text: str) -> bool:
    tl = (text or "").lower()
    return any(k in tl for k in ("model:", "short part description:", "service question:", "question:"))


def _is_filter_list_intent(text: str) -> bool:
    tl = (text or "").lower()
    return ("filter" in tl) and ("list" in tl or "all filters" in tl or "filters list" in tl)


if ui_mode == "Chat":
    st.subheader("Chat")
    st.caption('Ask naturally (e.g., "list all filters for JT20", "oil filter for JT10", "machine won’t track").')

    with st.sidebar:
        st.subheader("Chat Settings")
        default_model = st.text_input("Model (sticky)", value="", placeholder="JT20, JT10, FX30, ...")
        request_type = st.selectbox("Request type", ["Parts lookup", "Service question"], index=0)
        auto_structure = st.checkbox(
            "Auto-structure messages",
            value=True,
            help="Formats your message into Model/Part/Question fields so feedback matches better.",
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.pop("chat_messages", None)
            st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

    user_text = st.chat_input("Message dwFixIT…")
    if user_text:
        st.session_state["chat_messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Build a single-shot RAG query (no multi-turn memory yet), but format it so:
        # - feedback matching works (needs Model + Short part description when possible)
        # - parts vs service intent is clear
        if _contains_structured_fields(user_text) or not auto_structure:
            question = user_text.strip()
            if default_model.strip() and "model:" not in user_text.lower():
                question = f"Model: {default_model.strip()}\n{question}"
        else:
            q_lines: list[str] = []
            if default_model.strip():
                q_lines.append(f"Model: {default_model.strip()}")

            if request_type == "Parts lookup":
                if _is_filter_list_intent(user_text):
                    q_lines.append("Short part description: filters")
                    q_lines.append(f"Question: {user_text.strip()}")
                elif (not _looks_like_question(user_text)) and (len(user_text.strip()) <= 80):
                    q_lines.append(f"Short part description: {user_text.strip()}")
                    q_lines.append(
                        "Question: Provide the best answer using the manuals; if multiple variants fit, list part numbers by variant."
                    )
                else:
                    q_lines.append(f"Service question: {user_text.strip()}")
                    q_lines.append(
                        "Question: Provide the best answer using the manuals; if multiple variants fit, list part numbers by variant."
                    )
            else:
                q_lines.append(f"Service question: {user_text.strip()}")
                q_lines.append("Question: Answer using the manuals; ask only for the minimum missing info.")

            question = "\n".join(q_lines).strip()

        with st.chat_message("assistant"):
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
                    msg = str(e)
                    if "Read timed out" in msg and "11434" in msg:
                        msg += (
                            "\n\nTip: Ollama didn’t respond before the client timeout. "
                            "Try pre-warming the chat model (`ollama run qwen3:4b \"hi\"`), "
                            "or set `ollama_generate_timeout_s` in your `scripts/rag_config.json` (e.g. 900), "
                            "or switch `chat_model` to a smaller/faster model."
                        )
                    st.error(msg)
                else:
                    answer = result.get("answer", "")
                    st.markdown(answer)
                    st.session_state["chat_messages"].append({"role": "assistant", "content": answer})

                    sources = result.get("sources") or []
                    if sources:
                        with st.expander("Sources", expanded=False):
                            for i, p in enumerate(sources, start=1):
                                st.write(f"[{i}] `{p}`")

                    if show_debug:
                        with st.expander("Retrieval debug", expanded=False):
                            st.json(result.get("hits", []))
else:  # Form (advanced)
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
            placeholder="Optional extra context (symptoms, what you're trying to do, serial break notes, etc.)",
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

    if show_ollama_logs and ollama_log_path.strip():
        with st.expander("Ollama server.log (tail)", expanded=False):
            log_path = Path(ollama_log_path.strip())
            if not log_path.exists():
                st.warning(f"Log file not found: `{log_path}`")
            else:
                # Re-render when the refresh button is clicked.
                if refresh_logs:
                    pass
                st.code(tail_text_file(log_path, max_lines=int(tail_lines)), language="text")

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
                    msg = str(e)
                    if "Read timed out" in msg and "11434" in msg:
                        msg += (
                            "\n\nTip: Ollama didn’t respond before the client timeout. "
                            "Try pre-warming the chat model (`ollama run qwen3:4b \"hi\"`), "
                            "or set `ollama_generate_timeout_s` in your `scripts/rag_config.json` (e.g. 900), "
                            "or switch `chat_model` to a smaller/faster model."
                        )
                    st.error(msg)
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
