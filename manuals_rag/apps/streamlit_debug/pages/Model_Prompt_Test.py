from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st


API_BASE = os.getenv("MANUALS_RAG_API_BASE", "http://127.0.0.1:8600")
AUTH_TOKEN = os.getenv("MANUALS_RAG_AUTH_TOKEN", "admin-token")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def _post(path: str, payload: dict[str, Any], **params: Any) -> Any:
    with httpx.Client(base_url=API_BASE, timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        response = client.post(path, headers=_headers(), json=payload, params=params or None)
        response.raise_for_status()
        return response.json()


def _render_json_block(label: str, payload: Any) -> None:
    st.markdown(f"**{label}**")
    st.json(payload, expanded=False)


def _format_duration(duration_ms: Any) -> str:
    if duration_ms is None:
        return "n/a"
    return f"{float(duration_ms):,.2f} ms"


def main() -> None:
    st.set_page_config(page_title="Model Prompt Test", layout="wide")
    st.title("Model Prompt Test")
    st.caption("Directly test Ollama model behavior outside the retrieval pipeline.")

    with st.sidebar:
        st.markdown("### Connection")
        st.text_input("API Base", value=API_BASE, disabled=True)
        token = st.text_input("Auth Token", value=AUTH_TOKEN, type="password", disabled=True)
        st.caption(f"Using token `{token[:4]}...`.")

    left, right = st.columns([1, 2])
    with left:
        model = st.selectbox("Model", options=["qwen3.5:4b", "qwen3.5:9b"], index=0)
        think_mode = st.selectbox("Thinking", options=["disabled", "provider default", "enabled"], index=0)
        json_mode = st.checkbox("JSON mode", value=False)
        system_prompt = st.text_area(
            "System Prompt",
            value="Reply concisely. Do not include thinking or reasoning traces.",
            height=100,
        )
        user_prompt = st.text_area(
            "User Prompt",
            value='Return exactly: {"ok": true}',
            height=180,
        )
        if st.button("Run Model Prompt", use_container_width=True):
            think_value = {"disabled": False, "provider default": None, "enabled": True}[think_mode]
            with st.spinner(f"Running {model}..."):
                st.session_state["model_prompt_payload"] = _post(
                    "/debug/model-prompt",
                    {
                        "model": model,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "think": think_value,
                        "json_mode": json_mode,
                    },
                )

    with right:
        payload = st.session_state.get("model_prompt_payload")
        if not payload:
            st.info("Run a prompt to inspect raw and cleaned model output.")
            return
        summary_cols = st.columns(4)
        summary_cols[0].metric("Requested", payload.get("model_requested", ""))
        summary_cols[1].metric("Response", payload.get("model_response", ""))
        summary_cols[2].metric("Duration", _format_duration(payload.get("duration_ms")))
        summary_cols[3].metric("Think Tags", "yes" if payload.get("raw_contains_think_tag") else "no")

        raw_col, cleaned_col = st.columns(2)
        with raw_col:
            _render_json_block(
                "Raw Output",
                {
                    "raw_content": payload.get("raw_content", ""),
                    "raw_contains_think_tag": payload.get("raw_contains_think_tag"),
                },
            )
        with cleaned_col:
            _render_json_block(
                "Cleaned Output",
                {
                    "cleaned_content": payload.get("cleaned_content", ""),
                    "cleaned_contains_think_tag": payload.get("cleaned_contains_think_tag"),
                },
            )
        _render_json_block("Request Payload", payload.get("request_payload", {}))
        _render_json_block("Raw Ollama Response", payload.get("ollama_response", {}))


if __name__ == "__main__":
    main()
