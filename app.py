# app.py — Streamlit UI + 1 agent WML (SedziaAI.4 — Uszkodzenie ciała)
# ---------------------------------------------------------------------
# Integracja z IBM Watson Machine Learning + persistent debug.
# Wysyłamy tylko ostatnią wiadomość użytkownika (bez historii).
# ---------------------------------------------------------------------

from __future__ import annotations
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import requests
import streamlit as st

# -----------------------------
# USTAWIENIA STRONY / TEMA
# -----------------------------
st.set_page_config(
    page_title="Asystent AI – SedziaAI.4",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.chat-bubble {padding: 10px; border-radius: 10px; margin-bottom: 8px;}
.chat-user {background: rgba(31,110,235,0.10);}  /* niebieski */
.chat-assistant {background: rgba(46,160,67,0.10);} /* zielony */
.status-bar {padding: 10px 14px; border-radius: 10px; background: rgba(0,0,0,0.04);} 
.small {font-size: 12px; color: #6b7280;}
pre {white-space: pre-wrap;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# KONFIG / SECRETS
# -----------------------------
def cfg(name: str, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

IBM_CLOUD_API_KEY: str = cfg("IBM_CLOUD_API_KEY", "")
WML_BASE_URL: str = (cfg("WML_BASE_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/"))
DEPLOYMENT_ID: str = cfg("DEPLOYMENT_ID", "")
API_VERSION: str = str(cfg("API_VERSION", "2021-05-01"))
USE_STREAM: bool = str(cfg("USE_STREAM", "false")).lower() == "true"
TIMEOUT = float(cfg("TIMEOUT_SECONDS", 30))
RETRY_COUNT = int(cfg("RETRY_COUNT", 1))
RETRY_BACKOFF = float(cfg("RETRY_BACKOFF", 1.5))
SHOW_WML_DEBUG: bool = str(cfg("SHOW_WML_DEBUG", "true")).lower() == "true"

# -----------------------------
# DANE / MAPOWANIA (1 agent)
# -----------------------------
JUDGE_NAME = "Sędzia: Jan Kowalski"
CASES: Dict[str, str] = {"sprawa Nr.523": "Uszkodzenie ciała"}
AGENT_NAME_FOR = {"Uszkodzenie ciała": "SedziaAI.4"}

# -----------------------------
# STAN SESJI
# -----------------------------
if "active_case" not in st.session_state:
    st.session_state.active_case = list(CASES.keys())[0]

if "chats" not in st.session_state:
    st.session_state.chats = {k: [] for k in CASES.keys()}

if "api_status" not in st.session_state:
    st.session_state.api_status = None

if "wml_debug" not in st.session_state:
    st.session_state.wml_debug = None

# -----------------------------
# FUNKCJE POMOCNICZE
# -----------------------------
def add_message(case_key: str, role: str, text: str):
    st.session_state.chats[case_key].append({
        "role": role,
        "text": text,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

def reset_chat(case_key: str):
    st.session_state.chats[case_key] = []

def get_agent_name(case_key: str) -> str:
    topic = CASES[case_key]
    return AGENT_NAME_FOR.get(topic, "Agent AI")

# -----------------------------
# WARSTWA WML
# -----------------------------
_token_cache: Dict[str, Any] = {"token": None, "exp": 0.0}

def _iam_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["exp"]:
        return _token_cache["token"]
    if not IBM_CLOUD_API_KEY:
        raise RuntimeError("Brak IBM_CLOUD_API_KEY.")
    url = "https://iam.cloud.ibm.com/identity/token"
    data = {"apikey": IBM_CLOUD_API_KEY, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, data=data, headers=headers, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"IAM token error {resp.status_code}: {resp.text[:200]}")
    js = resp.json()
    token = js.get("access_token")
    expires_in = int(js.get("expires_in", 3600))
    if not token:
        raise RuntimeError("IAM token: brak access_token w odpowiedzi.")
    _token_cache["token"] = token
    _token_cache["exp"] = now + max(60, expires_in - 60)
    return token

def _build_wml_url() -> str:
    if not WML_BASE_URL or not DEPLOYMENT_ID:
        raise RuntimeError("Brak WML_BASE_URL lub DEPLOYMENT_ID.")
    path = "/ai_service_stream" if USE_STREAM else "/ai_service"
    return f"{WML_BASE_URL}/ml/v4/deployments/{DEPLOYMENT_ID}{path}?version={API_VERSION}"

def _headers_bearer() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_iam_token()}", "Content-Type": "application/json"}

def _extract_reply(js: Dict[str, Any]) -> Optional[str]:
    """Obsługa różnych kontraktów WML, w tym choices[].message.content."""
    if not isinstance(js, dict):
        return None
    # --- A) choices[0].message.content ---
    try:
        msg = js["choices"][0]["message"]
        if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"].strip():
            return msg["content"]
    except Exception:
        pass
    # --- B) choices[0].text (fallback) ---
    try:
        txt = js["choices"][0]["text"]
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        pass
    # --- C) klasyczne results ---
    for keypath in [
        ("results", 0, "generated_text"),
        ("results", 0, "response"),
        ("results", 0, "output_text"),
        ("results", 0, "output", 0, "text"),
    ]:
        try:
            val = js
            for k in keypath:
                val = val[k]
            if isinstance(val, str) and val.strip():
                return val
        except Exception:
            pass
    try:
        blocks = js["results"][0]["output"][0]["content"]
        for b in blocks:
            if isinstance(b, dict) and "text" in b and isinstance(b["text"], str) and b["text"].strip():
                return b["text"]
    except Exception:
        pass
    # --- D) predictions.values[0][0] ---
    try:
        val = js["predictions"][0]["values"][0][0]
        if isinstance(val, str) and val.strip():
            return val
    except Exception:
        pass
    # --- E) reply/text fallback ---
    if isinstance(js.get("reply"), str) and js["reply"].strip():
        return js["reply"]
    if isinstance(js.get("text"), str) and js["text"].strip():
        return js["text"]
    return None

def call_wml_chat(user_text: str) -> str:
    url = _build_wml_url()
    headers = _headers_bearer()
    payload = {"messages": [{"content": user_text, "role": "user"}]}

    last_err = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
            debug_entry = {
                "request_url": url,
                "request_payload": payload,
                "http_status": resp.status_code,
                "headers": dict(resp.headers),
                "text": None,
                "json": None,
            }
            try:
                js = resp.json()
                debug_entry["json"] = js
            except Exception:
                debug_entry["text"] = resp.text
            st.session_state.wml_debug = debug_entry

            if resp.status_code == 200 and debug_entry["json"] is not None:
                out = _extract_reply(debug_entry["json"])
                return out or "[Brak treści w odpowiedzi modelu]"
            elif resp.status_code in (401, 403):
                _token_cache["token"] = None
                last_err = f"{resp.status_code} {resp.text[:400]}"
            else:
                last_err = f"{resp.status_code} {resp.text[:400]}"
        except Exception as e:
            last_err = str(e)
        if attempt < RETRY_COUNT:
            time.sleep(RETRY_BACKOFF ** attempt)
    raise RuntimeError(f"Błąd wywołania WML: {last_err}")

# -----------------------------
# SMOKETEST
# -----------------------------
def run_api_smoketest():
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        _ = call_wml_chat("ping")
        st.session_state.api_status = {"ok": True, "msg": "Połączono z IBM WML (SedziaAI.4).", "when": when}
    except Exception as e:
        st.session_state.api_status = {"ok": False, "msg": f"Błąd: {e}", "when": when}

if st.session_state.api_status is None:
    run_api_smoketest()

if st.session_state.api_status:
    status = st.session_state.api_status
    if status["ok"]:
        st.success(f"✅ API OK — {status['msg']} ( {status['when']} )", icon="✅")
    else:
        st.error(f"❌ API problem — {status['msg']} ( {status['when']} )", icon="🚨")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("Asystent AI")
    st.caption(JUDGE_NAME)

    selected = st.radio("Wybierz sprawę:", list(CASES.keys()), format_func=lambda k: CASES[k])
    if selected != st.session_state.active_case:
        st.session_state.active_case = selected

    st.divider()
    st.button("📎 Załącz dokument", disabled=True)
    st.divider()

    if st.button("♻️ Resetuj rozmowę dla tej sprawy"):
        reset_chat(st.session_state.active_case)
        st.success("Historia wyczyszczona.")

    with st.expander("🔧 Diagnostyka API", expanded=False):
        st.caption(f"WML_BASE_URL: {WML_BASE_URL or 'brak'}")
        st.caption(f"DEPLOYMENT_ID: {DEPLOYMENT_ID or 'brak'}")
        st.caption(f"API_VERSION: {API_VERSION}")
        st.caption(f"Endpoint: {'/ai_service_stream' if USE_STREAM else '/ai_service'}")
        if st.button("🔁 Powtórz test API"):
            run_api_smoketest()
            st.experimental_rerun()

# -----------------------------
# GŁÓWNY LAYOUT
# -----------------------------
left, right = st.columns([2.2, 1])
case_key = st.session_state.active_case

with left:
    st.subheader(CASES[case_key])
    with st.container():
        st.markdown("<div class='status-bar'><b>Tryb:</b> LIVE – połączenie z IBM WML (SedziaAI.4).</div>", unsafe_allow_html=True)

    if SHOW_WML_DEBUG and st.session_state.wml_debug:
        dbg = st.session_state.wml_debug
        with st.expander("📦 Ostatnia odpowiedź WML (debug)", expanded=True):
            st.write(f"HTTP status: {dbg.get('http_status')}")
            st.code(dbg.get("request_payload"), language="json")
            if dbg.get("json") is not None:
                st.json(dbg["json"])
            elif dbg.get("text") is not None:
                st.text(dbg["text"])
            else:
                st.write("Brak ciała odpowiedzi.")

    has_chat = hasattr(st, "chat_message") and hasattr(st, "chat_input")
    for msg in st.session_state.chats[case_key]:
        if has_chat:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["text"])
                st.caption(msg["ts"])
        else:
            cls = "chat-user" if msg["role"] == "user" else "chat-assistant"
            st.markdown(
                f"<div class='chat-bubble {cls}'>"
                f"<b>{'Ty' if msg['role']=='user' else 'Agent'}</b><br>{msg['text']}<br>"
                f"<span class='small'>{msg['ts']}</span></div>",
                unsafe_allow_html=True,
            )

    user_text = st.chat_input("Napisz wiadomość do agenta…") if has_chat else st.text_input("Napisz wiadomość do agenta…")
    if user_text:
        add_message(case_key, "user", user_text)
        try:
            reply = call_wml_chat(user_text)
        except Exception as e:
            reply = f"❌ Błąd połączenia z deploymentem WML: {e}"
        add_message(case_key, "assistant", reply)
        st.rerun()

with right:
    st.subheader("Szczegóły agenta")
    st.markdown(f"**Aktywny agent:** {get_agent_name(case_key)}")
    st.markdown("- **Status:** " + ("🟢 API OK" if (st.session_state.api_status and st.session_state.api_status.get("ok")) else "🔴 API problem"))
    st.markdown("- **Funkcje:** analiza zapytań, streszczenia, listy pytań")
    st.divider()
    st.subheader("Szybkie akcje")
    c1, c2 = st.columns(2)
    if c1.button("📝 Szkic uzasadnienia"):
        add_message(case_key, "assistant", "LIVE: szkic uzasadnienia – generowanie zlecone.")
        st.rerun()
    if c2.button("📋 Pytania na rozprawę"):
        add_message(case_key, "assistant", "LIVE: lista pytań – generowanie zlecone.")
        st.rerun()
    if st.button("🧾 Podsumowanie akt"):
        add_message(case_key, "assistant", "LIVE: podsumowanie akt – generowanie zlecone.")
        st.rerun()
