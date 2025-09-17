# app.py — Streamlit UI + 1 agent WML (SedziaAI.4 — Uszkodzenie ciała)
# ---------------------------------------------------------------------
# Copy‑paste ready. Minimalna integracja z IBM Watson Machine Learning
# dla jednego deploymentu (SedziaAI.4). Reszta UI — jak w demie.
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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# KONFIG / SECRETS (copy‑paste)
# -----------------------------
def cfg(name: str, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

# WYMAGANE
IBM_CLOUD_API_KEY: str = cfg("IBM_CLOUD_API_KEY", "")  # wstaw w .streamlit/secrets.toml
WML_BASE_URL: str = (cfg("WML_BASE_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/"))
DEPLOYMENT_ID: str = cfg("DEPLOYMENT_ID", "")  # np. "2717d333-4597-4afb-80b4-7dbdb26999b0"

# OPCJONALNE
API_VERSION: str = str(cfg("API_VERSION", "2021-05-01"))
USE_STREAM: bool = str(cfg("USE_STREAM", "false")).lower() == "true"  # użyj /ai_service_stream
TIMEOUT = float(cfg("TIMEOUT_SECONDS", 30))
RETRY_COUNT = int(cfg("RETRY_COUNT", 1))
RETRY_BACKOFF = float(cfg("RETRY_BACKOFF", 1.5))

# -----------------------------
# DANE / MAPOWANIA (tylko 1 agent na start)
# -----------------------------
JUDGE_NAME = "Sędzia: Jan Kowalski"
CASES: Dict[str, str] = {
    "sprawa Nr.523": "Uszkodzenie ciała",  # aktywna do testów
}
AGENT_NAME_FOR = {
    "Uszkodzenie ciała": "SedziaAI.4",
}

# -----------------------------
# STAN SESJI
# -----------------------------
if "active_case" not in st.session_state:
    st.session_state.active_case = list(CASES.keys())[0]

if "chats" not in st.session_state:
    st.session_state.chats = {k: [] for k in CASES.keys()}

if "api_status" not in st.session_state:
    st.session_state.api_status = None  # {ok: bool, msg: str, when: str}

# -----------------------------
# FUNKCJE POMOCNICZE (chat)
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
# WARSTWA WML — 1 AGENT
# -----------------------------
_token_cache: Dict[str, Any] = {"token": None, "exp": 0.0}


def _iam_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["exp"]:
        return _token_cache["token"]
    if not IBM_CLOUD_API_KEY:
        raise RuntimeError("Brak IBM_CLOUD_API_KEY (secrets.toml lub env).")
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
    return {
        "Authorization": f"Bearer {_iam_token()}",
        "Content-Type": "application/json",
    }


def _extract_reply(js: Dict[str, Any]) -> Optional[str]:
    # 1) generated_text
    try:
        return js["results"][0]["generated_text"]
    except Exception:
        pass
    # 2) output->content[]->text
    try:
        blocks = js["results"][0]["output"][0]["content"]
        for b in blocks:
            if isinstance(b, dict) and "text" in b:
                return b["text"]
    except Exception:
        pass
    # 3) response
    try:
        return js["results"][0]["response"]
    except Exception:
        pass
    # 4) fallback
    if isinstance(js, dict):
        return js.get("reply") or js.get("text")
    return None


def call_wml_chat(messages: List[str]) -> str:
    url = _build_wml_url()
    headers = _headers_bearer()
    payload = {"messages": [{"content": m, "role": "user"} for m in messages]}

    last_err = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 200:
                js = resp.json()
                out = _extract_reply(js)
                return out or "[Brak treści w odpowiedzi modelu]"
            elif resp.status_code in (401, 403):
                _token_cache["token"] = None  # odśwież
                last_err = f"{resp.status_code} {resp.text[:400]}"
            else:
                last_err = f"{resp.status_code} {resp.text[:400]}"
        except Exception as e:
            last_err = str(e)
        if attempt < RETRY_COUNT:
            time.sleep(RETRY_BACKOFF ** attempt)
    raise RuntimeError(f"Błąd wywołania WML: {last_err}")


def get_reply(case_key: str, user_text: str) -> str:
    # proste utrzymanie krótkiego kontekstu użytkownika
    history_user = [m["text"] for m in st.session_state.chats[case_key] if m["role"] == "user"]
    messages = (history_user + [user_text])[-5:]
    return call_wml_chat(messages)

# -----------------------------
# SMOKETEST PO STARcie
# -----------------------------

def run_api_smoketest():
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        _ = call_wml_chat(["ping"])
        st.session_state.api_status = {"ok": True, "msg": "Połączono z IBM WML (SedziaAI.4).", "when": when}
    except Exception as e:
        st.session_state.api_status = {"ok": False, "msg": f"Błąd: {e}", "when": when}

if st.session_state.api_status is None:
    run_api_smoketest()

# Pasek statusu
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

    keys = list(CASES.keys())
    try:
        current_idx = keys.index(st.session_state.active_case)
    except ValueError:
        current_idx = 0

    selected = st.radio(
        "Wybierz sprawę:",
        keys,
        index=current_idx,
        format_func=lambda k: CASES[k],
    )
    if selected != st.session_state.active_case:
        st.session_state.active_case = selected

    st.divider()
    st.button("📎 Załącz dokument", disabled=True, help="Funkcja w przygotowaniu")
    st.divider()

    if st.button("♻️ Resetuj rozmowę dla tej sprawy"):
        reset_chat(st.session_state.active_case)
        st.success("Historia dla tej sprawy wyczyszczona.")

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
    st.subheader(CASES[case_key], anchor=False)
    with st.container():
        st.markdown(
            "<div class='status-bar'><b>Tryb:</b> LIVE – połączenie z IBM WML (SedziaAI.4)." \
            "</div>",
            unsafe_allow_html=True,
        )

    # Historia rozmowy
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

    # Wejście użytkownika
    prompt_label = "Napisz wiadomość do agenta…"
    user_text = st.chat_input(prompt_label) if has_chat else st.text_input(prompt_label)

    if user_text:
        add_message(case_key, "user", user_text)
        try:
            reply = get_reply(case_key, user_text)
        except Exception as e:
            reply = (
                "❌ Błąd połączenia z deploymentem WML: "
                f"{e}\n(Sprawdź WML_BASE_URL / DEPLOYMENT_ID / IBM_CLOUD_API_KEY oraz uprawnienia)"
            )
        add_message(case_key, "assistant", reply)
        st.rerun()

with right:
    st.subheader("Szczegóły agenta", anchor=False)
    st.markdown(f"**Aktywny agent:** {get_agent_name(case_key)}")
    st.markdown("- **Status:** " + ("🟢 API OK" if (st.session_state.api_status and st.session_state.api_status.get("ok")) else "🔴 API problem"))
    st.markdown("- **Funkcje:** analiza zapytań, streszczenia, listy pytań")
    st.divider()
    st.subheader("Szybkie akcje", anchor=False)
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


