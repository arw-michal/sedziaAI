# app.py — Asystent AI (Streamlit) + auto-test API Watson Orchestrate
import os
import time
from datetime import datetime

import requests
import streamlit as st

# -----------------------------
# USTAWIENIA STRONY
# -----------------------------
st.set_page_config(page_title="Asystent AI - Demo", page_icon="⚖️", layout="wide")
st.write("")

# -----------------------------
# KONFIG / SECRETS
# -----------------------------
def cfg(name: str, default=None):
    # Czytaj najpierw ze secrets, ale nie wywalaj app przy braku pliku
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

SERVICE_URL = (cfg("SERVICE_INSTANCE_URL", "") or "").rstrip("/")
API_KEY = cfg("API_KEY", "")
DEMO_MODE = str(cfg("DEMO_MODE", "true")).lower() == "true"  # bez secrets domyślnie demo
TIMEOUT = float(cfg("TIMEOUT_SECONDS", 30))
RETRY_COUNT = int(cfg("RETRY_COUNT", 1))
RETRY_BACKOFF = float(cfg("RETRY_BACKOFF", 1.5))
AUTH_SCHEME = str(cfg("AUTH_SCHEME", "bearer")).lower()
ENDPOINT_TEMPLATE = str(cfg("ENDPOINT_TEMPLATE", "")).strip()  # opcjonalny

# Mapowanie agentów na nazwy w Orchestrate (z secrets lub default)
AGENT_NAME_FOR = {
    "Kradzież": cfg("AGENT_KRADZIEZ", "SedziaAI.1"),
    "Zakłócanie miru domowego": cfg("AGENT_MIR_DOMOWY", "SedziaAI.2"),
    "Groźba": cfg("AGENT_GROZBA", "SedziaAI.3"),
    "Uszkodzenie ciała": cfg("AGENT_USZKODZENIE", "SedziaAI.4"),
}

# -----------------------------
# DANE DEMO (na sztywno)
# -----------------------------
JUDGE_NAME = "Sędzia: Jan Kowalski"
CASES = {
    "sprawa Nr.221": "Kradzież",
    "sprawa Nr.325": "Zakłócanie miru domowego",
    "sprawa Nr.523": "Uszkodzenie ciała",
    "sprawa Nr.128": "Groźba",
}

# -----------------------------
# STAN SESJI
# -----------------------------
if "active_case" not in st.session_state:
    st.session_state.active_case = list(CASES.keys())[0]

if "chats" not in st.session_state:
    st.session_state.chats = {k: [] for k in CASES.keys()}

# Status testu API: {"ok": bool, "msg": str, "when": str}
if "api_status" not in st.session_state:
    st.session_state.api_status = None

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
    mapping = {
        "Kradzież": "Agent (Kradzież)",
        "Zakłócanie miru domowego": "Agent (Mir domowy)",
        "Uszkodzenie ciała": "Agent (Uszkodzenie ciała)",
        "Groźba": "Agent (Groźba)",
    }
    return mapping.get(topic, "Agent AI")

# -----------------------------
# STUB DEMO (bez API)
# -----------------------------
def demo_reply(case_key: str, user_text: str) -> str:
    base = {
        "Kradzież": "Kradzież: istotne są zamiar, zawłaszczenie i wartość mienia.",
        "Zakłócanie miru domowego": "Zakłócanie miru: bezprawne wdarcie lub nieopuszczenie mimo żądania.",
        "Uszkodzenie ciała": "Uszkodzenie ciała: zakres obrażeń i czas naruszenia funkcji narządu.",
        "Groźba": "Groźba: realność spełnienia i uzasadniona obawa zagrożonego.",
    }.get(CASES[case_key], "Przyjmuję kontekst sprawy.")
    return f"{base}\n\n(DEM0) Otrzymałem: „{user_text}”. Mam przygotować szkic uzasadnienia, pytania na rozprawę czy podsumowanie akt?"

# -----------------------------
# POŁĄCZENIE Z WATSON ORCHESTRATE (LIVE)
# -----------------------------
def _auth_headers():
    if AUTH_SCHEME == "x-api-key":
        return {"x-api-key": API_KEY, "Content-Type": "application/json"}
    # domyślnie bearer
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def _candidate_urls(agent_label: str):
    base = SERVICE_URL.rstrip("/")
    if ENDPOINT_TEMPLATE:
        yield ENDPOINT_TEMPLATE.format(base=base, agent=agent_label)
        return
    # spróbuj popularnych ścieżek
    for c in [
        "{base}/api/agents/{agent}/chat",
        "{base}/v1/agents/{agent}:chat",
        "{base}/api/v1/agents/{agent}/chat",
        "{base}/orchestrate/api/v1/agents/{agent}/chat",
    ]:
        yield c.format(base=base, agent=agent_label)

def call_orchestrate(agent_label: str, messages: list[str]) -> str:
    if not SERVICE_URL or not API_KEY:
        raise RuntimeError("Brak SERVICE_INSTANCE_URL lub API_KEY (secrets.toml).")

    headers = _auth_headers()
    payload = {
        "messages": [{"role": "user", "content": m} for m in messages],
        "metadata": {"user_id": "s_jan_kowalski_demo", "purpose": "demo_court_assistant"},
    }

    last_err = None
    tried = []
    for url in _candidate_urls(agent_label):
        tried.append(url)
        for attempt in range(RETRY_COUNT + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("reply") or data.get("text") or "[Brak treści]"
                elif resp.status_code in (401, 403):
                    raise RuntimeError(f"Autoryzacja odrzucona ({resp.status_code}) dla URL: {url}")
                else:
                    last_err = f"{resp.status_code} {resp.text[:400]}"
            except Exception as e:
                last_err = str(e)
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF ** attempt)

    raise RuntimeError(
        "Nie znaleziono działającej ścieżki.\n"
        f"Ostatni błąd: {last_err}\n"
        "Próbowane URL-e:\n- " + "\n- ".join(tried)
    )

def get_reply(case_key: str, user_text: str) -> str:
    if DEMO_MODE:
        return demo_reply(case_key, user_text)
    topic = CASES[case_key]
    agent_label = AGENT_NAME_FOR.get(topic)
    if not agent_label:
        raise RuntimeError(f"Brak mapowania agenta dla tematu: {topic}")
    # kontekst: ostatnie 5 wejść użytkownika + nowa wiadomość
    history_user = [m["text"] for m in st.session_state.chats[case_key] if m["role"] == "user"]
    messages = (history_user + [user_text])[-5:]
    return call_orchestrate(agent_label, messages)

# -----------------------------
# AUTO-TEST API NA STARCIE
# -----------------------------
def run_api_smoketest():
    """Wykonuje lekki test połączenia i zapisuje wynik do session_state.api_status."""
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if DEMO_MODE:
        st.session_state.api_status = {"ok": False, "msg": "Tryb DEMO — test API pominięty.", "when": when}
        return

    try:
        # bierzemy agenta od „Kradzież” jako kanarka
        agent = AGENT_NAME_FOR["Kradzież"]
        _ = call_orchestrate(agent, ["ping"])
        st.session_state.api_status = {"ok": True, "msg": "Połączono z Watson Orchestrate.", "when": when}
    except Exception as e:
        st.session_state.api_status = {"ok": False, "msg": f"Błąd: {e}", "when": when}

# Uruchom smoketest raz po starcie (lub gdy ręcznie wyzerowano)
if st.session_state.api_status is None:
    run_api_smoketest()

# Pasek statusu u góry
if st.session_state.api_status:
    status = st.session_state.api_status
    if status["ok"]:
        st.success(f"✅ API OK — {status['msg']} ( {status['when']} )", icon="✅")
    else:
        if DEMO_MODE:
            st.info(f"ℹ️ {status['msg']} ( {status['when']} )", icon="ℹ️")
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
        format_func=lambda k: CASES[k]
    )
    if selected != st.session_state.active_case:
        st.session_state.active_case = selected

    st.divider()
    st.button("📎 Załącz dokument (niedostępne w demo)", disabled=True, help="Makieta – brak funkcji")
    st.divider()

    if st.button("♻️ Resetuj rozmowę dla tej sprawy"):
        reset_chat(st.session_state.active_case)
        st.success("Historia dla tej sprawy wyczyszczona.")

    with st.expander("🔧 Diagnostyka API", expanded=False):
        st.caption(f"BASE: {SERVICE_URL or 'brak'}")
        st.caption(f"AUTH_SCHEME: {AUTH_SCHEME}")
        st.caption(f"ENDPOINT_TEMPLATE: {ENDPOINT_TEMPLATE or '(auto)'}")
        if st.button("🔁 Powtórz test API"):
            run_api_smoketest()
            st.experimental_rerun()

# -----------------------------
# LAYOUT / GŁÓWNA CZĘŚĆ
# -----------------------------
left, right = st.columns([2.2, 1])
case_key = st.session_state.active_case

with left:
    st.subheader(CASES[case_key])
    with st.container(border=True):
        mode = "demo (bez połączenia z Watson Orchestrate)" if DEMO_MODE else "LIVE (połączono z Watson Orchestrate)"
        st.markdown(f"**Tryb:** {mode}. Odpowiedzi: {'stuby' if DEMO_MODE else 'z agentów Orchestrate'}.")

    has_chat = hasattr(st, "chat_message") and hasattr(st, "chat_input")

    # Historia rozmowy
    for msg in st.session_state.chats[case_key]:
        if has_chat:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["text"])
                st.caption(msg["ts"])
        else:
            # Fallback dla starszego Streamlit
            box_color = "#1f6feb22" if msg["role"] == "user" else "#2ea04322"
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;background:{box_color};margin-bottom:6px'>"
                f"<b>{'Ty' if msg['role']=='user' else 'Agent'}</b><br>{msg['text']}<br>"
                f"<span style='font-size:12px;color:#888'>{msg['ts']}</span></div>",
                unsafe_allow_html=True
            )

    # Wejście użytkownika
    user_text = st.chat_input("Napisz wiadomość do agenta…") if has_chat else st.text_input("Napisz wiadomość do agenta… (tryb zgodności)")

    if user_text:
        add_message(case_key, "user", user_text)
        try:
            reply = get_reply(case_key, user_text)
        except Exception as e:
            reply = f"❌ Błąd połączenia z agentem: {e}\n(Sprawdź SERVICE_INSTANCE_URL/API_KEY/AUTH_SCHEME oraz ścieżkę w call_orchestrate())"
        add_message(case_key, "assistant", reply)
        st.rerun()

with right:
    st.subheader("Szczegóły agenta")
    st.markdown(f"**Aktywny agent:** {get_agent_name(case_key)}")
    st.markdown("- **Status:** " + ("🟡 makieta (bez API)" if DEMO_MODE else ("🟢 API OK" if (st.session_state.api_status and st.session_state.api_status.get("ok")) else "🔴 API problem")))
    st.markdown("- **Funkcje:** analiza zapytań, streszczenia, listy pytań")
    st.divider()
    st.subheader("Szybkie akcje")
    c1, c2 = st.columns(2)
    if c1.button("📝 Szkic uzasadnienia"):
        add_message(case_key, "assistant", "DEMO/LIVE: szkic uzasadnienia – generowanie zlecone.")
        st.rerun()
    if c2.button("📋 Pytania na rozprawę"):
        add_message(case_key, "assistant", "DEMO/LIVE: lista pytań – generowanie zlecone.")
        st.rerun()
    if st.button("🧾 Podsumowanie akt"):
        add_message(case_key, "assistant", "DEMO/LIVE: podsumowanie akt – generowanie zlecone.")
        st.rerun()
