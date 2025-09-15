import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time

API_URL = "http://localhost:8000"
USERNAME = "bundeskanzler"
PASSWORD = "ki2025"

st.title("🤖 Bundeskanzler KI - Web GUI")

def show_admin_interface():
    """Zeigt das Admin-Interface"""
    st.title("🔐 Admin Panel - Bundeskanzler KI")
    
    admin_headers = {"Authorization": f"Bearer {st.session_state['admin_token']}"}
    
    # Admin Tabs
    admin_tabs = st.tabs([
        "📊 Dashboard", 
        "👥 Benutzer-Management", 
        "📋 Log-Viewer", 
        "💾 Memory-Management",
        "⚙️ Konfiguration"
    ])
    
    # === DASHBOARD TAB ===
    with admin_tabs[0]:
        st.subheader("System Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        # System Stats
        try:
            resp = requests.get(f"{API_URL}/admin/system-stats", headers=admin_headers, timeout=10)
            if resp.status_code == 200:
                stats = resp.json()
                
                with col1:
                    st.metric("API Requests (24h)", stats.get("api_requests_24h", 0))
                    st.metric("Active Users", stats.get("active_users", 0))
                
                with col2:
                    st.metric("Memory Entries", stats.get("memory_entries", 0))
                    st.metric("Error Rate %", f"{stats.get('error_rate', 0):.1f}%")
                
                with col3:
                    st.metric("CPU Usage %", f"{stats.get('cpu_usage', 0):.1f}%")
                    st.metric("Memory Usage %", f"{stats.get('memory_usage', 0):.1f}%")
            
            # Health Check
            st.markdown("---")
            st.subheader("System Health")
            resp_health = requests.get(f"{API_URL}/admin/health", headers=admin_headers, timeout=10)
            if resp_health.status_code == 200:
                health = resp_health.json()
                
                if health["system"]["components_initialized"]:
                    st.success("✅ Alle Komponenten initialisiert")
                else:
                    st.error("❌ Komponenten nicht vollständig initialisiert")
                    
                if health["files"]["logs_accessible"]:
                    st.success("✅ Log-Dateien zugänglich")
                else:
                    st.warning("⚠️ Log-Dateien nicht alle zugänglich")
                    
                st.info(f"🕐 Uptime: {health['system']['uptime']:.1f} Sekunden")
                st.info(f"📈 Requests: {health['system']['request_count']}")
                
        except Exception as e:
            st.error(f"Fehler beim Laden der Dashboard-Daten: {e}")
    
    # === BENUTZER-MANAGEMENT TAB ===
    with admin_tabs[1]:
        st.subheader("Benutzer-Management")
        
        # Alle Benutzer anzeigen
        try:
            resp = requests.get(f"{API_URL}/admin/users", headers=admin_headers, timeout=10)
            if resp.status_code == 200:
                users_data = resp.json()
                users = users_data.get("users", [])
                
                if users:
                    # Benutzer-Tabelle
                    user_df = pd.DataFrame([
                        {
                            "User ID": user["user_id"],
                            "Email": user["email"],
                            "Admin": "✅" if user["is_admin"] else "❌",
                            "Active": "✅" if user["is_active"] else "❌",
                            "Created": user["created_at"],
                            "Login Count": user["login_count"]
                        }
                        for user in users
                    ])
                    st.dataframe(user_df, use_container_width=True)
                    
                    # Benutzer deaktivieren
                    st.markdown("---")
                    st.subheader("Benutzer deaktivieren")
                    user_to_deactivate = st.selectbox("Benutzer auswählen", 
                        [u["user_id"] for u in users if u["is_active"]])
                    
                    if st.button("Benutzer deaktivieren", type="secondary"):
                        try:
                            resp_del = requests.delete(
                                f"{API_URL}/admin/users/{user_to_deactivate}", 
                                headers=admin_headers, 
                                timeout=10
                            )
                            if resp_del.status_code == 200:
                                st.success(f"Benutzer {user_to_deactivate} deaktiviert!")
                                st.rerun()
                            else:
                                st.error("Fehler beim Deaktivieren")
                        except Exception as e:
                            st.error(f"Fehler: {e}")
                
                # Neuen Benutzer erstellen
                st.markdown("---")
                st.subheader("Neuen Benutzer erstellen")
                with st.form("create_user_form"):
                    new_user_id = st.text_input("User ID")
                    new_email = st.text_input("Email")
                    new_password = st.text_input("Password", type="password")
                    new_is_admin = st.checkbox("Admin-Rechte")
                    
                    if st.form_submit_button("Benutzer erstellen"):
                        try:
                            user_data = {
                                "user_id": new_user_id,
                                "email": new_email,
                                "password": new_password,
                                "is_admin": new_is_admin
                            }
                            resp_create = requests.post(
                                f"{API_URL}/admin/users", 
                                json=user_data, 
                                headers=admin_headers, 
                                timeout=10
                            )
                            if resp_create.status_code == 200:
                                st.success("Benutzer erfolgreich erstellt!")
                                st.rerun()
                            else:
                                st.error("Fehler beim Erstellen des Benutzers")
                        except Exception as e:
                            st.error(f"Fehler: {e}")
                            
        except Exception as e:
            st.error(f"Fehler beim Laden der Benutzer: {e}")
    
    # === LOG-VIEWER TAB ===
    with admin_tabs[2]:
        st.subheader("Live Log-Viewer")
        
        log_type = st.selectbox("Log-Datei", ["api.log", "memory.log", "errors.log"])
        lines_count = st.slider("Anzahl Zeilen", 10, 200, 50)
        
        if st.button("Logs laden") or st.button("🔄 Refresh"):
            try:
                resp = requests.get(
                    f"{API_URL}/admin/logs/{log_type}?lines={lines_count}", 
                    headers=admin_headers, 
                    timeout=10
                )
                if resp.status_code == 200:
                    log_data = resp.json()
                    entries = log_data.get("entries", [])
                    
                    if entries:
                        # Log-Einträge in umgekehrter Reihenfolge (neueste zuerst)
                        for entry in reversed(entries):
                            level_color = {
                                "ERROR": "🔴",
                                "WARNING": "🟡", 
                                "INFO": "🔵",
                                "DEBUG": "⚪"
                            }.get(entry["level"], "⚪")
                            
                            st.markdown(f"""
                            **{level_color} {entry['timestamp']}** - {entry['level']} - {entry['logger']}  
                            {entry['message']}
                            """)
                    else:
                        st.info("Keine Log-Einträge gefunden")
                else:
                    st.error(f"Fehler beim Laden der Logs: {resp.status_code}")
            except Exception as e:
                st.error(f"Fehler: {e}")
    
    # === MEMORY-MANAGEMENT TAB ===
    with admin_tabs[3]:
        st.subheader("Memory-Management")
        
        # Memory Stats
        try:
            resp = requests.get(f"{API_URL}/admin/memory/stats", headers=admin_headers, timeout=10)
            if resp.status_code == 200:
                memory_stats = resp.json()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Kurzzeitgedächtnis", memory_stats.get("kurzzeitgedaechtnis_entries", 0))
                with col2:
                    st.metric("Langzeitgedächtnis", memory_stats.get("langzeitgedaechtnis_entries", 0))
                with col3:
                    st.metric("Effizienz %", f"{memory_stats.get('memory_efficiency', 0):.1f}%")
                    
                # Memory leeren (mit Bestätigung)
                st.markdown("---")
                st.subheader("⚠️ Memory verwalten")
                st.warning("**ACHTUNG:** Das Löschen des Memory ist irreversibel!")
                
                confirm_clear = st.checkbox("Ich bestätige, dass ich das gesamte Memory löschen möchte")
                
                if st.button("🗑️ Memory komplett leeren", type="secondary", disabled=not confirm_clear):
                    try:
                        resp_clear = requests.post(
                            f"{API_URL}/admin/memory/clear", 
                            headers=admin_headers, 
                            timeout=10
                        )
                        if resp_clear.status_code == 200:
                            result = resp_clear.json()
                            st.success("Memory erfolgreich geleert!")
                            st.info(f"Backup erstellt: {result.get('backup', 'N/A')}")
                            st.rerun()
                        else:
                            st.error("Fehler beim Leeren des Memory")
                    except Exception as e:
                        st.error(f"Fehler: {e}")
                        
        except Exception as e:
            st.error(f"Fehler beim Laden der Memory-Statistiken: {e}")
    
    # === KONFIGURATION TAB ===
    with admin_tabs[4]:
        st.subheader("System-Konfiguration")
        
        try:
            resp = requests.get(f"{API_URL}/admin/config", headers=admin_headers, timeout=10)
            if resp.status_code == 200:
                config = resp.json()
                
                st.subheader("API Settings")
                api_settings = config.get("api_settings", {})
                st.json(api_settings)
                
                st.subheader("Memory Settings") 
                memory_settings = config.get("memory_settings", {})
                st.json(memory_settings)
                
                st.subheader("Logging Settings")
                logging_settings = config.get("logging_settings", {})
                st.json(logging_settings)
                
                st.subheader("Security Settings")
                security_settings = config.get("security_settings", {})
                st.json(security_settings)
                
            else:
                st.error("Fehler beim Laden der Konfiguration")
                
        except Exception as e:
            st.error(f"Fehler: {e}")

# --- Sidebar für Admin-Login ---
st.sidebar.header("Login")
login_type = st.sidebar.radio("Login-Typ", ["Benutzer", "Admin"])

if login_type == "Admin":
    admin_username = st.sidebar.text_input("Admin Username", value="admin")
    admin_password = st.sidebar.text_input("Admin Password", type="password", value="admin123!")
    admin_login = st.sidebar.button("Admin Login")
    
    if admin_login or st.session_state.get("admin_logged_in", False):
        if not st.session_state.get("admin_logged_in", False):
            try:
                resp = requests.post(
                    f"{API_URL}/auth/admin-token", 
                    data={"username": admin_username, "password": admin_password}, 
                    timeout=10
                )
                if resp.status_code == 200:
                    st.session_state["admin_token"] = resp.json()["access_token"]
                    st.session_state["admin_logged_in"] = True
                    st.sidebar.success("Admin Login erfolgreich!")
                else:
                    st.sidebar.error("Admin Login fehlgeschlagen!")
                    st.stop()
            except Exception as e:
                st.sidebar.error(f"Fehler bei Admin-Authentifizierung: {e}")
                st.stop()
                
        # Admin-Interface anzeigen
        show_admin_interface()
        st.stop()

# --- Normale Benutzer-Authentifizierung ---
@st.cache_data(show_spinner=False)
def get_token():
    try:
        resp = requests.post(f"{API_URL}/auth/token", data={"username": USERNAME, "password": PASSWORD}, timeout=10)
        if resp.status_code == 200:
            return resp.json()["access_token"]
    except Exception as e:
        st.error(f"Fehler bei Authentifizierung: {e}")
    return None

token = get_token()
if not token:
    st.error("Authentifizierung fehlgeschlagen. API aktiv?")
    st.stop()

headers = {"Authorization": f"Bearer {token}"}


# --- Normale Benutzer-Authentifizierung ---

# --- Tabs ---
tabs = st.tabs(["💬 Chat", "🔍 Memory-Suche", "📊 Statistiken"])

# --- Chat Tab ---
with tabs[0]:
    st.subheader("Chat mit der Bundeskanzler KI")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_input = st.text_input("Ihre Frage:", key="chat_input")
    if st.button("Absenden", key="chat_send") and user_input:
        data = {"message": user_input, "include_sources": True, "max_length": 500}
        try:
            resp = requests.post(f"{API_URL}/chat", json=data, headers=headers, timeout=15)
            if resp.status_code == 200:
                result = resp.json()
                st.session_state["chat_history"].append((user_input, result["response"]))
            else:
                st.warning(f"Fehler: {resp.status_code}")
        except Exception as e:
            st.warning(f"Fehler: {e}")
    for user, answer in reversed(st.session_state["chat_history"]):
        st.markdown(f"**Sie:** {user}")
        st.markdown(f"**KI:** {answer}")


# --- Memory-Suche & Explorer Tab ---
with tabs[1]:
    st.subheader("Memory durchsuchen & Explorer")
    search_query = st.text_input("Suchbegriff:", key="memory_search")
    top_k = st.slider("Anzahl Ergebnisse", 1, 10, 5)
    if st.button("Suchen", key="memory_search_btn") and search_query:
        data = {"query": search_query, "top_k": top_k, "min_similarity": 0.1}
        try:
            resp = requests.post(f"{API_URL}/memory/search", json=data, headers=headers, timeout=10)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    for i, r in enumerate(results, 1):
                        st.markdown(f"**{i}.** {r['content']}")
                        st.caption(f"Ähnlichkeit: {r['similarity']:.3f}")
                else:
                    st.info("Keine Ergebnisse gefunden.")
            else:
                st.warning(f"Fehler: {resp.status_code}")
        except Exception as e:
            st.warning(f"Fehler: {e}")

    st.markdown("---")
    st.subheader("Alle Memories (Explorer)")
    if st.button("Alle Memories laden", key="load_all_memories") or "all_memories" not in st.session_state:
        try:
            resp = requests.get(f"{API_URL}/memory/all", headers=headers, timeout=15)
            if resp.status_code == 200:
                st.session_state["all_memories"] = resp.json().get("memories", [])
            else:
                st.session_state["all_memories"] = []
                st.info("Endpoint /memory/all nicht verfügbar oder Fehler beim Laden.")
        except Exception as e:
            st.session_state["all_memories"] = []
            st.warning(f"Fehler: {e}")
    memories = st.session_state.get("all_memories", [])
    if memories:
        df = pd.DataFrame(memories)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Export als CSV", csv, "memories.csv", "text/csv")
    else:
        st.info("Noch keine Memories geladen.")

# --- Statistiken Tab ---
with tabs[2]:
    st.subheader("System- und Memory-Statistiken")
    try:
        resp = requests.get(f"{API_URL}/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.write(f"**Status:** {data['status']}")
            st.write(f"**Version:** {data.get('version', '-')}")
            st.write(f"**Uptime:** {data.get('uptime', 0):.1f} Sekunden")
            st.write(f"**Requests:** {data.get('request_count', 0)}")
        resp2 = requests.get(f"{API_URL}/memory/stats", headers=headers, timeout=5)
        if resp2.status_code == 200:
            stats = resp2.json()
            st.write(f"**Memory gesamt:** {stats.get('total_memories', 0)}")
            st.write(f"**Kurzzeitgedächtnis:** {stats.get('short_term_count', 0)}")
            st.write(f"**Langzeitgedächtnis:** {stats.get('long_term_count', 0)}")
    except Exception as e:
        st.warning(f"Fehler: {e}")