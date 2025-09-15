import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import threading

API_URL = "http://localhost:8000"
USERNAME = "bundeskanzler"
PASSWORD = "ki2025"

# Global variables for live data
metrics_history = deque(maxlen=100)
alerts = []

st.title("🤖 Bundeskanzler KI - Enhanced Web GUI")

def get_system_metrics():
    """Hole aktuelle System-Metriken"""
    try:
        # Hier würden wir normalerweise die API aufrufen
        # Für Demo-Zwecke generieren wir Beispieldaten
        import psutil
        import os

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "timestamp": datetime.now(),
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_usage": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
            "active_processes": len(psutil.pids())
        }
    except:
        # Fallback für Systeme ohne psutil
        return {
            "timestamp": datetime.now(),
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "memory_used_gb": 8.5,
            "memory_total_gb": 16.0,
            "disk_usage": 34.1,
            "disk_used_gb": 120.5,
            "disk_total_gb": 500.0,
            "active_processes": 245
        }

def check_alerts(metrics):
    """Prüfe auf kritische Werte und erstelle Alerts"""
    new_alerts = []

    if metrics["cpu_usage"] > 90:
        new_alerts.append({
            "level": "critical",
            "message": f"🚨 CPU-Auslastung kritisch: {metrics['cpu_usage']:.1f}%",
            "timestamp": metrics["timestamp"]
        })

    if metrics["memory_usage"] > 90:
        new_alerts.append({
            "level": "critical",
            "message": f"🚨 Speicher-Auslastung kritisch: {metrics['memory_usage']:.1f}%",
            "timestamp": metrics["timestamp"]
        })

    if metrics["disk_usage"] > 95:
        new_alerts.append({
            "level": "warning",
            "message": f"⚠️ Festplatte fast voll: {metrics['disk_usage']:.1f}%",
            "timestamp": metrics["timestamp"]
        })

    return new_alerts

def create_metrics_chart(data, title, ylabel):
    """Erstelle ein einfaches Liniendiagramm mit matplotlib"""
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    timestamps = [d["timestamp"] for d in data]
    values = [d["value"] for d in data]

    ax.plot(timestamps, values, 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # Setze die x-Achse auf die letzten 10 Minuten
    if timestamps:
        ax.set_xlim(timestamps[0], timestamps[-1])

    plt.tight_layout()
    return fig

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
        st.subheader("🚀 Enhanced System Dashboard")

        # Auto-Refresh Toggle
        col_refresh, col_alerts, col_export = st.columns([1, 2, 1])
        with col_refresh:
            auto_refresh = st.checkbox("🔄 Auto-Refresh (5s)", value=True)
        with col_alerts:
            show_alerts = st.checkbox("🚨 Show Alerts", value=True)
        with col_export:
            if st.button("📊 Export Data"):
                st.info("Export feature coming soon!")

        # Live Metrics Collection
        if auto_refresh:
            # Sammle Metriken für Charts
            current_metrics = get_system_metrics()
            metrics_history.append({
                "timestamp": current_metrics["timestamp"],
                "cpu": current_metrics["cpu_usage"],
                "memory": current_metrics["memory_usage"],
                "disk": current_metrics["disk_usage"]
            })

            # Prüfe auf neue Alerts
            new_alerts = check_alerts(current_metrics)
            alerts.extend(new_alerts)

            # Behalte nur die letzten 10 Alerts
            alerts[:] = alerts[-10:]

        # === ALERTS SECTION ===
        if show_alerts and alerts:
            st.markdown("---")
            st.subheader("🚨 System Alerts")

            for alert in reversed(alerts[-5:]):  # Zeige die letzten 5 Alerts
                if alert["level"] == "critical":
                    st.error(f"{alert['message']} - {alert['timestamp'].strftime('%H:%M:%S')}")
                elif alert["level"] == "warning":
                    st.warning(f"{alert['message']} - {alert['timestamp'].strftime('%H:%M:%S')}")
                else:
                    st.info(f"{alert['message']} - {alert['timestamp'].strftime('%H:%M:%S')}")

        # === REAL-TIME METRICS ===
        st.markdown("---")
        st.subheader("📊 Real-Time System Metrics")

        # Aktuelle Metriken abrufen
        metrics = get_system_metrics()

        # Metriken in Spalten anzeigen
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cpu_color = "🟢" if metrics["cpu_usage"] < 70 else "🟡" if metrics["cpu_usage"] < 90 else "🔴"
            st.metric("CPU Usage", f"{cpu_color} {metrics['cpu_usage']:.1f}%")

        with col2:
            mem_color = "🟢" if metrics["memory_usage"] < 70 else "🟡" if metrics["memory_usage"] < 90 else "🔴"
            st.metric("Memory Usage", f"{mem_color} {metrics['memory_usage']:.1f}%",
                     f"{metrics['memory_used_gb']:.1f}GB / {metrics['memory_total_gb']:.1f}GB")

        with col3:
            disk_color = "🟢" if metrics["disk_usage"] < 80 else "🟡" if metrics["disk_usage"] < 95 else "🔴"
            st.metric("Disk Usage", f"{disk_color} {metrics['disk_usage']:.1f}%",
                     f"{metrics['disk_used_gb']:.1f}GB / {metrics['disk_total_gb']:.1f}GB")

        with col4:
            st.metric("Active Processes", f"🔄 {metrics['active_processes']}")

        # === PERFORMANCE CHARTS ===
        st.markdown("---")
        st.subheader("📈 Performance History (Live Charts)")

        if len(metrics_history) > 1:
            # CPU Chart
            st.subheader("CPU Usage Over Time")
            cpu_data = [{"timestamp": m["timestamp"], "value": m["cpu"]} for m in metrics_history]
            cpu_fig = create_metrics_chart(cpu_data, "CPU Usage History", "CPU %")
            if cpu_fig:
                st.pyplot(cpu_fig)

            # Memory Chart
            st.subheader("Memory Usage Over Time")
            mem_data = [{"timestamp": m["timestamp"], "value": m["memory"]} for m in metrics_history]
            mem_fig = create_metrics_chart(mem_data, "Memory Usage History", "Memory %")
            if mem_fig:
                st.pyplot(mem_fig)

            # Disk Chart
            st.subheader("Disk Usage Over Time")
            disk_data = [{"timestamp": m["timestamp"], "value": m["disk"]} for m in metrics_history]
            disk_fig = create_metrics_chart(disk_data, "Disk Usage History", "Disk %")
            if disk_fig:
                st.pyplot(disk_fig)
        else:
            st.info("📊 Sammle Daten für Charts... (5-10 Sekunden warten)")

        # === API METRICS ===
        st.markdown("---")
        st.subheader("🔗 API Performance")

        try:
            # API System Stats
            resp = requests.get(f"{API_URL}/admin/system-stats", headers=admin_headers, timeout=10)
            if resp.status_code == 200:
                api_stats = resp.json()

                api_col1, api_col2, api_col3, api_col4 = st.columns(4)

                with api_col1:
                    st.metric("API Requests (24h)", f"📈 {api_stats.get('api_requests_24h', 0)}")

                with api_col2:
                    st.metric("Active Users", f"👥 {api_stats.get('active_users', 0)}")

                with api_col3:
                    st.metric("Memory Entries", f"🧠 {api_stats.get('memory_entries', 0)}")

                with api_col4:
                    error_rate = api_stats.get('error_rate', 0)
                    error_color = "🟢" if error_rate < 1 else "🟡" if error_rate < 5 else "🔴"
                    st.metric("Error Rate", f"{error_color} {error_rate:.1f}%")

        except Exception as e:
            st.warning(f"API-Metriken nicht verfügbar: {e}")

        # === SYSTEM HEALTH ===
        st.markdown("---")
        st.subheader("🏥 System Health Check")

        try:
            resp_health = requests.get(f"{API_URL}/admin/health", headers=admin_headers, timeout=10)
            if resp_health.status_code == 200:
                health = resp_health.json()

                health_col1, health_col2 = st.columns(2)

                with health_col1:
                    if health["system"]["components_initialized"]:
                        st.success("✅ Alle Komponenten initialisiert")
                    else:
                        st.error("❌ Komponenten nicht vollständig initialisiert")

                    if health["files"]["logs_accessible"]:
                        st.success("✅ Log-Dateien zugänglich")
                    else:
                        st.warning("⚠️ Log-Dateien nicht alle zugänglich")

                with health_col2:
                    uptime_seconds = health['system']['uptime']
                    uptime_str = f"{uptime_seconds/3600:.1f}h" if uptime_seconds > 3600 else f"{uptime_seconds/60:.1f}min"
                    st.info(f"� Uptime: {uptime_str}")

                    request_count = health['system']['request_count']
                    st.info(f"📈 Total Requests: {request_count:,}")

        except Exception as e:
            st.error(f"Health Check fehlgeschlagen: {e}")

        # Auto-Refresh Logik
        if auto_refresh:
            time.sleep(5)  # 5 Sekunden warten
            st.rerun()  # Seite neu laden
    
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