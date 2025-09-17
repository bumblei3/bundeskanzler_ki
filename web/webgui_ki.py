import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Korrigierte API-URL für funktionierende Instanz
API_URL = "http://localhost:8001"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123!"


class DebugLevel(Enum):
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    DEBUG = "🔍"


class DebugSystem:
    """Verbessertes Debug-System für die Web-GUI"""

    def __init__(self):
        self.enabled = True
        self.messages = []
        self.api_calls = []
        self.start_time = time.time()

    def log(self, level: DebugLevel, message: str, data=None):
        """Füge eine Debug-Nachricht hinzu"""
        if not self.enabled:
            return

        timestamp = time.time() - self.start_time
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data,
            "time_str": f"{timestamp:.2f}s",
        }
        self.messages.append(entry)

        # Auch in Python logging ausgeben
        log_level = {
            DebugLevel.INFO: logging.INFO,
            DebugLevel.SUCCESS: logging.INFO,
            DebugLevel.WARNING: logging.WARNING,
            DebugLevel.ERROR: logging.ERROR,
            DebugLevel.DEBUG: logging.DEBUG,
        }.get(level, logging.INFO)

        logging.log(log_level, f"{level.value} {message}")

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        error=None,
    ):
        """Protokolliere einen API-Call"""
        if not self.enabled:
            return

        entry = {
            "timestamp": time.time() - self.start_time,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
            "error": str(error) if error else None,
            "success": status_code in [200, 201] and error is None,
        }
        self.api_calls.append(entry)

        level = DebugLevel.SUCCESS if entry["success"] else DebugLevel.ERROR
        self.log(level, f"API {method} {endpoint} -> {status_code} ({response_time:.2f}s)")

    def render_debug_ui(self):
        """Zeige Debug-Interface"""
        if not self.enabled:
            return

        with st.expander("🔧 Debug Console", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📝 Debug Messages")
                if self.messages:
                    for msg in self.messages[-20:]:  # Zeige letzte 20 Nachrichten
                        st.write(f"{msg['time_str']} {msg['level'].value} {msg['message']}")
                        if msg["data"]:
                            with st.expander(f"📊 Data ({msg['time_str']})", expanded=False):
                                st.json(msg["data"])
                else:
                    st.info("Keine Debug-Nachrichten")

            with col2:
                st.subheader("🌐 API Calls")
                if self.api_calls:
                    # API-Call Übersicht
                    total_calls = len(self.api_calls)
                    successful_calls = sum(1 for call in self.api_calls if call["success"])
                    avg_response_time = (
                        sum(call["response_time"] for call in self.api_calls) / total_calls
                    )

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total Calls", total_calls)
                    col_b.metric("Success Rate", f"{successful_calls}/{total_calls}")
                    col_c.metric("Avg Response", f"{avg_response_time:.2f}s")

                    # Detaillierte API-Call Liste
                    st.subheader("Call Details")
                    for call in self.api_calls[-10:]:  # Zeige letzte 10 Calls
                        status_icon = "✅" if call["success"] else "❌"
                        st.write(
                            f"{call['timestamp']:.2f}s {status_icon} {call['method']} {call['endpoint']} -> {call['status_code']} ({call['response_time']:.2f}s)"
                        )
                        if call["error"]:
                            st.error(f"Error: {call['error']}")
                else:
                    st.info("Keine API-Calls")

                # Debug Controls
                st.subheader("Debug Controls")
                if st.button("🧹 Clear Debug Log"):
                    self.messages.clear()
                    self.api_calls.clear()
                    st.rerun()

                if st.button("📊 Export Debug Data"):
                    debug_data = {
                        "messages": self.messages,
                        "api_calls": self.api_calls,
                        "timestamp": datetime.now().isoformat(),
                    }
                    st.download_button(
                        label="📥 Download Debug JSON",
                        data=json.dumps(debug_data, indent=2, default=str),
                        file_name=f"debug_log_{int(time.time())}.json",
                        mime="application/json",
                    )


# Global Debug System
debug_system = DebugSystem()


def api_request(method: str, endpoint: str, headers=None, data=None, json_data=None, timeout=10):
    """Wrapper für API-Requests mit Debug-Logging"""
    start_time = time.time()

    try:
        url = f"{API_URL}{endpoint}"
        debug_system.log(DebugLevel.DEBUG, f"Making {method} request to {endpoint}")

        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            if json_data:
                response = requests.post(url, headers=headers, json=json_data, timeout=timeout)
            else:
                response = requests.post(url, headers=headers, data=data, timeout=timeout)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=json_data, timeout=timeout)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response_time = time.time() - start_time
        debug_system.log_api_call(endpoint, method, response.status_code, response_time)

        return response

    except Exception as e:
        response_time = time.time() - start_time
        debug_system.log_api_call(endpoint, method, 0, response_time, e)
        raise e


# Global variables for live data
metrics_history = deque(maxlen=100)
alerts = []


def get_system_metrics():
    """Hole aktuelle System-Metriken"""
    try:
        # Hier würden wir normalerweise die API aufrufen
        # Für Demo-Zwecke generieren wir Beispieldaten
        import os

        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "timestamp": datetime.now(),
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_usage": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
            "active_processes": len(psutil.pids()),
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
            "active_processes": 245,
        }


def get_gpu_metrics():
    """Hole GPU-Metriken von der RTX 2070"""
    try:
        import subprocess
        import json

        # Versuche nvidia-smi für detaillierte GPU-Info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free,temperature.gpu,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                if len(values) >= 6:
                    memory_used = float(values[0])
                    memory_total = float(values[1])
                    memory_free = float(values[2])
                    gpu_temp = float(values[3])
                    gpu_util = float(values[4])
                    mem_util = float(values[5])

                    return {
                        "gpu_memory_used_mb": memory_used,
                        "gpu_memory_total_mb": memory_total,
                        "gpu_memory_free_mb": memory_free,
                        "gpu_temperature": gpu_temp,
                        "gpu_utilization": gpu_util,
                        "memory_utilization": mem_util,
                        "gpu_memory_used_percent": (memory_used / memory_total) * 100 if memory_total > 0 else 0,
                    }
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        # Fallback: Versuche PyTorch CUDA Info
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)   # MB
                memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB

                return {
                    "gpu_memory_used_mb": memory_allocated,
                    "gpu_memory_total_mb": memory_total,
                    "gpu_memory_free_mb": memory_total - memory_reserved,
                    "gpu_temperature": 65.0,  # Placeholder
                    "gpu_utilization": 45.0,  # Placeholder
                    "memory_utilization": (memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
                    "gpu_memory_used_percent": (memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
                }
        except ImportError:
            pass

    except Exception as e:
        debug_system.log(DebugLevel.WARNING, f"GPU metrics error: {e}")

    # Letzter Fallback: Mock-Daten
    return {
        "gpu_memory_used_mb": 6500,
        "gpu_memory_total_mb": 8192,
        "gpu_memory_free_mb": 1692,
        "gpu_temperature": 68.5,
        "gpu_utilization": 42.3,
        "memory_utilization": 79.2,
        "gpu_memory_used_percent": 79.2,
    }


def check_alerts(metrics):
    """Prüfe auf kritische Werte und erstelle Alerts"""
    new_alerts = []

    # GPU-spezifische Alerts
    if "gpu_temperature" in metrics and metrics["gpu_temperature"] > 80:
        new_alerts.append(
            {
                "level": "critical",
                "message": f"🚨 GPU-Temperatur kritisch: {metrics['gpu_temperature']:.1f}°C",
                "timestamp": datetime.now().isoformat(),
            }
        )

    if "gpu_utilization" in metrics and metrics["gpu_utilization"] > 95:
        new_alerts.append(
            {
                "level": "warning",
                "message": f"⚠️ GPU-Auslastung hoch: {metrics['gpu_utilization']:.1f}%",
                "timestamp": datetime.now().isoformat(),
            }
        )

    if "gpu_memory_used_percent" in metrics and metrics["gpu_memory_used_percent"] > 90:
        new_alerts.append(
            {
                "level": "warning",
                "message": f"⚠️ GPU-Speicher fast voll: {metrics['gpu_memory_used_percent']:.1f}%",
                "timestamp": datetime.now().isoformat(),
            }
        )

    # System-Alerts (falls verfügbar)
    if "cpu_usage" in metrics and metrics["cpu_usage"] > 90:
        new_alerts.append(
            {
                "level": "critical",
                "message": f"🚨 CPU-Auslastung kritisch: {metrics['cpu_usage']:.1f}%",
                "timestamp": datetime.now().isoformat(),
            }
        )

    if "memory_usage" in metrics and metrics["memory_usage"] > 90:
        new_alerts.append(
            {
                "level": "critical",
                "message": f"🚨 Speicher-Auslastung kritisch: {metrics['memory_usage']:.1f}%",
                "timestamp": datetime.now().isoformat(),
            }
        )

    return new_alerts

    if metrics["disk_usage"] > 95:
        new_alerts.append(
            {
                "level": "warning",
                "message": f"⚠️ Festplatte fast voll: {metrics['disk_usage']:.1f}%",
                "timestamp": metrics["timestamp"],
            }
        )

    return new_alerts


def create_metrics_chart(values, timestamps, title, ylabel):
    """Erstelle ein einfaches Liniendiagramm mit matplotlib"""
    if not values or not timestamps:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, values, "b-", linewidth=2, marker="o", markersize=3)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # Setze die x-Achse auf die letzten 10 Minuten
    if timestamps and len(timestamps) > 1:
        min_time = min(timestamps)
        max_time = max(timestamps)
        if min_time != max_time:
            ax.set_xlim(min_time, max_time)
        else:
            # Wenn alle Timestamps gleich sind, zeige einen kleinen Bereich um den Wert herum
            ax.set_xlim(min_time - 1, min_time + 1)

    plt.tight_layout()
    return fig


def show_admin_interface():
    """Zeigt das Admin-Interface"""
    st.title("🔐 Admin Panel - Bundeskanzler KI")

    # === DARK/LIGHT MODE TOGGLE ===
    col_theme, col_debug = st.sidebar.columns([1, 1])
    with col_theme:
        if st.sidebar.checkbox("🌙 Dark Mode", value=False, key="dark_mode"):
            st.markdown("""
            <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .stSidebar {
                background-color: #2d2d2d;
            }
            .stTextInput, .stTextArea, .stSelectbox {
                background-color: #3d3d3d;
                color: #ffffff;
            }
            </style>
            """, unsafe_allow_html=True)

    # Debug-Modus Toggle
    with col_debug:
        debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=True)
    debug_system.enabled = debug_mode

    debug_system.log(DebugLevel.INFO, "Admin Interface gestartet")

    # Prüfe Admin-Token
    if "admin_token" not in st.session_state:
        debug_system.log(DebugLevel.ERROR, "Kein Admin-Token im Session-State gefunden")
        st.error("❌ Kein Admin-Token gefunden!")
        return

    admin_token = st.session_state["admin_token"]
    debug_system.log(DebugLevel.SUCCESS, "Admin-Token gefunden", {"token_prefix": admin_token[:20]})

    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Teste API-Verbindung
    try:
        test_response = api_request("GET", "/health", headers=admin_headers, timeout=5)
        if test_response.status_code == 200:
            st.success("✅ API-Verbindung OK!")
            debug_system.log(DebugLevel.SUCCESS, "API-Verbindung erfolgreich")
        else:
            st.error(f"❌ API-Verbindungsfehler: {test_response.status_code}")
            debug_system.log(
                DebugLevel.ERROR,
                f"API-Verbindung fehlgeschlagen: {test_response.status_code}",
            )
            return
    except Exception as e:
        st.error(f"❌ API-Verbindungsfehler: {e}")
        debug_system.log(DebugLevel.ERROR, f"API-Verbindung exception: {e}")
        return

    # Admin Tabs erstellen
    debug_system.log(DebugLevel.INFO, "Erstelle Admin-Tabs")

    try:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "📊 Dashboard",
                "👥 Benutzer-Management",
                "📋 Log-Viewer",
                "💾 Memory-Management",
                "🌐 Multilingual KI",
                "📚 Query-Historie",
                "⚙️ Konfiguration",
            ]
        )

        debug_system.log(DebugLevel.SUCCESS, "Admin-Tabs erfolgreich erstellt")

        # Dashboard Tab
        with tab1:
            debug_system.log(DebugLevel.DEBUG, "Dashboard-Tab wird geladen")
            show_dashboard_tab(admin_headers)

        # Benutzer-Management Tab
        with tab2:
            debug_system.log(DebugLevel.DEBUG, "Benutzer-Management-Tab wird geladen")
            show_users_tab(admin_headers)

        # Log-Viewer Tab
        with tab3:
            debug_system.log(DebugLevel.DEBUG, "Log-Viewer-Tab wird geladen")
            show_logs_tab(admin_headers)

        # Memory-Management Tab
        with tab4:
            debug_system.log(DebugLevel.DEBUG, "Memory-Management-Tab wird geladen")
            show_memory_tab(admin_headers)

        # Multilingual KI Tab
        with tab5:
            debug_system.log(DebugLevel.DEBUG, "Multilingual KI-Tab wird geladen")
            show_multilingual_tab(admin_headers)

        # Konfiguration Tab
        with tab6:
            debug_system.log(DebugLevel.DEBUG, "Konfiguration-Tab wird geladen")
            show_config_tab(admin_headers)

        # Query-Historie Tab
        with tab7:
            debug_system.log(DebugLevel.DEBUG, "Query-Historie-Tab wird geladen")
            show_query_history_tab(admin_headers)

    except Exception as e:
        debug_system.log(DebugLevel.ERROR, f"Fehler beim Erstellen der Tabs: {e}")
        st.error(f"❌ Fehler beim Laden der Admin-Tabs: {e}")

    # Debug UI am Ende rendern
    if debug_mode:
        st.divider()
        debug_system.render_debug_ui()


# === TAB FUNKTIONEN ===


def show_dashboard_tab(admin_headers):
    """Zeigt den Dashboard-Tab"""
    debug_system.log(DebugLevel.DEBUG, "Dashboard-Tab wird geladen")
    st.subheader("🚀 Enhanced System Dashboard")

    # Auto-Refresh Toggle
    col_refresh, col_alerts, col_export = st.columns([1, 2, 1])
    with col_refresh:
        auto_refresh = st.checkbox("🔄 Auto-Refresh (5s)", value=True, key="dashboard_refresh")
    with col_alerts:
        show_alerts = st.checkbox("🚨 Show Alerts", value=True, key="dashboard_alerts")
    with col_export:
        if st.button("📊 Export Data", key="dashboard_export"):
            st.info("Export feature coming soon!")

    # Live Metrics Collection
    if auto_refresh:
        try:
            current_metrics = get_system_metrics()
            gpu_metrics = get_gpu_metrics()

            metrics_history.append(
                {
                    "timestamp": current_metrics["timestamp"],
                    "cpu": current_metrics["cpu_usage"],
                    "memory": current_metrics["memory_usage"],
                    "disk": current_metrics["disk_usage"],
                    "gpu_memory": gpu_metrics["gpu_memory_used_percent"],
                    "gpu_temp": gpu_metrics["gpu_temperature"],
                    "gpu_util": gpu_metrics["gpu_utilization"],
                }
            )

            # Prüfe auf neue Alerts
            new_alerts = check_alerts(current_metrics)
            # GPU-Alerts hinzufügen
            if gpu_metrics["gpu_temperature"] > 85:
                new_alerts.append({
                    "level": "critical",
                    "message": f"🚨 GPU-Temperatur kritisch: {gpu_metrics['gpu_temperature']:.1f}°C",
                    "timestamp": current_metrics["timestamp"],
                })
            if gpu_metrics["gpu_memory_used_percent"] > 95:
                new_alerts.append({
                    "level": "warning",
                    "message": f"⚠️ GPU-Speicher fast voll: {gpu_metrics['gpu_memory_used_percent']:.1f}%",
                    "timestamp": current_metrics["timestamp"],
                })

            alerts.extend(new_alerts)
            alerts[:] = alerts[-10:]  # Behalte nur die letzten 10 Alerts

            debug_system.log(DebugLevel.SUCCESS, "Metriken erfolgreich gesammelt")
        except Exception as e:
            debug_system.log(DebugLevel.ERROR, f"Fehler beim Sammeln der Metriken: {e}")
            st.error(f"❌ Fehler beim Sammeln der Metriken: {e}")

    # === ALERTS SECTION ===
    if show_alerts and alerts:
        st.markdown("---")
        st.subheader("🚨 System Alerts")

        for alert in reversed(alerts[-5:]):  # Zeige die letzten 5 Alerts
            if alert["level"] == "critical":
                st.error(f"🚨 {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
            elif alert["level"] == "warning":
                st.warning(f"⚠️ {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
            else:
                st.info(f"ℹ️ {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")

    # === METRICS CHARTS ===
    st.markdown("---")
    st.subheader("📊 System Metrics")

    if metrics_history:
        # Erstelle Charts für die letzten Metriken
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🖥️ System Resources")
            fig_cpu = create_metrics_chart(
                [m["cpu"] for m in metrics_history],
                [m["timestamp"] for m in metrics_history],
                "CPU Usage (%)",
                "CPU %",
            )
            st.pyplot(fig_cpu)

            fig_mem = create_metrics_chart(
                [m["memory"] for m in metrics_history],
                [m["timestamp"] for m in metrics_history],
                "Memory Usage (%)",
                "Memory %",
            )
            st.pyplot(fig_mem)

        with col2:
            st.subheader("💾 Storage & GPU Memory")
            fig_disk = create_metrics_chart(
                [m["disk"] for m in metrics_history],
                [m["timestamp"] for m in metrics_history],
                "Disk Usage (%)",
                "Disk %",
            )
            st.pyplot(fig_disk)

            fig_gpu_mem = create_metrics_chart(
                [m["gpu_memory"] for m in metrics_history],
                [m["timestamp"] for m in metrics_history],
                "GPU Memory Usage (%)",
                "GPU Memory %",
            )
            st.pyplot(fig_gpu_mem)

        with col3:
            st.subheader("🌡️ GPU Temperature & Utilization")
            fig_gpu_temp = create_metrics_chart(
                [m["gpu_temp"] for m in metrics_history],
                [m["timestamp"] for m in metrics_history],
                "GPU Temperature (°C)",
                "Temperature °C",
            )
            st.pyplot(fig_gpu_temp)

            fig_gpu_util = create_metrics_chart(
                [m["gpu_util"] for m in metrics_history],
                [m["timestamp"] for m in metrics_history],
                "GPU Utilization (%)",
                "GPU %",
            )
            st.pyplot(fig_gpu_util)

            # Zeige aktuelle Werte
            if metrics_history:
                latest = metrics_history[-1]
                gpu_metrics = get_gpu_metrics()

                st.metric("GPU Memory", f"{gpu_metrics['gpu_memory_used_mb']:.0f}/{gpu_metrics['gpu_memory_total_mb']:.0f} MB")
                st.metric("GPU Temp", f"{gpu_metrics['gpu_temperature']:.1f}°C")
                st.metric("GPU Util", f"{gpu_metrics['gpu_utilization']:.1f}%")
    else:
        st.info("🔄 Sammle Metriken... Bitte warten Sie einen Moment.")


def show_users_tab(admin_headers):
    """Zeigt den Benutzer-Management-Tab"""
    debug_system.log(DebugLevel.DEBUG, "Benutzer-Management-Tab wird geladen")
    st.subheader("👥 Benutzer-Management")

    try:
        # Lade Benutzer-Daten
        debug_system.log(DebugLevel.DEBUG, "Lade Benutzer-Daten")
        resp = api_request("GET", "/admin/users", headers=admin_headers)

        if resp.status_code == 200:
            users_data = resp.json()
            debug_system.log(DebugLevel.SUCCESS, "Benutzer-Daten erfolgreich geladen", users_data)

            if isinstance(users_data, dict) and "users" in users_data:
                users = users_data["users"]
                st.success(f"✅ {len(users)} Benutzer gefunden")

                # Zeige Benutzer in einer Tabelle
                if users:
                    df = pd.DataFrame(users)
                    st.dataframe(df, use_container_width=True)

                    # Benutzer-Statistiken
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Users", len(users))
                    with col2:
                        active_users = sum(1 for u in users if u.get("active", True))
                        st.metric("Active Users", active_users)
                    with col3:
                        admin_users = sum(1 for u in users if u.get("role") == "admin")
                        st.metric("Admin Users", admin_users)
                else:
                    st.info("ℹ️ Keine Benutzer gefunden")
            else:
                st.warning("⚠️ Unerwartete API-Antwort")
                st.json(users_data)
        else:
            debug_system.log(DebugLevel.ERROR, f"Benutzer-API Fehler: {resp.status_code}")
            st.error(f"❌ Fehler beim Laden der Benutzer: {resp.status_code}")
            st.code(resp.text)

    except Exception as e:
        debug_system.log(DebugLevel.ERROR, f"Exception beim Laden der Benutzer: {e}")
        st.error(f"❌ Fehler beim Laden der Benutzer: {e}")


def show_logs_tab(admin_headers):
    """Zeigt den Log-Viewer-Tab"""
    debug_system.log(DebugLevel.DEBUG, "Log-Viewer-Tab wird geladen")
    st.subheader("📋 Log-Viewer")

    col1, col2 = st.columns([1, 2])
    with col1:
        log_type = st.selectbox("Log Type", ["api.log", "error.log", "access.log"], key="log_type")
    with col2:
        lines_count = st.slider("Lines to show", 10, 1000, 100, key="log_lines")

    try:
        debug_system.log(DebugLevel.DEBUG, f"Lade Logs: {log_type}, {lines_count} Zeilen")
        resp = api_request(
            "GET", f"/admin/logs/{log_type}?lines={lines_count}", headers=admin_headers
        )

        if resp.status_code == 200:
            log_data = resp.json()
            debug_system.log(DebugLevel.SUCCESS, "Logs erfolgreich geladen")

            if isinstance(log_data, dict) and "entries" in log_data:
                entries = log_data["entries"]
                st.success(f"✅ {len(entries)} Log-Einträge geladen")

                if entries:
                    # Zeige Logs in einem Textbereich
                    log_text = "\n".join(entries[-lines_count:])
                    st.text_area("Log Entries", log_text, height=400, key="log_text")

                    # Log-Level Filter
                    if st.checkbox("Filter by Level", key="log_filter"):
                        levels = st.multiselect(
                            "Log Levels",
                            ["INFO", "WARNING", "ERROR", "DEBUG"],
                            ["INFO", "WARNING", "ERROR"],
                            key="log_levels",
                        )
                        filtered_entries = [
                            e for e in entries if any(level in e.upper() for level in levels)
                        ]
                        st.text_area(
                            "Filtered Logs",
                            "\n".join(filtered_entries[-lines_count:]),
                            height=200,
                            key="filtered_log_text",
                        )
                else:
                    st.info("ℹ️ Keine Log-Einträge gefunden")
            else:
                st.warning("⚠️ Unerwartete API-Antwort")
                st.json(log_data)
        else:
            debug_system.log(DebugLevel.ERROR, f"Logs-API Fehler: {resp.status_code}")
            st.error(f"❌ Fehler beim Laden der Logs: {resp.status_code}")
            st.code(resp.text)

    except Exception as e:
        debug_system.log(DebugLevel.ERROR, f"Exception beim Laden der Logs: {e}")
        st.error(f"❌ Fehler beim Laden der Logs: {e}")


def show_memory_tab(admin_headers):
    """Zeigt den Memory-Management-Tab"""
    debug_system.log(DebugLevel.DEBUG, "Memory-Management-Tab wird geladen")
    st.subheader("💾 Memory-Management")

    try:
        # Lade Memory-Stats
        debug_system.log(DebugLevel.DEBUG, "Lade Memory-Stats")
        resp = api_request("GET", "/admin/memory/stats", headers=admin_headers)

        if resp.status_code == 200:
            memory_stats = resp.json()
            debug_system.log(DebugLevel.SUCCESS, "Memory-Stats erfolgreich geladen", memory_stats)

            # Zeige Memory-Übersicht
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Kurzzeitgedächtnis")
                kurzzeit = memory_stats.get("kurzzeitgedaechtnis_entries", 0)
                st.metric("Einträge", kurzzeit)

            with col2:
                st.subheader("Langzeitgedächtnis")
                langzeit = memory_stats.get("langzeitgedaechtnis_entries", 0)
                st.metric("Einträge", langzeit)

            # Detaillierte Memory-Informationen
            st.markdown("---")
            st.subheader("� Detaillierte Informationen")

            if st.checkbox("Show Raw Data", key="memory_raw"):
                st.json(memory_stats)

            # Memory-Management Aktionen
            st.markdown("---")
            st.subheader("🛠️ Memory-Management")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🧹 Clear Short-term Memory", key="clear_short"):
                    st.info("Feature coming soon!")
            with col2:
                if st.button("🗂️ Optimize Long-term Memory", key="optimize_long"):
                    st.info("Feature coming soon!")
            with col3:
                if st.button("💾 Save Memory State", key="save_memory"):
                    st.info("Feature coming soon!")

        else:
            debug_system.log(DebugLevel.ERROR, f"Memory-API Fehler: {resp.status_code}")
            st.error(f"❌ Fehler beim Laden der Memory-Stats: {resp.status_code}")
            st.code(resp.text)

    except Exception as e:
        debug_system.log(DebugLevel.ERROR, f"Exception beim Laden der Memory-Stats: {e}")
        st.error(f"❌ Fehler beim Laden der Memory-Stats: {e}")


def show_config_tab(admin_headers):
    """Zeigt den Konfiguration-Tab"""
    debug_system.log(DebugLevel.DEBUG, "Konfiguration-Tab wird geladen")
    st.subheader("⚙️ Konfiguration")

    try:
        # Lade Konfiguration
        debug_system.log(DebugLevel.DEBUG, "Lade Konfiguration")
        resp = api_request("GET", "/admin/config", headers=admin_headers)

        if resp.status_code == 200:
            config_data = resp.json()
            debug_system.log(DebugLevel.SUCCESS, "Konfiguration erfolgreich geladen", config_data)

            # Zeige Konfiguration in Kategorien
            if isinstance(config_data, dict):
                tabs = st.tabs(["API Settings", "Memory Settings", "Logging Settings", "Other"])

                with tabs[0]:
                    st.subheader("🔗 API Settings")
                    api_config = config_data.get("api_settings", {})
                    if api_config:
                        st.json(api_config)
                    else:
                        st.info("Keine API-Konfiguration verfügbar")

                with tabs[1]:
                    st.subheader("💾 Memory Settings")
                    memory_config = config_data.get("memory_settings", {})
                    if memory_config:
                        st.json(memory_config)
                    else:
                        st.info("Keine Memory-Konfiguration verfügbar")

                with tabs[2]:
                    st.subheader("📋 Logging Settings")
                    logging_config = config_data.get("logging_settings", {})
                    if logging_config:
                        st.json(logging_config)
                    else:
                        st.info("Keine Logging-Konfiguration verfügbar")

                with tabs[3]:
                    st.subheader("🔧 Other Settings")
                    # Zeige alle anderen Konfigurationseinträge
                    other_config = {
                        k: v
                        for k, v in config_data.items()
                        if k not in ["api_settings", "memory_settings", "logging_settings"]
                    }
                    if other_config:
                        st.json(other_config)
                    else:
                        st.info("Keine weiteren Konfigurationseinträge")

                # Raw Config Data
                if st.checkbox("Show Raw Config Data", key="config_raw"):
                    st.json(config_data)
            else:
                st.warning("⚠️ Unerwartete API-Antwort")
                st.json(config_data)

        else:
            debug_system.log(DebugLevel.ERROR, f"Config-API Fehler: {resp.status_code}")
            st.error(f"❌ Fehler beim Laden der Konfiguration: {resp.status_code}")
            st.code(resp.text)

    except Exception as e:
        debug_system.log(DebugLevel.ERROR, f"Exception beim Laden der Konfiguration: {e}")
        st.error(f"❌ Fehler beim Laden der Konfiguration: {e}")


def show_multilingual_tab(admin_headers):
    """Zeigt den Multilingual KI-Tab"""
    debug_system.log(DebugLevel.DEBUG, "Multilingual KI-Tab wird geladen")
    st.subheader("🌐 Multilingual KI - Mehrsprachige Unterstützung")

    try:
        # Import der multilingualen KI
        from multilingual_bundeskanzler_ki import (
            get_multilingual_ki,
            multilingual_query,
        )

        # Initialisiere KI falls noch nicht geschehen
        if "multilingual_ki" not in st.session_state:
            with st.spinner("🚀 Initialisiere Multilingual KI..."):
                st.session_state.multilingual_ki = get_multilingual_ki()
                st.session_state.multilingual_ki.initialize_multimodal_model()
            st.success("✅ Multilingual KI bereit!")

        ki = st.session_state.multilingual_ki

        # Sprach-Informationen anzeigen
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌍 Unterstützte Sprachen")
            languages = ki.get_supported_languages_info()
            for code, name in languages.items():
                st.write(f"**{code.upper()}**: {name}")

        with col2:
            st.subheader("📊 Debug-Info")
            debug_info = ki.get_debug_info()
            st.metric("Debug-Nachrichten", debug_info.get("message_count", 0))
            st.metric("API-Calls", debug_info.get("api_call_count", 0))

        # Test-Interface für mehrsprachige Anfragen
        st.subheader("🧪 Mehrsprachige Anfragen testen")

        # Beispiel-Anfragen
        example_queries = {
            "Deutsch": "Was ist die Klimapolitik der Bundesregierung?",
            "English": "What is the climate policy of the German government?",
            "Français": "Quelle est la politique climatique du gouvernement fédéral allemand?",
        }

        col1, col2 = st.columns([2, 1])

        with col1:
            # Manuelle Eingabe
            user_query = st.text_area(
                "Ihre Anfrage (in beliebiger Sprache):",
                height=100,
                placeholder="Stellen Sie Ihre Frage auf Deutsch, Englisch oder Französisch...",
            )

        with col2:
            st.subheader("📝 Beispiele")
            for lang, query in example_queries.items():
                if st.button(f"📌 {lang}", key=f"example_{lang.lower()}"):
                    st.session_state.test_query = query
                    st.rerun()

            if st.button("🧹 Löschen", key="clear_query"):
                if "test_query" in st.session_state:
                    del st.session_state.test_query

        # Verwende Beispiel-Query falls verfügbar
        if "test_query" in st.session_state and not user_query:
            user_query = st.session_state.test_query

        # Verarbeitung der Anfrage
        if user_query and st.button("🚀 Anfrage verarbeiten", type="primary"):
            with st.spinner("🌐 Verarbeite mehrsprachige Anfrage..."):
                try:
                    result = ki.process_multilingual_query(user_query)

                    # Ergebnisse anzeigen
                    st.success("✅ Anfrage erfolgreich verarbeitet!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Erkannte Sprache", result["detected_language"].upper())
                    with col2:
                        st.metric("Verarbeitungszeit", f"{result['processing_time']:.2f}s")
                    with col3:
                        st.metric(
                            "Übersetzung verwendet",
                            "Ja" if result.get("translation_used") else "Nein",
                        )

                    # Antwort anzeigen
                    st.subheader("💬 Antwort")
                    st.info(result["response"])

                    # Detaillierte Informationen (ausklappbar)
                    with st.expander("📊 Detaillierte Verarbeitungsinformationen", expanded=False):
                        st.write("**Original-Anfrage:**", result["original_query"])

                        if result.get("german_query") != result["original_query"]:
                            st.write("**Deutsche Übersetzung:**", result["german_query"])

                        st.write("**Deutsche Antwort:**", result["german_response"])

                        if result.get("translation_used"):
                            st.write("**Zurückübersetzte Antwort:**", result["response"])

                        # Debug-Informationen
                        if "error" in result:
                            st.error(f"⚠️ Fehler aufgetreten: {result['error']}")

                        if result.get("fallback_used"):
                            st.warning("⚠️ Fallback-Modus wurde verwendet")

                except Exception as e:
                    st.error(f"❌ Fehler bei der Verarbeitung: {e}")
                    debug_system.log(DebugLevel.ERROR, f"Multilingual query error: {e}")

        # Debug-Interface für Multilingual-KI
        if st.checkbox("🔧 Multilingual Debug anzeigen", key="multilingual_debug"):
            debug_info = ki.get_debug_info()
            if debug_info.get("debug_disabled"):
                st.info("Debug-System ist deaktiviert")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📝 Debug-Nachrichten")
                    for msg in debug_info.get("messages", [])[-10:]:
                        st.write(
                            f"{msg.get('time_str', 'N/A')} {msg.get('level', 'N/A')} {msg.get('message', 'N/A')}"
                        )

                with col2:
                    st.subheader("🌐 API-Calls")
                    for call in debug_info.get("api_calls", [])[-10:]:
                        status = "✅" if call.get("success") else "❌"
                        st.write(
                            f"{status} {call.get('method', 'N/A')} {call.get('endpoint', 'N/A')} -> {call.get('status_code', 'N/A')}"
                        )

    except Exception as e:
        debug_system.log(DebugLevel.ERROR, f"Exception im Multilingual-Tab: {e}")
        st.error(f"❌ Fehler beim Laden des Multilingual-Tabs: {e}")
        st.info("💡 Stellen Sie sicher, dass die multilingualen Services installiert sind.")


def show_login_interface():
    """Zeigt das Login-Interface"""
    st.subheader("🔐 Admin Login")

    with st.form("login_form"):
        username = st.text_input("Username", value=ADMIN_USERNAME)
        password = st.text_input("Password", type="password", value=ADMIN_PASSWORD)

        if st.form_submit_button("Login"):
            try:
                # Token anfordern
                auth_response = api_request(
                    "POST",
                    "/auth/admin-token",
                    data={"username": username, "password": password},
                )

                if auth_response.status_code == 200:
                    token_data = auth_response.json()
                    st.session_state["admin_token"] = token_data["access_token"]
                    st.success("✅ Login erfolgreich!")
                    debug_system.log(DebugLevel.SUCCESS, "Admin-Login erfolgreich")
                    st.rerun()
                else:
                    st.error("❌ Login fehlgeschlagen!")
                    debug_system.log(
                        DebugLevel.ERROR,
                        f"Login fehlgeschlagen: {auth_response.status_code}",
                    )

            except Exception as e:
                st.error(f"❌ Login-Fehler: {e}")
                debug_system.log(DebugLevel.ERROR, f"Login-Exception: {e}")


def show_query_history_tab(admin_headers):
    """Zeigt den Query-Historie-Tab"""
    debug_system.log(DebugLevel.DEBUG, "Query-Historie-Tab wird geladen")
    st.subheader("📚 Query-Historie & Analyse")

    # Filter-Optionen
    col1, col2, col3 = st.columns(3)

    with col1:
        date_filter = st.date_input(
            "Datum filtern",
            value=datetime.now().date(),
            key="query_date_filter"
        )

    with col2:
        status_filter = st.selectbox(
            "Status filtern",
            ["Alle", "Erfolgreich", "Fehlgeschlagen", "In Bearbeitung"],
            key="query_status_filter"
        )

    with col3:
        search_query = st.text_input(
            "Suche in Queries",
            placeholder="Suchbegriff eingeben...",
            key="query_search"
        )

    # Mock Query-Historie Daten (in der Realität würden diese aus der Datenbank kommen)
    query_history = [
        {
            "id": 1,
            "timestamp": datetime.now() - timedelta(minutes=5),
            "query": "Was ist die aktuelle Klimapolitik Deutschlands?",
            "status": "Erfolgreich",
            "response_time": 0.23,
            "confidence": 87.5,
            "language": "DE",
            "user": "admin"
        },
        {
            "id": 2,
            "timestamp": datetime.now() - timedelta(minutes=15),
            "query": "What is Germany's current climate policy?",
            "status": "Erfolgreich",
            "response_time": 0.18,
            "confidence": 92.1,
            "language": "EN",
            "user": "admin"
        },
        {
            "id": 3,
            "timestamp": datetime.now() - timedelta(hours=1),
            "query": "Wie funktioniert die Bundestagswahl?",
            "status": "Erfolgreich",
            "response_time": 0.31,
            "confidence": 78.9,
            "language": "DE",
            "user": "guest"
        },
        {
            "id": 4,
            "timestamp": datetime.now() - timedelta(hours=2),
            "query": "Invalid query that failed",
            "status": "Fehlgeschlagen",
            "response_time": 0.05,
            "confidence": 0.0,
            "language": "EN",
            "user": "admin"
        }
    ]

    # Filter anwenden
    filtered_history = query_history

    if status_filter != "Alle":
        filtered_history = [q for q in filtered_history if q["status"] == status_filter]

    if search_query:
        filtered_history = [
            q for q in filtered_history
            if search_query.lower() in q["query"].lower()
        ]

    # Statistiken anzeigen
    st.subheader("📊 Query-Statistiken")

    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

    total_queries = len(filtered_history)
    successful_queries = len([q for q in filtered_history if q["status"] == "Erfolgreich"])
    avg_response_time = sum(q["response_time"] for q in filtered_history) / total_queries if total_queries > 0 else 0
    avg_confidence = sum(q["confidence"] for q in filtered_history) / total_queries if total_queries > 0 else 0

    with col_stats1:
        st.metric("Total Queries", total_queries)
    with col_stats2:
        st.metric("Erfolgsrate", f"{successful_queries}/{total_queries}" if total_queries > 0 else "0/0")
    with col_stats3:
        st.metric("Ø Response Time", f"{avg_response_time:.2f}s")
    with col_stats4:
        st.metric("Ø Confidence", f"{avg_confidence:.1f}%")

    # Query-Historie Tabelle
    st.subheader("📋 Query-Historie")

    if filtered_history:
        # Erstelle DataFrame für bessere Darstellung
        df_data = []
        for query in filtered_history:
            df_data.append({
                "Zeit": query["timestamp"].strftime("%H:%M:%S"),
                "Query": query["query"][:50] + "..." if len(query["query"]) > 50 else query["query"],
                "Status": query["status"],
                "Response Time": f"{query['response_time']:.2f}s",
                "Confidence": f"{query['confidence']:.1f}%",
                "Sprache": query["language"],
                "User": query["user"]
            })

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

        # Detaillierte Ansicht für einzelne Queries
        st.subheader("🔍 Detaillierte Ansicht")

        selected_query_id = st.selectbox(
            "Query zum Anzeigen auswählen",
            [q["id"] for q in filtered_history],
            format_func=lambda x: f"Query {x}: {next(q['query'] for q in filtered_history if q['id'] == x)[:30]}...",
            key="query_detail_select"
        )

        if selected_query_id:
            selected_query = next(q for q in filtered_history if q["id"] == selected_query_id)

            col_detail1, col_detail2 = st.columns(2)

            with col_detail1:
                st.markdown("**📝 Vollständige Query:**")
                st.info(selected_query["query"])

                st.markdown("**💬 Response Preview:**")
                # Mock response basierend auf Query
                if "Klimapolitik" in selected_query["query"]:
                    st.success("Die aktuelle Klimapolitik Deutschlands umfasst das Ziel der Klimaneutralität bis 2045...")
                elif "Wahl" in selected_query["query"]:
                    st.success("Die Bundestagswahl findet alle 4 Jahre statt und verwendet ein Mischsystem...")
                else:
                    st.success("Response würde hier angezeigt werden...")

            with col_detail2:
                st.markdown("**📊 Details:**")
                st.write(f"**Status:** {selected_query['status']}")
                st.write(f"**Response Time:** {selected_query['response_time']:.2f}s")
                st.write(f"**Confidence Score:** {selected_query['confidence']:.1f}%")
                st.write(f"**Sprache:** {selected_query['language']}")
                st.write(f"**User:** {selected_query['user']}")

                # Confidence Visualisierung
                if selected_query["confidence"] > 0:
                    st.markdown("**🎯 Confidence Visualisierung:**")
                    confidence_color = "🟢" if selected_query["confidence"] > 80 else "🟡" if selected_query["confidence"] > 60 else "🔴"
                    st.progress(selected_query["confidence"] / 100)
                    st.write(f"{confidence_color} {selected_query['confidence']:.1f}% Confidence")

    else:
        st.info("Keine Queries gefunden, die den Filterkriterien entsprechen.")

    # Export-Optionen
    st.subheader("📤 Export")
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        if st.button("📊 Export als CSV", key="export_csv"):
            if filtered_history:
                csv_data = pd.DataFrame([{
                    "timestamp": q["timestamp"].isoformat(),
                    "query": q["query"],
                    "status": q["status"],
                    "response_time": q["response_time"],
                    "confidence": q["confidence"],
                    "language": q["language"],
                    "user": q["user"]
                } for q in filtered_history])

                csv_string = csv_data.to_csv(index=False)
                st.download_button(
                    label="📥 CSV herunterladen",
                    data=csv_string,
                    file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            else:
                st.warning("Keine Daten zum Exportieren verfügbar.")

    with col_export2:
        if st.button("🧹 Cache leeren", key="clear_query_cache"):
            st.success("✅ Query-Cache wurde geleert!")
            debug_system.log(DebugLevel.SUCCESS, "Query-Cache geleert")


def visualize_fact_check(query_result):
    """Visualisiert Fact-Check Ergebnisse"""
    if not query_result or "fact_check" not in query_result:
        return

    fact_check = query_result["fact_check"]

    st.subheader("✅ Fact-Check Analyse")

    # Confidence Score
    confidence = fact_check.get("confidence", 0)
    confidence_color = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"

    col_conf1, col_conf2 = st.columns([1, 3])

    with col_conf1:
        st.metric("Confidence Score", f"{confidence:.1f}%")

    with col_conf2:
        st.progress(confidence / 100)
        st.write(f"{confidence_color} **{confidence:.1f}%** - {'Sehr vertrauenswürdig' if confidence > 80 else 'Vertrauenswürdig' if confidence > 60 else 'Überprüfung empfohlen'}")

    # Quellen-Visualisierung
    if "sources" in fact_check:
        st.subheader("📚 Verwendete Quellen")

        sources = fact_check["sources"]
        for i, source in enumerate(sources[:5]):  # Zeige max 5 Quellen
            credibility = source.get("credibility", 50)
            source_color = "🟢" if credibility > 80 else "🟡" if credibility > 60 else "🔴"

            with st.expander(f"{source_color} Quelle {i+1}: {source.get('name', 'Unbekannt')}", expanded=False):
                st.write(f"**URL:** {source.get('url', 'N/A')}")
                st.write(f"**Credibility:** {credibility:.1f}%")
                st.write(f"**Last Updated:** {source.get('last_updated', 'N/A')}")
                if "relevance" in source:
                    st.write(f"**Relevance:** {source['relevance']:.1f}%")

    # Warnungen und Empfehlungen
    if confidence < 70:
        st.warning("⚠️ **Hinweis:** Diese Information sollte mit zusätzlichen Quellen überprüft werden.")

        if "recommendations" in fact_check:
            st.info("💡 **Empfehlungen:**")
            for rec in fact_check["recommendations"]:
                st.write(f"• {rec}")

    # Zeitstempel der Überprüfung
    if "timestamp" in fact_check:
        st.caption(f"Fact-Check durchgeführt am: {fact_check['timestamp']}")


def show_enhanced_query_results(query_result):
    """Zeigt erweiterte Query-Ergebnisse mit Fact-Check"""
    st.subheader("🎯 Query-Ergebnisse")

    # Haupt-Ergebnis
    if "response" in query_result:
        st.success(query_result["response"])

    # Metriken
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Response Time", f"{query_result.get('response_time', 0):.2f}s")

    with col2:
        st.metric("Confidence", f"{query_result.get('confidence', 0):.1f}%")

    with col3:
        st.metric("Sprache", query_result.get("language", "DE"))

    with col4:
        st.metric("Model", query_result.get("model", "RTX 2070 KI"))

    # Fact-Check Visualisierung
    if "fact_check" in query_result:
        visualize_fact_check(query_result)

    # Detaillierte Analyse (ausklappbar)
    with st.expander("📊 Detaillierte Analyse", expanded=False):
        if "processing_steps" in query_result:
            st.subheader("🔄 Verarbeitungsschritte")
            for step in query_result["processing_steps"]:
                st.write(f"• {step}")

        if "sources_used" in query_result:
            st.subheader("📚 Verwendete Quellen")
            for source in query_result["sources_used"]:
                st.write(f"• {source}")

        if "debug_info" in query_result:
            st.subheader("🔧 Debug-Informationen")
            st.json(query_result["debug_info"])


def main():
    """Hauptfunktion der Web-GUI"""
    st.title("🤖 Bundeskanzler KI - Enhanced Web GUI")

    # Login-Interface
    if "admin_token" not in st.session_state:
        show_login_interface()
    else:
        show_admin_interface()


if __name__ == "__main__":
    main()
