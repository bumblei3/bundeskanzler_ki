#!/usr/bin/env python3
"""
Kommandozeilen Admin-Tools für die Bundeskanzler KI
"""

import json
import sys
from datetime import datetime

import requests


class AdminCLI:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.token = None

    def login(self, username="admin", password="admin123!"):
        """Admin-Login"""
        try:
            response = requests.post(
                f"{self.api_url}/auth/admin-token",
                data={"username": username, "password": password},
                timeout=10,
            )
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                print("✅ Admin-Login erfolgreich!")
                return True
            else:
                print(f"❌ Login fehlgeschlagen: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Login-Fehler: {e}")
            return False

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def system_status(self):
        """Zeige System-Status"""
        print("📊 SYSTEM STATUS")
        print("=" * 30)

        try:
            # Health Check
            resp = requests.get(f"{self.api_url}/admin/health", headers=self.headers)
            if resp.status_code == 200:
                health = resp.json()
                print(f"⏱️  Uptime: {health['system']['uptime']:.1f} Sekunden")
                print(f"📈 Requests: {health['system']['request_count']}")
                print(
                    f"🔧 Components: {'✅ OK' if health['system']['components_initialized'] else '❌ Error'}"
                )
                print(f"📁 Files: {'✅ OK' if health['files']['logs_accessible'] else '❌ Error'}")

            # System Stats
            resp = requests.get(f"{self.api_url}/admin/system-stats", headers=self.headers)
            if resp.status_code == 200:
                stats = resp.json()
                print(f"📊 API Requests (24h): {stats.get('api_requests_24h', 0)}")
                print(f"👥 Active Users: {stats.get('active_users', 0)}")
                print(f"💾 Memory Entries: {stats.get('memory_entries', 0)}")
                print(f"⚠️  Error Rate: {stats.get('error_rate', 0):.1f}%")

        except Exception as e:
            print(f"❌ Fehler: {e}")

    def list_users(self):
        """Liste alle Benutzer"""
        print("👥 BENUTZER-MANAGEMENT")
        print("=" * 30)

        try:
            resp = requests.get(f"{self.api_url}/admin/users", headers=self.headers)
            if resp.status_code == 200:
                data = resp.json()
                print(f"Gefunden: {data['total']} Benutzer\\n")

                for user in data["users"]:
                    admin_badge = "🔐 Admin" if user["is_admin"] else "👤 User"
                    status_badge = "✅ Aktiv" if user["is_active"] else "❌ Inaktiv"
                    print(f"{admin_badge} {user['user_id']}")
                    print(f"   📧 Email: {user['email']}")
                    print(f"   📊 Status: {status_badge}")
                    print(f"   🔢 Login Count: {user['login_count']}")
                    print(f"   ⚙️  API Limits: {user['api_limits']['requests_per_minute']}/min")
                    print()
            else:
                print(f"❌ Fehler: {resp.status_code}")

        except Exception as e:
            print(f"❌ Fehler: {e}")

    def view_logs(self, log_type="api.log", lines=10):
        """Zeige Logs"""
        print(f"📋 LOGS: {log_type}")
        print("=" * 30)

        try:
            resp = requests.get(
                f"{self.api_url}/admin/logs/{log_type}?lines={lines}",
                headers=self.headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                print(f"Zeige {len(data['entries'])} von {log_type}:\\n")

                for entry in reversed(data["entries"]):
                    level_icon = {
                        "ERROR": "🔴",
                        "WARNING": "🟡",
                        "INFO": "🔵",
                        "DEBUG": "⚪",
                    }.get(entry["level"], "⚪")

                    print(f"{level_icon} {entry['timestamp']} [{entry['level']}] {entry['logger']}")
                    print(f"   {entry['message']}")
                    print()
            else:
                print(f"❌ Fehler: {resp.status_code}")

        except Exception as e:
            print(f"❌ Fehler: {e}")

    def memory_stats(self):
        """Zeige Memory-Statistiken"""
        print("💾 MEMORY MANAGEMENT")
        print("=" * 30)

        try:
            resp = requests.get(f"{self.api_url}/admin/memory/stats", headers=self.headers)
            if resp.status_code == 200:
                stats = resp.json()
                print(
                    f"📊 Kurzzeitgedächtnis: {stats.get('kurzzeitgedaechtnis_entries', 0)} Einträge"
                )
                print(
                    f"📊 Langzeitgedächtnis: {stats.get('langzeitgedaechtnis_entries', 0)} Einträge"
                )
                print(f"📊 Total Entries: {stats.get('total_entries', 0)}")
                print(f"📊 Effizienz: {stats.get('memory_efficiency', 0):.1f}%")
            else:
                print(f"❌ Memory Stats nicht verfügbar: {resp.status_code}")

        except Exception as e:
            print(f"❌ Fehler: {e}")

    def interactive_menu(self):
        """Interaktives Admin-Menu"""
        print("🔐 BUNDESKANZLER KI - ADMIN CLI")
        print("=" * 40)

        if not self.login():
            return

        while True:
            print("\\n📋 ADMIN MENU:")
            print("1. System Status")
            print("2. Benutzer-Management")
            print("3. API Logs anzeigen")
            print("4. Memory Logs anzeigen")
            print("5. Error Logs anzeigen")
            print("6. Memory-Statistiken")
            print("0. Beenden")

            try:
                choice = input("\\nWähle eine Option (0-6): ").strip()

                if choice == "1":
                    self.system_status()
                elif choice == "2":
                    self.list_users()
                elif choice == "3":
                    self.view_logs("api.log")
                elif choice == "4":
                    self.view_logs("memory.log")
                elif choice == "5":
                    self.view_logs("errors.log")
                elif choice == "6":
                    self.memory_stats()
                elif choice == "0":
                    print("👋 Auf Wiedersehen!")
                    break
                else:
                    print("❌ Ungültige Auswahl!")

                input("\\nDrücke Enter um fortzufahren...")

            except KeyboardInterrupt:
                print("\\n👋 Auf Wiedersehen!")
                break
            except Exception as e:
                print(f"❌ Fehler: {e}")


def main():
    if len(sys.argv) > 1:
        admin = AdminCLI()
        if not admin.login():
            return

        command = sys.argv[1].lower()

        if command == "status":
            admin.system_status()
        elif command == "users":
            admin.list_users()
        elif command == "logs":
            log_type = sys.argv[2] if len(sys.argv) > 2 else "api.log"
            admin.view_logs(log_type)
        elif command == "memory":
            admin.memory_stats()
        else:
            print(f"❌ Unbekanntes Kommando: {command}")
            print("Verfügbare Kommandos: status, users, logs, memory")
    else:
        # Interaktives Menu
        AdminCLI().interactive_menu()


if __name__ == "__main__":
    main()
