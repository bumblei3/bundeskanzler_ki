"""
Load-Tests für die Bundeskanzler KI API mit Locust
Führen Sie aus mit: locust -f tests/load_tests.py --host=http://localhost:8000
"""

import json
import time

from locust import HttpUser, between, task


class BundeskanzlerUser(HttpUser):
    """Simuliert einen typischen Benutzer der Bundeskanzler KI"""

    wait_time = between(1, 3)  # Wartezeit zwischen Tasks: 1-3 Sekunden

    def on_start(self):
        """Wird beim Start jedes Users ausgeführt"""
        # Admin-Token für authentifizierte Requests
        response = self.client.post(
            "/auth/admin-token", data={"username": "admin", "password": "admin123!"}
        )
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task(3)  # 30% der Tasks
    def health_check(self):
        """Health-Check (häufigste Operation)"""
        self.client.get("/health")

    @task(5)  # 50% der Tasks
    def chat_request(self):
        """Chat-Request (Hauptfunktionalität)"""
        if self.token:
            chat_data = {
                "message": "Wie geht es der deutschen Wirtschaft?",
                "session_id": f"load_test_{self.user_id}_{int(time.time())}",
                "include_sources": False,
                "max_length": 300,
            }
            self.client.post("/chat", json=chat_data, headers=self.headers)

    @task(1)  # 10% der Tasks
    def admin_health(self):
        """Admin Health-Check"""
        if self.token:
            self.client.get("/admin/health", headers=self.headers)

    @task(1)  # 10% der Tasks
    def api_docs(self):
        """API-Dokumentation"""
        self.client.get("/docs")


class AggressiveUser(HttpUser):
    """Simuliert einen aggressiven Benutzer (für Stress-Tests)"""

    wait_time = between(0.1, 0.5)  # Sehr kurze Wartezeiten

    def on_start(self):
        response = self.client.post(
            "/auth/admin-token", data={"username": "admin", "password": "admin123!"}
        )
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task
    def rapid_chat_requests(self):
        """Sehr schnelle Chat-Requests"""
        if self.token:
            chat_data = {
                "message": "Test message for load testing",
                "session_id": f"stress_test_{int(time.time() * 1000)}",
                "include_sources": False,
                "max_length": 200,
            }
            self.client.post("/chat", json=chat_data, headers=self.headers)


class MemoryIntensiveUser(HttpUser):
    """Simuliert Memory-intensive Operationen"""

    wait_time = between(2, 5)

    def on_start(self):
        response = self.client.post(
            "/auth/admin-token", data={"username": "admin", "password": "admin123!"}
        )
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task
    def long_context_chat(self):
        """Chat mit langen Kontexten"""
        if self.token:
            long_message = (
                "Bitte analysieren Sie die aktuelle Situation in Deutschland bezüglich Klimapolitik, Wirtschaftsentwicklung, Digitalisierung und Sozialpolitik. Geben Sie eine umfassende Bewertung der aktuellen Herausforderungen und Zukunftsaussichten. "
                * 3
            )

            chat_data = {
                "message": long_message,
                "session_id": f"memory_test_{int(time.time())}",
                "include_sources": True,
                "max_length": 800,
            }
            self.client.post("/chat", json=chat_data, headers=self.headers)
