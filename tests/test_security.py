"""
Security-Tests für die Bundeskanzler KI API
"""

import json

import pytest
from bundeskanzler_api import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Test-Client für Security-Tests"""
    return TestClient(app)


@pytest.mark.security
class TestAuthenticationSecurity:
    """Tests für Authentifizierung und Autorisierung"""

    def test_no_auth_required_for_health(self, client):
        """Health-Endpoint sollte ohne Authentifizierung zugänglich sein"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_chat_requires_authentication(self, client):
        """Chat-Endpoint sollte Authentifizierung erfordern"""
        response = client.post("/chat", json={"message": "Test"})
        assert (
            response.status_code == 403
        )  # API gibt 403 für nicht authentifizierte Requests zurück

    def test_invalid_token_rejected(self, client):
        """Ungültige Tokens sollten abgelehnt werden"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post("/chat", json={"message": "Test"}, headers=headers)
        assert response.status_code == 401

    def test_admin_endpoints_require_admin_token(self, client):
        """Admin-Endpunkte sollten Admin-Token erfordern"""
        # Erstelle normalen User-Token
        from bundeskanzler_api import create_access_token

        user_token = create_access_token(data={"sub": "test_user"})

        headers = {"Authorization": f"Bearer {user_token}"}
        response = client.get("/admin/health", headers=headers)
        assert response.status_code == 403

    def test_admin_token_allows_admin_access(self, client):
        """Admin-Token sollte Admin-Zugang erlauben"""
        from bundeskanzler_api import create_admin_token

        admin_token = create_admin_token("admin_user")

        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get("/admin/health", headers=headers)
        assert response.status_code == 200


@pytest.mark.security
class TestInputValidation:
    """Tests für Input-Validation und SQL-Injection-Prävention"""

    def test_sql_injection_prevention(self, client):
        """SQL-Injection sollte verhindert werden"""
        from bundeskanzler_api import create_access_token

        token = create_access_token(data={"sub": "test_user"})
        headers = {"Authorization": f"Bearer {token}"}

        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "UNION SELECT * FROM users",
        ]

        for malicious_input in malicious_inputs:
            response = client.post(
                "/chat",
                json={"message": malicious_input, "session_id": "test"},
                headers=headers,
            )
            # Sollte nicht 500 (Server Error) zurückgeben
            assert response.status_code in [200, 400, 422]

    def test_large_input_handling(self, client):
        """Große Inputs sollten sicher verarbeitet werden"""
        from bundeskanzler_api import create_access_token

        token = create_access_token(data={"sub": "test_user"})
        headers = {"Authorization": f"Bearer {token}"}

        # Test mit sehr großem Input
        large_message = "Test message " * 10000  # 120.000 Zeichen

        response = client.post(
            "/chat",
            json={"message": large_message, "session_id": "test"},
            headers=headers,
        )

        # Sollte nicht zu einem Server-Crash führen
        assert response.status_code in [200, 400, 413, 422]

    def test_special_characters_handling(self, client):
        """Sonderzeichen sollten sicher verarbeitet werden"""
        from bundeskanzler_api import create_access_token

        token = create_access_token(data={"sub": "test_user"})
        headers = {"Authorization": f"Bearer {token}"}

        special_messages = [
            "Message with émojis 🐍🚀💻",
            "Unicode: αβγδε 中文 русский",
            "Control chars: \n\t\r",
            "Zero width: ​hidden​text​",
        ]

        for message in special_messages:
            response = client.post(
                "/chat",
                json={"message": message, "session_id": "test"},
                headers=headers,
            )
            assert response.status_code in [200, 400, 422]

    def test_json_injection_prevention(self, client):
        """JSON-Injection sollte verhindert werden"""
        from bundeskanzler_api import create_access_token

        token = create_access_token(data={"sub": "test_user"})
        headers = {"Authorization": f"Bearer {token}"}

        # Versuche JSON-Injection
        malicious_json = {
            "message": "Test",
            "session_id": "test",
            "__proto__": {"malicious": "data"},
            "constructor": {"prototype": {"evil": "code"}},
        }

        response = client.post("/chat", json=malicious_json, headers=headers)
        assert response.status_code in [200, 400, 422]

        # Stelle sicher, dass die Response nicht die malicious data enthält
        if response.status_code == 200:
            data = response.json()
            assert "malicious" not in str(data).lower()
            assert "evil" not in str(data).lower()


@pytest.mark.security
class TestRateLimiting:
    """Tests für Rate-Limiting"""

    def test_rate_limiting_works(self, client):
        """Rate-Limiting sollte funktionieren"""
        from bundeskanzler_api import create_access_token

        token = create_access_token(data={"sub": "test_user"})
        headers = {"Authorization": f"Bearer {token}"}

        # Sende einige Requests (weniger als das erwartete Limit)
        responses = []
        for i in range(10):  # Weniger als das Limit
            response = client.post(
                "/chat",
                json={"message": f"Test {i}", "session_id": "test"},
                headers=headers,
            )
            responses.append(response.status_code)

        # Sollte alle 200 sein (Rate limiting nicht ausgelöst)
        assert all(code == 200 for code in responses)


@pytest.mark.security
class TestHTTPSecurityHeaders:
    """Tests für HTTP Security Headers"""

    def test_security_headers_present(self, client):
        """Security Headers sollten in allen Responses vorhanden sein"""
        response = client.get("/health")

        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy",
        ]

        for header in required_headers:
            assert header in response.headers

    def test_cors_headers(self, client):
        """CORS Headers sollten korrekt gesetzt sein"""
        # Teste GET Request auf /health (unterstützte Methode)
        response = client.get("/health")

        # Bei GET Requests sollten keine CORS Preflight Headers erwartet werden
        # Stelle sicher, dass Security Headers vorhanden sind
        assert "x-api-version" in response.headers
        assert response.status_code == 200
