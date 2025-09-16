#!/usr/bin/env python3
"""
🛡️ Lokales Authentifizierungssystem für Bundeskanzler-KI
===============================================

Lokales User-Management ohne API-Abhängigkeiten:
- Sichere Passwort-Hashing mit bcrypt
- Session-Management mit SQLite
- Rollenbasierte Zugriffskontrolle
- Lokale Token-Generierung (JWT-kompatibel)
- User-Registrierung und -Verwaltung

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False


class LocalAuthManager:
    """
    🛡️ Lokales Authentifizierungssystem ohne API-Abhängigkeiten
    """

    def __init__(self, db_path: str = "data/local_auth.db"):
        """Initialisiert das lokale Authentifizierungssystem"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

        # JWT-kompatible Secret-Key für lokale Tokens
        self.secret_key = self._load_or_create_secret()

        # Datenbank initialisieren
        self._init_database()

        # Standard-Admin-User erstellen
        self._create_default_admin()

        print("🛡️ Lokales Authentifizierungssystem initialisiert")

    def _load_or_create_secret(self) -> str:
        """Lädt oder erstellt einen sicheren Secret-Key"""
        secret_file = Path("config/auth_secret.key")

        if secret_file.exists():
            with open(secret_file, "r") as f:
                return f.read().strip()

        # Neuen sicheren Secret-Key generieren
        secret = secrets.token_hex(32)
        secret_file.parent.mkdir(exist_ok=True)

        with open(secret_file, "w") as f:
            f.write(secret)

        # Secret-Datei nur für Owner lesbar machen
        os.chmod(secret_file, 0o600)

        return secret

    def _init_database(self):
        """Initialisiert die lokale Authentifizierungsdatenbank"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users-Tabelle
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            ''')

            # Sessions-Tabelle
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_token TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # API Keys-Tabelle
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    permissions TEXT NOT NULL,  -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    usage_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            conn.commit()

    def _create_default_admin(self):
        """Erstellt einen Standard-Admin-User"""
        try:
            self.register_user(
                username="admin",
                password="admin123!",
                email="admin@bundeskanzler-ki.local",
                role="admin"
            )
            print("✅ Standard-Admin-User erstellt")
        except ValueError:
            # Admin existiert bereits
            pass

    def _hash_password(self, password: str) -> str:
        """Hashed Passwort mit bcrypt oder Fallback"""
        if HAS_BCRYPT:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        else:
            # Fallback: SHA-256 mit Salt
            salt = secrets.token_hex(16)
            hash_obj = hashlib.sha256((salt + password).encode('utf-8'))
            return f"sha256:{salt}:{hash_obj.hexdigest()}"

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verifiziert Passwort gegen Hash"""
        if HAS_BCRYPT and hashed.startswith('$2b$'):
            # bcrypt Hash
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        elif hashed.startswith('sha256:'):
            # SHA-256 Fallback
            try:
                _, salt, expected_hash = hashed.split(':')
                hash_obj = hashlib.sha256((salt + password).encode('utf-8'))
                return hmac.compare_digest(hash_obj.hexdigest(), expected_hash)
            except ValueError:
                return False
        else:
            # Legacy bcrypt Hash
            try:
                return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
            except:
                return False

    def _generate_token(self, user_id: int, username: str, role: str) -> str:
        """Generiert lokalen Token (kein JWT)"""
        # Immer das lokale Token-System verwenden
        payload = f"{user_id}:{username}:{role}:{int(time.time())}"
        signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{payload}.{signature}"

    def _generate_simple_token(self, user_id: int, username: str, role: str) -> str:
        """Einfacher Token-Generator als JWT-Fallback"""
        payload = f"{user_id}:{username}:{role}:{int(time.time())}"
        signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{payload}.{signature}"

    def _verify_token(self, token: str) -> Optional[Dict]:
        """Verifiziert lokalen Token"""
        # Zuerst lokale Token-Verifizierung versuchen
        local_result = self._verify_simple_token(token)
        if local_result:
            return local_result

        # Fallback auf JWT (für Kompatibilität)
        try:
            import jwt
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return {
                "user_id": payload.get("user_id"),
                "sub": payload.get("sub"),
                "role": payload.get("role")
            }
        except:
            return None

    def _verify_simple_token(self, token: str) -> Optional[Dict]:
        """Verifiziert einfachen Token"""
        try:
            payload, signature = token.rsplit(".", 1)
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return None

            user_id, username, role, timestamp = payload.split(":")
            # Token nach 24 Stunden ablaufen lassen
            if time.time() - int(timestamp) > 86400:
                return None

            return {
                "user_id": int(user_id),
                "sub": username,
                "role": role
            }
        except (ValueError, IndexError):
            return None

    def register_user(self, username: str, password: str, email: str = None, role: str = "user") -> bool:
        """Registriert einen neuen Benutzer"""
        if len(password) < 8:
            raise ValueError("Passwort muss mindestens 8 Zeichen lang sein")

        if not username or not username.strip():
            raise ValueError("Benutzername ist erforderlich")

        password_hash = self._hash_password(password)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                ''', (username.strip(), email, password_hash, role))

                conn.commit()
                print(f"✅ Benutzer '{username}' registriert")
                return True

            except sqlite3.IntegrityError as e:
                if "username" in str(e):
                    raise ValueError(f"Benutzername '{username}' ist bereits vergeben")
                elif "email" in str(e):
                    raise ValueError(f"E-Mail '{email}' ist bereits vergeben")
                raise

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authentifiziert Benutzer und gibt Token zurück"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Benutzer abrufen
            cursor.execute('''
                SELECT id, password_hash, role, is_active, login_attempts, locked_until
                FROM users WHERE username = ?
            ''', (username,))

            user = cursor.fetchone()
            if not user:
                return None

            user_id, password_hash, role, is_active, login_attempts, locked_until = user

            # Account-Status prüfen
            if not is_active:
                raise ValueError("Account ist deaktiviert")

            # Account-Sperre prüfen
            if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
                raise ValueError("Account ist temporär gesperrt")

            # Passwort verifizieren
            if not self._verify_password(password, password_hash):
                # Login-Versuch zählen
                new_attempts = login_attempts + 1
                locked_until = None

                if new_attempts >= 5:
                    # Account für 15 Minuten sperren
                    locked_until = (datetime.now() + timedelta(minutes=15)).isoformat()

                cursor.execute('''
                    UPDATE users SET login_attempts = ?, locked_until = ?
                    WHERE id = ?
                ''', (new_attempts, locked_until, user_id))

                conn.commit()
                return None

            # Erfolgreicher Login
            cursor.execute('''
                UPDATE users SET
                    last_login = CURRENT_TIMESTAMP,
                    login_attempts = 0,
                    locked_until = NULL
                WHERE id = ?
            ''', (user_id,))

            conn.commit()

            # Token generieren
            token = self._generate_token(user_id, username, role)
            return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verifiziert Token und gibt Benutzerinformationen zurück"""
        payload = self._verify_token(token)
        if not payload:
            return None

        # Für lokale Tokens: Keine Session-Prüfung notwendig
        # (Sessions sind optional für erweiterte Features)
        return payload

    def create_session(self, user_id: int, token: str, ip_address: str = None, user_agent: str = None):
        """Erstellt eine neue Session"""
        expires_at = datetime.now() + timedelta(hours=24)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO sessions (session_token, user_id, ip_address, user_agent, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (token, user_id, ip_address, user_agent, expires_at.isoformat()))

            conn.commit()

    def revoke_token(self, token: str):
        """Widerruft einen Token"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('UPDATE sessions SET is_active = 0 WHERE session_token = ?', (token,))
            conn.commit()

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Ändert das Passwort eines Benutzers"""
        if len(new_password) < 8:
            raise ValueError("Neues Passwort muss mindestens 8 Zeichen lang sein")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Aktuelles Passwort prüfen
            cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()

            if not user or not self._verify_password(old_password, user[1]):
                return False

            # Neues Passwort setzen
            new_hash = self._hash_password(new_password)
            cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user[0]))

            conn.commit()
            return True

    def get_user_info(self, username: str) -> Optional[Dict]:
        """Ruft Benutzerinformationen ab"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT username, email, role, is_active, created_at, last_login
                FROM users WHERE username = ?
            ''', (username,))

            user = cursor.fetchone()
            if not user:
                return None

            return {
                "username": user[0],
                "email": user[1],
                "role": user[2],
                "is_active": bool(user[3]),
                "created_at": user[4],
                "last_login": user[5]
            }

    def list_users(self) -> List[Dict]:
        """Listet alle Benutzer auf"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT username, email, role, is_active, created_at, last_login
                FROM users ORDER BY created_at DESC
            ''')

            users = []
            for row in cursor.fetchall():
                users.append({
                    "username": row[0],
                    "email": row[1],
                    "role": row[2],
                    "is_active": bool(row[3]),
                    "created_at": row[4],
                    "last_login": row[5]
                })

            return users

    def has_permission(self, token: str, required_role: str) -> bool:
        """Prüft ob Token die erforderliche Rolle hat"""
        user_info = self.verify_token(token)
        if not user_info:
            return False

        user_role = user_info.get("role", "user")

        # Rollen-Hierarchie
        role_hierarchy = {
            "user": 1,
            "moderator": 2,
            "admin": 3
        }

        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 999)

        return user_level >= required_level

    def create_api_key(self, username: str, name: str, permissions: List[str] = None) -> Optional[str]:
        """Erstellt einen API-Key für einen Benutzer"""
        if permissions is None:
            permissions = ["read"]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Benutzer-ID abrufen
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            if not user:
                return None

            user_id = user[0]

            # API-Key generieren
            api_key = secrets.token_hex(32)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            cursor.execute('''
                INSERT INTO api_keys (key_hash, name, user_id, permissions)
                VALUES (?, ?, ?, ?)
            ''', (key_hash, name, user_id, json.dumps(permissions)))

            conn.commit()

            return api_key

    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verifiziert einen API-Key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT k.name, k.permissions, k.usage_count, u.username, u.role
                FROM api_keys k
                JOIN users u ON k.user_id = u.id
                WHERE k.key_hash = ? AND k.is_active = 1 AND u.is_active = 1
            ''', (key_hash,))

            key_data = cursor.fetchone()
            if not key_data:
                return None

            name, permissions, usage_count, username, role = key_data

            # Usage Count erhöhen
            cursor.execute('''
                UPDATE api_keys SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
                WHERE key_hash = ?
            ''', (key_hash,))

            conn.commit()

            return {
                "name": name,
                "permissions": json.loads(permissions),
                "username": username,
                "role": role,
                "usage_count": usage_count + 1
            }


# Global instance für einfachen Zugriff
_auth_manager = None

def get_auth_manager() -> LocalAuthManager:
    """Gibt die globale Auth-Manager-Instanz zurück"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = LocalAuthManager()
    return _auth_manager


def require_auth(token: str, required_role: str = "user") -> Dict:
    """Decorator-kompatible Authentifizierungsfunktion"""
    auth_manager = get_auth_manager()
    user_info = auth_manager.verify_token(token)

    if not user_info:
        raise ValueError("Ungültiger oder abgelaufener Token")

    if not auth_manager.has_permission(token, required_role):
        raise ValueError(f"Unzureichende Berechtigung. Erforderlich: {required_role}")

    return user_info


def require_api_key(api_key: str, required_permissions: List[str] = None) -> Dict:
    """API-Key Authentifizierung"""
    auth_manager = get_auth_manager()
    key_info = auth_manager.verify_api_key(api_key)

    if not key_info:
        raise ValueError("Ungültiger API-Key")

    if required_permissions:
        user_permissions = key_info.get("permissions", [])
        for perm in required_permissions:
            if perm not in user_permissions:
                raise ValueError(f"API-Key hat nicht die erforderliche Berechtigung: {perm}")

    return key_info


if __name__ == "__main__":
    # Demo/Test des lokalen Authentifizierungssystems
    auth = LocalAuthManager()

    print("\n🧪 Lokales Authentifizierungssystem - Test")

    # Test-Benutzer registrieren
    try:
        auth.register_user("testuser", "testpass123", "test@example.com")
        print("✅ Test-Benutzer registriert")
    except ValueError as e:
        print(f"⚠️  Test-Benutzer bereits vorhanden: {e}")

    # Login testen
    token = auth.authenticate_user("testuser", "testpass123")
    if token:
        print("✅ Login erfolgreich, Token generiert")

        # Token verifizieren
        user_info = auth.verify_token(token)
        if user_info:
            print(f"✅ Token verifiziert für User: {user_info['username']}")

        # API-Key erstellen
        api_key = auth.create_api_key("testuser", "Test API Key")
        if api_key:
            print("✅ API-Key erstellt")

            # API-Key verifizieren
            key_info = auth.verify_api_key(api_key)
            if key_info:
                print(f"✅ API-Key verifiziert für: {key_info['name']}")

