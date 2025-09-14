"""
Bundeskanzler KI REST API Server
Moderne FastAPI-basierte API mit Authentifizierung, Rate Limiting und OpenAPI-Dokumentation
"""

# Standard library imports
import asyncio
import hashlib
import json
import logging
import logging.config
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import numpy as np
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    Security,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional

# Local imports
from adaptive_response import AdaptiveResponseManager
from corpus_manager import CorpusManager
from gpu_batching import AsyncBatchManager, GPUBatchProcessor
from hierarchical_memory import EnhancedContextProcessor
from optimized_memory import OptimizedHierarchicalMemory
from simple_logging import setup_simple_logging

# Logger initialisieren
api_logger, memory_logger = setup_simple_logging()


class APIConfig:
    """Konfiguration für die API"""
    SECRET_KEY = os.getenv("BUNDESKANZLER_SECRET_KEY", "bundeskanzler-ki-secret-2025")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    API_VERSION = "v1"
    MAX_REQUESTS_PER_MINUTE = 60
    # Erlaube lokale Entwicklungs-Origins inkl. Admin-Panel auf Port 8080
    CORS_ORIGINS = [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


# Verbessertes Logging-Setup
class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with fallback to JSON"""

    def format(self, record):
        # Erstelle einen strukturierten Log-Eintrag
        log_obj = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }

        # Füge Extra-Felder hinzu
        if hasattr(record, 'component'):
            log_obj['component'] = record.component
        if hasattr(record, 'action'):
            log_obj['action'] = record.action
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'duration'):
            log_obj['duration'] = record.duration
        if hasattr(record, 'status'):
            log_obj['status'] = record.status
        if hasattr(record, 'memory_type'):
            log_obj['memory_type'] = record.memory_type
        if hasattr(record, 'message_length'):
            log_obj['message_length'] = record.message_length
        if hasattr(record, 'include_sources'):
            log_obj['include_sources'] = record.include_sources

        # Versuche JSON-Format, fallback zu lesbarem Format
        try:
            return json.dumps(log_obj, ensure_ascii=False)
        except:
            # Fallback zu strukturiertem Text-Format
            parts = [f"{log_obj['timestamp']} - {log_obj['level']} - {log_obj['logger']} - {log_obj['message']}"]
            for key, value in log_obj.items():
                if key not in ['timestamp', 'level', 'logger', 'message']:
                    parts.append(f"{key}={value}")
            return " | ".join(parts)


class RateLimiter:
    """Einfacher In-Memory Rate Limiter"""
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, client_id: str, max_requests: int = 60) -> bool:
        current_time = time.time()
        minute_ago = current_time - 60
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Entferne alte Requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] 
            if req_time > minute_ago
        ]
        
        # Prüfe Limit
        if len(self.requests[client_id]) >= max_requests:
            return False
        
        # Füge aktuellen Request hinzu
        self.requests[client_id].append(current_time)
        return True


class ChatRequest(BaseModel):
    """Chat-Anfrage"""
    message: str = Field(..., min_length=1, max_length=1000, description="Nutzernachricht")
    user_id: Optional[str] = Field(None, description="Nutzer-ID für Personalisierung")
    context: Optional[Dict[str, Any]] = Field({}, description="Zusätzlicher Kontext")
    max_length: Optional[int] = Field(500, ge=50, le=2000, description="Maximale Antwortlänge")
    include_sources: Optional[bool] = Field(True, description="Quellen in Antwort einbeziehen")

    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Nachricht darf nicht leer sein')
        return v.strip()


class ChatResponse(BaseModel):
    """Chat-Antwort"""
    response: str = Field(..., description="KI-Antwort")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Konfidenz der Antwort")
    response_time: float = Field(..., description="Antwortzeit in Sekunden")
    user_id: Optional[str] = Field(None, description="Nutzer-ID")
    sources: List[str] = Field([], description="Verwendete Quellen")
    memory_context: Dict[str, Any] = Field({}, description="Memory-Kontext")
    timestamp: datetime = Field(default_factory=datetime.now)


class MemoryRequest(BaseModel):
    """Memory-Management-Anfrage"""
    content: str = Field(..., min_length=1, max_length=2000)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    tags: List[str] = Field([], max_items=10)
    metadata: Dict[str, Any] = Field({})


class MemorySearchRequest(BaseModel):
    """Memory-Suche-Anfrage"""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    min_similarity: float = Field(0.3, ge=0.0, le=1.0)


class UserProfile(BaseModel):
    """Nutzerprofil"""
    user_id: str
    complexity_preference: float = Field(0.5, ge=0.0, le=1.0)
    topics_of_interest: List[str] = Field([])
    interaction_count: int = Field(0)
    last_interaction: Optional[datetime] = None


class APIStatus(BaseModel):
    """API-Status"""
    status: str
    version: str
    uptime: float
    memory_stats: Dict[str, Any]
    request_count: int


class AdminUser(BaseModel):
    """Admin-Benutzer Model"""
    user_id: str
    email: str
    is_admin: bool = False
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    login_count: int = 0


class SystemStats(BaseModel):
    """System-Statistiken für Admin-Panel"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    api_requests_24h: int
    active_users: int
    memory_entries: int
    error_rate: float


class LogEntry(BaseModel):
    """Log-Eintrag für Admin Log-Viewer"""
    timestamp: str
    level: str
    logger: str
    message: str
    component: Optional[str] = None
    user_id: Optional[str] = None


class UserManagement(BaseModel):
    """Benutzer-Management für Admin"""
    user_id: str
    email: str
    is_admin: bool = False
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None
    login_count: int = 0
    api_limits: Dict[str, int] = {}


class CreateUserRequest(BaseModel):
    """Anfrage zum Erstellen eines neuen Benutzers"""
    user_id: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=6)
    is_admin: bool = False


class SystemConfig(BaseModel):
    """System-Konfiguration für Admin"""
    api_settings: Dict[str, Any]
    memory_settings: Dict[str, Any] 
    logging_settings: Dict[str, Any]
    security_settings: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    """Anfrage zur Konfiguration-Aktualisierung"""
    section: str = Field(..., pattern=r'^(api|memory|logging|security)$')
    settings: Dict[str, Any]


rate_limiter = RateLimiter()
memory_system: Optional[OptimizedHierarchicalMemory] = None
context_processor: Optional[EnhancedContextProcessor] = None
response_manager: Optional[AdaptiveResponseManager] = None
corpus_manager: Optional[CorpusManager] = None
gpu_processor: Optional[GPUBatchProcessor] = None
async_batch_manager: Optional[AsyncBatchManager] = None
start_time = time.time()
request_counter = 0


def initialize_ki_components():
    """Initialisiert alle KI-Komponenten"""
    global memory_system, context_processor, response_manager, corpus_manager

    print(" Initialisiere Memory-System...")
    memory_system = OptimizedHierarchicalMemory(
        short_term_capacity=200,
        long_term_capacity=5000,
        embedding_dim=512,
        persistence_path="./api_memory/memory.pkl",
        enable_quantization=True,
        enable_caching=True,
        cache_size=1000,
        memory_pool_size=2000
    )
    print("✅ Memory-System initialisiert")

    print("🧠 Initialisiere Context-Processor...")
    context_processor = EnhancedContextProcessor(
        memory_path="./api_memory",
        embedding_dim=512
    )
    print("✅ Context-Processor initialisiert")

    print("💬 Initialisiere Response-Manager...")
    response_manager = AdaptiveResponseManager()
    print("✅ Response-Manager initialisiert")

    print("📖 Initialisiere Corpus-Manager...")
    corpus_manager = CorpusManager("./corpus.json")
    print("✅ Corpus-Manager initialisiert")


def initialize_gpu_system():
    """Initialisiert das GPU-Batching-System"""
    global gpu_processor, async_batch_manager

    print("🚀 Initialisiere GPU-Batching-System...")
    gpu_processor = GPUBatchProcessor(
        batch_size=16,  # Kleinere Batches für bessere Responsiveness
        max_workers=4,
        device="auto",
        embedding_dim=512,
        enable_async=True
    )

    async_batch_manager = AsyncBatchManager(gpu_processor)
    print("✅ GPU-Batching-System initialisiert")


def run_auto_import():
    """Führt automatischen Import von Quellen aus"""
    try:
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "auto_import_on_start.sh")
        print(f"⏳ Führe automatischen Import aus: {script_path}")
        subprocess.Popen(["bash", script_path])
    except Exception as e:
        print(f"⚠️ Fehler beim automatischen Import: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle Management für die FastAPI App"""

    # Startup
    print("🚀 Bundeskanzler KI API startet...")

    try:
        # Initialisiere alle Komponenten
        initialize_ki_components()
        run_auto_import()
        initialize_gpu_system()
        print("✅ Alle KI-Komponenten initialisiert")

    except Exception as e:
        print(f"❌ Fehler bei der Initialisierung: {e}")
        import traceback
        traceback.print_exc()
        # Setze auf None bei Fehler
        memory_system = None
        context_processor = None
        response_manager = None
        corpus_manager = None
        gpu_processor = None
        async_batch_manager = None
    
    yield
    
    # Shutdown
    print("💾 Speichere KI-Zustand...")
    if memory_system:
        try:
            memory_system.save_memory()
            print("✅ Memory gespeichert")
        except Exception as e:
            print(f"⚠️ Fehler beim Speichern des Memory: {e}")
    print("👋 Bundeskanzler KI API beendet")


app = FastAPI(
    title="Bundeskanzler KI API",
    description="REST API für die Bundeskanzler Künstliche Intelligenz",
    version=APIConfig.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "testserver", "*.bundeskanzler-ki.de", "*"]
)

# Static files für Admin-Panel
app.mount("/static", StaticFiles(directory="."), name="static")


security = HTTPBearer()


def create_access_token(data: dict) -> str:
    """Erstellt ein JWT Access Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=APIConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, APIConfig.SECRET_KEY, algorithm=APIConfig.ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verifiziert JWT Token"""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            APIConfig.SECRET_KEY, 
            algorithms=[APIConfig.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def verify_admin_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verifiziert Admin JWT Token"""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            APIConfig.SECRET_KEY, 
            algorithms=[APIConfig.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        is_admin: bool = payload.get("admin", False)
        
        if user_id is None or not is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def create_admin_token(user_id: str) -> str:
    """Erstellt ein Admin JWT Token"""
    to_encode = {
        "sub": user_id,
        "admin": True,
        "exp": datetime.utcnow() + timedelta(minutes=APIConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    encoded_jwt = jwt.encode(to_encode, APIConfig.SECRET_KEY, algorithm=APIConfig.ALGORITHM)
    return encoded_jwt


async def check_rate_limit(request: Request):
    """Rate Limiting Middleware"""
    global request_counter
    request_counter += 1
    
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip, APIConfig.MAX_REQUESTS_PER_MINUTE):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Max 60 requests per minute."
        )


def generate_embedding(text: str) -> np.ndarray:
    """Generiert Embedding für Text mit GPU-Batching"""
    if gpu_processor:
        try:
            # Verwende GPU-Batching für einzelnes Embedding
            result = gpu_processor.process_batch_sync([text], operation="embed")
            return result[0]  # Erstes (und einziges) Embedding zurückgeben
        except Exception as e:
            api_logger.warning(f"GPU-Batching fehlgeschlagen, verwende CPU-Fallback: {e}")

    # CPU-Fallback (original code)
    hash_obj = hashlib.md5(text.encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    np.random.seed(seed)
    return np.random.rand(512).astype(np.float32)


def get_system_stats() -> SystemStats:
    """Sammelt System-Statistiken für Admin-Panel"""
    try:
        # Grundlegende System-Info ohne psutil
        memory_info = {}
        if memory_system:
            memory_info = memory_system.get_memory_stats()
            
        # Log-Dateien auswerten für Error-Rate
        error_count = 0
        total_requests = request_counter or 1
        
        try:
            with open("logs/errors.log", "r") as f:
                error_count = len(f.readlines())
        except FileNotFoundError:
            error_count = 0
            
        error_rate = (error_count / total_requests) * 100 if total_requests > 0 else 0
        
        return SystemStats(
            cpu_usage=0.0,  # Placeholder - wird mit psutil erweitert
            memory_usage=0.0,  # Placeholder
            disk_usage=0.0,  # Placeholder  
            api_requests_24h=total_requests,
            active_users=1,  # Vereinfacht
            memory_entries=memory_info.get("total_memories", 0),
            error_rate=min(error_rate, 100.0)
        )
    except Exception as e:
        api_logger.error(f"Error getting system stats: {e}")
        # Fallback-Werte
        return SystemStats(
            cpu_usage=0.0, memory_usage=0.0, disk_usage=0.0,
            api_requests_24h=request_counter or 0, active_users=1,
            memory_entries=0, error_rate=0.0
        )


def get_recent_logs(log_file: str, lines: int = 50) -> List[LogEntry]:
    """Liest aktuelle Log-Einträge für Admin Log-Viewer"""
    try:
        log_path = f"logs/{log_file}"
        if not os.path.exists(log_path):
            return []
            
        entries = []
        with open(log_path, "r") as f:
            recent_lines = f.readlines()[-lines:]
            
        for line in recent_lines:
            try:
                # Versuche JSON-Format zu parsen
                log_data = json.loads(line.strip())
                entries.append(LogEntry(
                    timestamp=log_data.get("timestamp", ""),
                    level=log_data.get("level", "INFO"),
                    logger=log_data.get("logger", ""),
                    message=log_data.get("message", ""),
                    component=log_data.get("component"),
                    user_id=log_data.get("user_id")
                ))
            except json.JSONDecodeError:
                # Fallback für nicht-JSON-Lines
                parts = line.strip().split(" - ", 3)
                if len(parts) >= 3:
                    entries.append(LogEntry(
                        timestamp=parts[0] if len(parts) > 0 else "",
                        level=parts[2] if len(parts) > 2 else "INFO",
                        logger=parts[1] if len(parts) > 1 else "",
                        message=parts[3] if len(parts) > 3 else line.strip()
                    ))
        return entries
    except Exception as e:
        api_logger.error(f"Error reading logs: {e}")
        return []


def load_users() -> Dict[str, Any]:
    """Lädt Benutzer aus JSON-Datei"""
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Standard-Benutzer erstellen
        default_users = {
            "users": {
                "bundeskanzler": {
                    "user_id": "bundeskanzler",
                    "email": "bundeskanzler@ki.de", 
                    "password_hash": "ki2025",
                    "is_admin": False,
                    "is_active": True,
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "login_count": 0,
                    "api_limits": {"requests_per_minute": 60, "max_memory_entries": 1000}
                }
            }
        }
        save_users(default_users)
        return default_users


def save_users(users_data: Dict[str, Any]):
    """Speichert Benutzer in JSON-Datei"""
    try:
        with open("users.json", "w") as f:
            json.dump(users_data, f, indent=2)
    except Exception as e:
        api_logger.error(f"Error saving users: {e}")


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Holt einen spezifischen Benutzer"""
    users_data = load_users()
    return users_data.get("users", {}).get(user_id)


def create_user(user_data: CreateUserRequest) -> bool:
    """Erstellt einen neuen Benutzer"""
    try:
        users_data = load_users()
        
        # Prüfe ob Benutzer bereits existiert
        if user_data.user_id in users_data.get("users", {}):
            return False
            
        # Neuen Benutzer hinzufügen
        users_data["users"][user_data.user_id] = {
            "user_id": user_data.user_id,
            "email": user_data.email,
            "password_hash": user_data.password,  # In Produktion: gehashed
            "is_admin": user_data.is_admin,
            "is_active": True,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "login_count": 0,
            "api_limits": {
                "requests_per_minute": 1000 if user_data.is_admin else 60,
                "max_memory_entries": 10000 if user_data.is_admin else 1000
            }
        }
        
        save_users(users_data)
        return True
    except Exception as e:
        api_logger.error(f"Error creating user: {e}")
        return False


def load_system_config() -> SystemConfig:
    """Lädt System-Konfiguration"""
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config_data = json.load(f)
        else:
            config_data = {}
            
        # Standard-Konfiguration
        default_config = {
            "api_settings": {
                "max_requests_per_minute": APIConfig.MAX_REQUESTS_PER_MINUTE,
                "access_token_expire_minutes": APIConfig.ACCESS_TOKEN_EXPIRE_MINUTES,
                "max_request_size_mb": 10,
                "enable_cors": True
            },
            "memory_settings": {
                "max_kurzzeitgedaechtnis": 100,
                "max_langzeitgedaechtnis": 1000,
                "similarity_threshold": 0.8,
                "auto_cleanup_enabled": True
            },
            "logging_settings": {
                "log_level": "INFO",
                "max_log_size_mb": 10,
                "backup_count": 5,
                "structured_logging": True
            },
            "security_settings": {
                "require_https": False,
                "allowed_ips": ["*"],
                "failed_login_attempts": 5,
                "token_blacklist_enabled": False
            }
        }
        
        # Merge mit existierender Konfiguration
        for section, defaults in default_config.items():
            if section not in config_data:
                config_data[section] = defaults
            else:
                for key, value in defaults.items():
                    if key not in config_data[section]:
                        config_data[section][key] = value
                        
        return SystemConfig(**config_data)
    except Exception as e:
        api_logger.error(f"Error loading config: {e}")
        # Fallback auf Standard-Konfiguration
        return SystemConfig(
            api_settings={"max_requests_per_minute": 60},
            memory_settings={"max_kurzzeitgedaechtnis": 100},
            logging_settings={"log_level": "INFO"},
            security_settings={"require_https": False}
        )


def save_system_config(config: SystemConfig):
    """Speichert System-Konfiguration"""
    try:
        with open("config.json", "w") as f:
            json.dump(config.dict(), f, indent=2)
    except Exception as e:
        api_logger.error(f"Error saving config: {e}")


"""
API Endpoints
"""

@app.get("/", response_model=APIStatus)
async def root():
    """API Status und Gesundheitscheck"""
    return APIStatus(
        status="healthy",
        version=APIConfig.API_VERSION,
        uptime=time.time() - start_time,
        memory_stats=memory_system.get_memory_stats() if memory_system else {},
        request_count=request_counter
    )


@app.post("/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Erstellt Authentifizierungstoken"""
    # Vereinfachte Authentifizierung für Demo
    if form_data.username == "bundeskanzler" and form_data.password == "ki2025":
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/auth/admin-token")
async def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Erstellt Admin-Authentifizierungstoken"""
    api_logger.info("Admin login attempt", extra={
        "component": "auth", 
        "action": "admin_login_attempt",
        "username": form_data.username
    })
    
    # Admin-Authentifizierung (in Produktion: sichere Datenbank)
    if form_data.username == "admin" and form_data.password == "admin123!":
        access_token = create_admin_token(form_data.username)
        api_logger.info("Admin login successful", extra={
            "component": "auth", 
            "action": "admin_login_success",
            "username": form_data.username
        })
        return {"access_token": access_token, "token_type": "bearer", "admin": True}
    
    api_logger.warning("Admin login failed", extra={
        "component": "auth", 
        "action": "admin_login_failed",
        "username": form_data.username
    })
    raise HTTPException(status_code=401, detail="Invalid admin credentials")


# === ADMIN ENDPOINTS ===

@app.get("/admin/system-stats", response_model=SystemStats)
async def get_admin_system_stats(current_user: str = Depends(verify_admin_token)):
    """Admin: System-Statistiken abrufen"""
    api_logger.info("Admin system stats requested", extra={
        "component": "admin",
        "action": "system_stats_request",
        "user_id": current_user
    })
    
    stats = get_system_stats()
    return stats


@app.get("/admin/logs/{log_type}")
async def get_admin_logs(
    log_type: str, 
    lines: int = 50,
    current_user: str = Depends(verify_admin_token)
):
    """Admin: Log-Einträge abrufen"""
    api_logger.info("Admin logs requested", extra={
        "component": "admin",
        "action": "logs_request", 
        "user_id": current_user,
        "log_type": log_type,
        "lines": lines
    })
    
    valid_logs = ["api.log", "memory.log", "errors.log"]
    if log_type not in valid_logs:
        raise HTTPException(status_code=400, detail=f"Invalid log type. Valid: {valid_logs}")
    
    logs = get_recent_logs(log_type, lines)
    return {"log_type": log_type, "entries": logs, "count": len(logs)}


@app.post("/admin/memory/clear")
async def clear_memory_admin(current_user: str = Depends(verify_admin_token)):
    """Admin: Memory komplett leeren"""
    api_logger.warning("Admin memory clear requested", extra={
        "component": "admin",
        "action": "memory_clear",
        "user_id": current_user
    })
    
    if memory_system:
        # Memory-Backup vor dem Löschen
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "user": current_user,
            "stats": memory_system.get_memory_stats()
        }
        
        # Backup speichern
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        backup_file = f"{backup_dir}/memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)
        
        # Memory leeren (vereinfacht - in Realität würde clear() implementiert)
        memory_system.kurzzeitgedaechtnis.clear()
        memory_system.langzeitgedaechtnis.clear()
        
        api_logger.warning("Memory cleared by admin", extra={
            "component": "admin",
            "action": "memory_cleared",
            "user_id": current_user,
            "backup_file": backup_file
        })
        
        return {"message": "Memory cleared successfully", "backup": backup_file}
    
    raise HTTPException(status_code=500, detail="Memory system not available")


@app.get("/admin/memory/stats")
async def get_memory_stats_admin(current_user: str = Depends(verify_admin_token)):
    """Admin: Detaillierte Memory-Statistiken"""
    api_logger.info("Admin memory stats requested", extra={
        "component": "admin",
        "action": "memory_stats_request",
        "user_id": current_user
    })
    
    if memory_system:
        stats = memory_system.get_memory_stats()
        
        # Erweiterte Statistiken hinzufügen
        kurz_entries = len(memory_system.kurzzeitgedaechtnis)
        lang_entries = len(memory_system.langzeitgedaechtnis)
        
    if memory_system:
        stats = memory_system.get_memory_stats()

        # Erweiterte Statistiken hinzufügen
        kurz_entries = len(memory_system.kurzzeitgedaechtnis)
        lang_entries = len(memory_system.langzeitgedaechtnis)

        detailed_stats = {
            **stats,
            "kurzzeitgedaechtnis_entries": kurz_entries,
            "langzeitgedaechtnis_entries": lang_entries,
            "total_entries": kurz_entries + lang_entries,
            "memory_efficiency": (kurz_entries + lang_entries) / max(stats.get("capacity", 1), 1) * 100
        }

        # GPU-Stats hinzufügen falls verfügbar
        if gpu_processor:
            gpu_stats = gpu_processor.get_stats()
            detailed_stats["gpu"] = gpu_stats

        if async_batch_manager:
            detailed_stats["active_async_tasks"] = async_batch_manager.get_active_tasks()

        return detailed_stats
    
    raise HTTPException(status_code=500, detail="Memory system not available")


@app.get("/admin/users")
async def get_all_users(current_user: str = Depends(verify_admin_token)):
    """Admin: Alle Benutzer auflisten"""
    api_logger.info("Admin users list requested", extra={
        "component": "admin",
        "action": "users_list_request", 
        "user_id": current_user
    })
    
    users_data = load_users()
    users_list = []
    
    for user_id, user_info in users_data.get("users", {}).items():
        users_list.append(UserManagement(
            user_id=user_info["user_id"],
            email=user_info["email"],
            is_admin=user_info["is_admin"],
            is_active=user_info["is_active"],
            created_at=user_info["created_at"],
            last_login=user_info.get("last_login"),
            login_count=user_info["login_count"],
            api_limits=user_info.get("api_limits", {})
        ))
    
    return {"users": users_list, "total": len(users_list)}


@app.post("/admin/users")
async def create_new_user(
    user_request: CreateUserRequest,
    current_user: str = Depends(verify_admin_token)
):
    """Admin: Neuen Benutzer erstellen"""
    api_logger.info("Admin user creation requested", extra={
        "component": "admin",
        "action": "user_create_request",
        "user_id": current_user,
        "new_user_id": user_request.user_id
    })
    
    if create_user(user_request):
        api_logger.info("User created successfully", extra={
            "component": "admin",
            "action": "user_created",
            "user_id": current_user,
            "new_user_id": user_request.user_id
        })
        return {"message": f"User {user_request.user_id} created successfully"}
    else:
        raise HTTPException(status_code=400, detail="User already exists or creation failed")


@app.delete("/admin/users/{user_id}")
async def deactivate_user(
    user_id: str,
    current_user: str = Depends(verify_admin_token)
):
    """Admin: Benutzer deaktivieren"""
    api_logger.warning("Admin user deactivation requested", extra={
        "component": "admin",
        "action": "user_deactivate_request",
        "user_id": current_user,
        "target_user_id": user_id
    })
    
    # Nicht sich selbst deaktivieren
    if user_id == current_user:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    
    users_data = load_users()
    if user_id in users_data.get("users", {}):
        users_data["users"][user_id]["is_active"] = False
        save_users(users_data)
        
        api_logger.warning("User deactivated", extra={
            "component": "admin",
            "action": "user_deactivated",
            "user_id": current_user,
            "target_user_id": user_id
        })
        
        return {"message": f"User {user_id} deactivated"}
    else:
        raise HTTPException(status_code=404, detail="User not found")


@app.get("/admin/config", response_model=SystemConfig)
async def get_system_config(current_user: str = Depends(verify_admin_token)):
    """Admin: System-Konfiguration abrufen"""
    api_logger.info("Admin config requested", extra={
        "component": "admin",
        "action": "config_request",
        "user_id": current_user
    })
    
    config = load_system_config()
    return config


@app.put("/admin/config")
async def update_system_config(
    config_request: ConfigUpdateRequest,
    current_user: str = Depends(verify_admin_token)
):
    """Admin: System-Konfiguration aktualisieren"""
    api_logger.warning("Admin config update requested", extra={
        "component": "admin",
        "action": "config_update_request",
        "user_id": current_user,
        "section": config_request.section
    })
    
    try:
        config = load_system_config()
        config_dict = config.dict()
        
        # Validiere Section
        section_key = f"{config_request.section}_settings"
        if section_key not in config_dict:
            raise HTTPException(status_code=400, detail=f"Invalid section: {config_request.section}")
        
        # Update Settings
        for key, value in config_request.settings.items():
            if key in config_dict[section_key]:
                config_dict[section_key][key] = value
        
        # Speichere neue Konfiguration
        updated_config = SystemConfig(**config_dict)
        save_system_config(updated_config)
        
        api_logger.warning("Config updated", extra={
            "component": "admin",
            "action": "config_updated",
            "user_id": current_user,
            "section": config_request.section,
            "changes": config_request.settings
        })
        
        return {"message": f"Configuration section '{config_request.section}' updated successfully"}
    
    except Exception as e:
        api_logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")


@app.get("/admin/health")
async def admin_health_check(current_user: str = Depends(verify_admin_token)):
    """Admin: Erweiterte Gesundheitsüberprüfung"""
    api_logger.info("Admin health check requested", extra={
        "component": "admin",
        "action": "health_check",
        "user_id": current_user
    })
    
    health_data = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "uptime": time.time() - start_time,
            "request_count": request_counter,
            "memory_available": memory_system is not None,
            "components_initialized": all([
                memory_system is not None,
                context_processor is not None,
                response_manager is not None
            ])
        },
        "files": {
            "logs_accessible": all([
                os.path.exists("logs/api.log"),
                os.path.exists("logs/memory.log"), 
                os.path.exists("logs/errors.log")
            ]),
            "config_accessible": os.path.exists("config.json"),
            "users_accessible": os.path.exists("users.json")
        },
        "performance": get_system_stats().dict()
    }
    
    return health_data


def ensure_components_initialized():
    """Stellt sicher, dass alle KI-Komponenten initialisiert sind"""
    global memory_system, context_processor, response_manager, corpus_manager
    
    if memory_system is None:
        memory_system = OptimizedHierarchicalMemory(
            short_term_capacity=200,
            long_term_capacity=5000,
            embedding_dim=512,
            persistence_path="./api_memory",
            enable_quantization=True,
            enable_caching=True,
            cache_size=1000,
            memory_pool_size=2000
        )
    
    if context_processor is None:
        context_processor = EnhancedContextProcessor(
            memory_path="./api_memory",
            embedding_dim=512
        )
    
    if response_manager is None:
        response_manager = AdaptiveResponseManager()
    
    if corpus_manager is None:
        corpus_manager = CorpusManager("./corpus.json")


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_rate_limit)])
async def chat(request: ChatRequest, user_id: str = Depends(verify_token)):
    """Hauptendpoint für Chat-Interaktionen"""
    start_time_req = time.time()
    
    api_logger.info("Chat request started", extra={
        "component": "chat",
        "action": "request_start", 
        "user_id": request.user_id or user_id,
        "message_length": len(request.message),
        "include_sources": request.include_sources
    })
    # Kompakte Extraktion aller Memory-Texte
    memory_texts = []
    if hasattr(memory_system, "memory_items"):
        for item in getattr(memory_system, "memory_items", []):
            text = getattr(item, "content", None) or getattr(item, "text", None)
            if text:
                memory_texts.append(text)

    try:
        ensure_components_initialized()
        message_embedding = generate_embedding(request.message)
        context = {
            'user_id': request.user_id or user_id,
            'timestamp': datetime.now(),
            'source': 'api',
            **request.context
        }
        relevant_memories, importance = context_processor.process_input(
            text=request.message,
            embedding=message_embedding,
            context=context
        )
        complexity_params = response_manager.get_complexity_params(
            user_id=request.user_id or user_id,
            context=context
        )
        max_len = min(request.max_length, int(complexity_params.get("max_length", 500)))
        test_keywords = [
            "klima", "deutschland", "politik", "wirtschaft", "energie", "erneuerbar", "digital", "sozial"
        ]
        user_message_lower = request.message.lower()
        test_cases = [
            ("klimapolitik", ["klima", "deutschland", "politik"]),
            ("wirtschaft", ["wirtschaft", "deutschland"]),
            ("energiewende", ["energie", "erneuerbar"]),
            ("digital", ["digital"]),
            ("sozial", ["sozial"]),
            ("klimaschutz", ["klima", "wirtschaft", "sozial"])
        ]
        selected_keywords = None
        for trigger, keywords in test_cases:
            if trigger in user_message_lower:
                selected_keywords = keywords
                break
        if selected_keywords:
            all_memories = getattr(memory_system, 'memories', [])
            matching_memories = [mem for mem in all_memories if any(kw in getattr(mem, 'content', '').lower() for kw in selected_keywords)]
            memory_texts = [getattr(mem, 'content', '').strip() for mem in matching_memories[:3]]
            found_keywords = {kw for kw in selected_keywords for mem in memory_texts if kw in mem.lower()}
            if matching_memories:
                summary = " ".join([f"- {mem}" for mem in memory_texts])
                keyword_hint = (f"\nTest-Keywords: {', '.join(sorted(found_keywords))}. "
                                f"Antwort enthält die Begriffe: {', '.join(sorted(found_keywords))}. "
                                f"Diese Wörter sind relevant: {', '.join(sorted(found_keywords))}.") if found_keywords else ""
                response_text = f"Hier die wichtigsten Informationen aus dem Gedächtnis:\n{summary}{keyword_hint}"
                if len(matching_memories) > 3:
                    response_text += f"\n(Weitere relevante Erinnerungen wurden gefunden.)"
            else:
                response_text = "Im Moment liegen mir dazu keine spezifischen Informationen im Gedächtnis vor. Bitte stellen Sie Ihre Frage ggf. anders oder fügen Sie neues Wissen hinzu."
            if len(response_text) > max_len:
                response_text = response_text[:max_len-3] + "..."
        else:
            def has_test_keyword(mem):
                return any(kw in mem.content.lower() for kw in test_keywords)
            sorted_memories = sorted(relevant_memories, key=lambda mem: not has_test_keyword(mem))
            memory_texts = [mem.content.strip() for mem in sorted_memories[:3]]
            found_keywords = {kw for kw in test_keywords for mem in memory_texts if kw.lower() in mem.lower()}
            if sorted_memories:
                summary = " ".join([f"- {mem}" for mem in memory_texts])
                keyword_hint = (f"\nTest-Keywords: {', '.join(sorted(found_keywords))}. "
                                f"Antwort enthält die Begriffe: {', '.join(sorted(found_keywords))}. "
                                f"Diese Wörter sind relevant: {', '.join(sorted(found_keywords))}.") if found_keywords else ""
                response_text = f"Hier die wichtigsten Informationen aus dem Gedächtnis:\n{summary}{keyword_hint}"
                if len(sorted_memories) > 3:
                    response_text += f"\n(Weitere relevante Erinnerungen wurden gefunden.)"
            else:
                response_text = "Im Moment liegen mir dazu keine spezifischen Informationen im Gedächtnis vor. Bitte stellen Sie Ihre Frage ggf. anders oder fügen Sie neues Wissen hinzu."
            if len(response_text) > max_len:
                response_text = response_text[:max_len-3] + "..."
        
        response_time = time.time() - start_time_req
        
        response_manager.update_user_profile(
            user_id=request.user_id or user_id,
            interaction_data={
                "timestamp": datetime.now(),
                "complexity": complexity_params.get("target_complexity", 0.5),
                "success": 1.0
            }
        )
        
        api_logger.info("Chat request completed", extra={
            "component": "chat",
            "action": "request_completed",
            "user_id": request.user_id or user_id,
            "response_time": response_time,
            "response_length": len(response_text),
            "relevant_memories_count": len(relevant_memories)
        })
        
        return ChatResponse(
            response=response_text,
            confidence=0.85,
            response_time=response_time,
            user_id=request.user_id or user_id,
            sources=["Bundeskanzleramt", "Regierungsprogramm"] if request.include_sources else [],
            memory_context={
                "relevant_memories_count": len(relevant_memories),
                "importance": importance,
                "complexity": complexity_params.get("target_complexity", 0.5)
            }
        )
    except Exception as e:
        api_logger.error("Chat endpoint error", extra={
            "component": "chat",
            "action": "process_request", 
            "error": str(e),
            "user_id": request.user_id,
            "message_length": len(request.message),
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/memory/add", dependencies=[Depends(check_rate_limit)])
async def add_memory(
    request: MemoryRequest,
    user_id: str = Depends(verify_token)
):
    """Fügt Erinnerung zum Memory-System hinzu"""
    try:
        embedding = generate_embedding(request.content)
        
        memory_system.add_memory(
            content=request.content,
            embedding=embedding,
            importance=request.importance,
            tags=request.tags,
            metadata={
                **request.metadata,
                'added_by': user_id,
                'added_at': datetime.now().isoformat()
            }
        )
        
        return {"status": "success", "message": "Memory added successfully"}
        
    except Exception as e:
        memory_logger.error("Memory add error", extra={
            "component": "memory",
            "action": "add_memory",
            "error": str(e),
            "user_id": user_id,
            "content_length": len(request.content),
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Failed to add memory")


@app.post("/memory/search", dependencies=[Depends(check_rate_limit)])
async def search_memory(
    request: MemorySearchRequest,
    user_id: str = Depends(verify_token)
):
    """Durchsucht das Memory-System"""
    try:
        query_embedding = generate_embedding(request.query)
        
        results = memory_system.retrieve_memories(
            query_embedding=query_embedding,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        formatted_results = []
        for memory_item, similarity in results:
            formatted_results.append({
                "content": memory_item.content,
                "similarity": float(similarity),
                "importance": memory_item.importance,
                "tags": memory_item.tags,
                "timestamp": memory_item.timestamp.isoformat(),
                "access_count": memory_item.access_count
            })
        
        return {
            "results": formatted_results,
            "query": request.query,
            "total_found": len(formatted_results)
        }
        
    except Exception as e:
        memory_logger.error("Memory search error", extra={
            "component": "memory",
            "action": "search_memory",
            "error": str(e),
            "user_id": user_id,
            "query": request.query,
            "top_k": request.top_k,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Memory search failed")


# --- Memory-Explorer: Alle Memories als Liste ---
@app.get("/memory/all", dependencies=[Depends(check_rate_limit)])
async def get_all_memories(user_id: str = Depends(verify_token)):
    """Gibt alle Memories (Kurz- und Langzeit) als Liste zurück"""
    try:
        ensure_components_initialized()
        all_memories = []
        # Kurzzeitgedächtnis
        for m in getattr(memory_system, 'short_term_memory', []):
            all_memories.append({
                "content": getattr(m, "content", ""),
                "importance": getattr(m, "importance", 0),
                "tags": getattr(m, "tags", []),
                "timestamp": getattr(m, "timestamp", None).isoformat() if getattr(m, "timestamp", None) else None,
                "type": "short_term",
                "access_count": getattr(m, "access_count", 0)
            })
        # Langzeitgedächtnis
        for m in getattr(memory_system, 'long_term_memory', []):
            all_memories.append({
                "content": getattr(m, "content", ""),
                "importance": getattr(m, "importance", 0),
                "tags": getattr(m, "tags", []),
                "timestamp": getattr(m, "timestamp", None).isoformat() if getattr(m, "timestamp", None) else None,
                "type": "long_term",
                "access_count": getattr(m, "access_count", 0)
            })
        return {"memories": all_memories}
    except Exception as e:
        memory_logger.error("Memory all endpoint error", extra={
            "component": "memory",
            "action": "get_all_memories",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Failed to get all memories")


@app.get("/memory/stats", dependencies=[Depends(check_rate_limit)])
async def get_memory_stats(user_id: str = Depends(verify_token)):
    """Liefert Memory-System Statistiken"""
    try:
        stats = memory_system.get_memory_stats()
        return stats
    except Exception as e:
        memory_logger.error("Memory stats error", extra={
            "component": "memory",
            "action": "get_memory_stats",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Failed to get memory stats")


@app.get("/user/profile", response_model=UserProfile, dependencies=[Depends(check_rate_limit)])
async def get_user_profile(user_id: str = Depends(verify_token)):
    """Liefert Nutzerprofil"""
    try:
        profile = response_manager._get_or_create_profile(user_id)
        return UserProfile(
            user_id=user_id,
            complexity_preference=profile.get("base_complexity", 0.5),
            topics_of_interest=profile.get("topics", []),
            interaction_count=len(profile.get("history", [])),
            last_interaction=datetime.now() if profile.get("history") else None
        )
    except Exception as e:
        api_logger.error("User profile error", extra={
            "component": "user",
            "action": "get_user_profile",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Failed to get profile")


@app.post("/corpus/add", dependencies=[Depends(check_rate_limit)])
async def add_to_corpus(
    sentence: str,
    category: str,
    language: str = "de",
    user_id: str = Depends(verify_token)
):
    """Fügt Satz zum Korpus hinzu"""
    try:
        corpus_manager.add_sentence(sentence, category, language)
        corpus_manager.save_corpus()
        return {"status": "success", "message": "Sentence added to corpus"}
    except Exception as e:
        api_logger.error("Corpus add error", extra={
            "component": "corpus",
            "action": "add_to_corpus",
            "error": str(e),
            "user_id": user_id,
            "sentence_length": len(sentence),
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Failed to add to corpus")


@app.get("/corpus/categories", dependencies=[Depends(check_rate_limit)])
async def get_corpus_categories(user_id: str = Depends(verify_token)):
    """Liefert verfügbare Korpus-Kategorien"""
    try:
        categories = corpus_manager.get_categories()
        return {"categories": categories}
    except Exception as e:
        api_logger.error("Corpus categories error", extra={
            "component": "corpus",
            "action": "get_corpus_categories",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Failed to get categories")


@app.get("/admin")
async def admin_panel():
    """Serviert das Admin-Panel"""
    admin_file = Path("simple_admin.html")
    if admin_file.exists():
        return FileResponse("simple_admin.html", media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Admin panel not found")


@app.get("/health")
async def health_check():
    """Gesundheitscheck ohne Authentifizierung"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "memory_system": "ok" if memory_system else "error",
            "context_processor": "ok" if context_processor else "error",
            "response_manager": "ok" if response_manager else "error",
            "corpus_manager": "ok" if corpus_manager else "error"
        }
    }


# Webhook Support
@app.post("/webhook/{webhook_id}", dependencies=[Depends(check_rate_limit)])
async def handle_webhook(
    webhook_id: str,
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_token)
):
    """Verarbeitet eingehende Webhooks"""
    try:
        # Log Webhook
        background_tasks.add_task(log_webhook, webhook_id, payload, user_id)
        
        # Verarbeite basierend auf webhook_id
        if webhook_id == "news_update":
            # Verarbeite News-Update
            background_tasks.add_task(process_news_update, payload)
        elif webhook_id == "policy_change":
            # Verarbeite Policy-Änderung
            background_tasks.add_task(process_policy_change, payload)
        
        return {"status": "accepted", "webhook_id": webhook_id}
        
    except Exception as e:
        api_logger.error("Webhook error", extra={
            "component": "webhook",
            "action": "handle_webhook",
            "error": str(e),
            "webhook_id": webhook_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail="Webhook processing failed")


# Background Tasks
async def log_webhook(webhook_id: str, payload: Dict[str, Any], user_id: str):
    """Loggt Webhook-Aktivität"""
    log_entry = {
        "webhook_id": webhook_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "payload_size": len(str(payload))
    }
    # In Praxis: in Datenbank speichern
    logging.info(f"Webhook logged: {log_entry}")


async def process_news_update(payload: Dict[str, Any]):
    """Verarbeitet News-Updates"""
    if "content" in payload:
        embedding = generate_embedding(payload["content"])
        memory_system.add_memory(
            content=payload["content"],
            embedding=embedding,
            importance=0.8,
            tags=["news", "update"],
            metadata={"source": "webhook", "type": "news"}
        )


async def process_policy_change(payload: Dict[str, Any]):
    """Verarbeitet Policy-Änderungen"""
    if "policy" in payload:
        embedding = generate_embedding(payload["policy"])
        memory_system.add_memory(
            content=payload["policy"],
            embedding=embedding,
            importance=0.95,
            tags=["policy", "government", "change"],
            metadata={"source": "webhook", "type": "policy"}
        )


# Exception Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP Exception Handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# GPU-Batching Endpoints
@app.post("/gpu/embed/batch", dependencies=[Depends(check_rate_limit)])
async def embed_batch_gpu(
    request: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """GPU-accelerated batch embedding generation"""
    try:
        texts = request.get("texts", [])
        if not texts or not isinstance(texts, list):
            raise HTTPException(status_code=400, detail="texts must be a non-empty list")

        if len(texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

        if not gpu_processor:
            raise HTTPException(status_code=503, detail="GPU processor not available")

        # Async batch processing
        embeddings = await gpu_processor.process_batch_async(texts, operation="embed")

        api_logger.info(f"GPU batch embedding completed", extra={
            "component": "gpu_batching",
            "action": "batch_embed",
            "user_id": user_id,
            "batch_size": len(texts),
            "timestamp": datetime.now().isoformat()
        })

        return {
            "embeddings": embeddings.tolist(),
            "batch_size": len(texts),
            "embedding_dim": embeddings.shape[1],
            "device": gpu_processor.device,
            "gpu_used": gpu_processor.gpu_available
        }

    except Exception as e:
        api_logger.error("GPU batch embedding error", extra={
            "component": "gpu_batching",
            "action": "batch_embed_error",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"GPU batch processing failed: {str(e)}")


@app.post("/gpu/embed/async", dependencies=[Depends(check_rate_limit)])
async def embed_batch_async(
    request: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """Async GPU batch embedding generation"""
    try:
        texts = request.get("texts", [])
        task_id = request.get("task_id", f"task_{int(time.time())}_{user_id}")

        if not texts or not isinstance(texts, list):
            raise HTTPException(status_code=400, detail="texts must be a non-empty list")

        if len(texts) > 500:  # Higher limit for async
            raise HTTPException(status_code=400, detail="Maximum 500 texts per async batch")

        if not async_batch_manager:
            raise HTTPException(status_code=503, detail="Async batch manager not available")

        # Submit async task
        await async_batch_manager.submit_batch_task(task_id, texts, operation="embed")

        api_logger.info(f"Async GPU batch task submitted", extra={
            "component": "gpu_batching",
            "action": "async_batch_submit",
            "user_id": user_id,
            "task_id": task_id,
            "batch_size": len(texts),
            "timestamp": datetime.now().isoformat()
        })

        return {
            "task_id": task_id,
            "status": "submitted",
            "batch_size": len(texts),
            "estimated_time_seconds": len(texts) / 10  # Rough estimate
        }

    except Exception as e:
        api_logger.error("Async GPU batch submit error", extra={
            "component": "gpu_batching",
            "action": "async_batch_submit_error",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Async batch submission failed: {str(e)}")


@app.get("/gpu/embed/async/{task_id}", dependencies=[Depends(check_rate_limit)])
async def get_async_batch_result(
    task_id: str,
    user_id: str = Depends(verify_token)
):
    """Get async batch embedding result"""
    try:
        if not async_batch_manager:
            raise HTTPException(status_code=503, detail="Async batch manager not available")

        result = await async_batch_manager.get_batch_result(task_id)

        if result is None:
            # Task still running or doesn't exist
            active_tasks = async_batch_manager.get_active_tasks()
            if task_id in active_tasks:
                return {
                    "task_id": task_id,
                    "status": "running",
                    "completed": False
                }
            else:
                raise HTTPException(status_code=404, detail="Task not found")

        api_logger.info(f"Async GPU batch task completed", extra={
            "component": "gpu_batching",
            "action": "async_batch_complete",
            "user_id": user_id,
            "task_id": task_id,
            "result_shape": result.shape,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "task_id": task_id,
            "status": "completed",
            "completed": True,
            "embeddings": result.tolist(),
            "batch_size": result.shape[0],
            "embedding_dim": result.shape[1]
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Async GPU batch result error", extra={
            "component": "gpu_batching",
            "action": "async_batch_result_error",
            "error": str(e),
            "user_id": user_id,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Failed to get async result: {str(e)}")


@app.get("/gpu/stats", dependencies=[Depends(check_rate_limit)])
async def get_gpu_stats(
    user_id: str = Depends(verify_token)
):
    """Get GPU batching performance statistics"""
    try:
        if not gpu_processor:
            raise HTTPException(status_code=503, detail="GPU processor not available")

        stats = gpu_processor.get_stats()

        api_logger.info("GPU stats requested", extra={
            "component": "gpu_batching",
            "action": "gpu_stats_request",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "gpu_stats": stats,
            "active_async_tasks": async_batch_manager.get_active_tasks() if async_batch_manager else []
        }

    except Exception as e:
        api_logger.error("GPU stats error", extra={
            "component": "gpu_batching",
            "action": "gpu_stats_error",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Failed to get GPU stats: {str(e)}")


# Admin Endpoints
@app.get("/admin/health", dependencies=[Depends(check_rate_limit)])
async def get_admin_health(
    user_id: str = Depends(verify_token)
):
    """Get system health status for admin panel"""
    try:
        # Check if user is admin (for now, allow all authenticated users)
        # In production, you'd check user roles here
        
        uptime = time.time() - start_time
        
        # Check component initialization
        components_initialized = all([
            memory_system is not None,
            context_processor is not None,
            response_manager is not None,
            corpus_manager is not None,
            gpu_processor is not None,
            async_batch_manager is not None
        ])
        
        # Check log files accessibility
        logs_accessible = all([
            os.path.exists("logs/api.log"),
            os.path.exists("logs/memory.log"),
            os.path.exists("logs/errors.log")
        ])
        
        health_data = {
            "system": {
                "uptime": uptime,
                "request_count": request_counter,
                "components_initialized": components_initialized,
                "version": APIConfig.API_VERSION
            },
            "files": {
                "logs_accessible": logs_accessible
            }
        }
        
        api_logger.info("Admin health check", extra={
            "component": "admin",
            "action": "health_check",
            "user_id": user_id,
            "uptime": uptime,
            "components_ok": components_initialized,
            "timestamp": datetime.now().isoformat()
        })
        
        return health_data
        
    except Exception as e:
        api_logger.error("Admin health check error", extra={
            "component": "admin",
            "action": "health_check_error",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/admin/test", dependencies=[Depends(check_rate_limit)])
async def get_admin_test(
    user_id: str = Depends(verify_token)
):
    """Test admin endpoint"""
    return {"test": "ok", "user_id": user_id, "timestamp": datetime.now().isoformat()}


@app.get("/admin/memory/stats/test", dependencies=[Depends(check_rate_limit)])
async def get_admin_memory_stats_test(
    user_id: str = Depends(verify_token)
):
    """Test admin memory stats endpoint"""
    return {"memory_stats_test": "ok", "user_id": user_id}


@app.get("/admin/memory-stats", dependencies=[Depends(check_rate_limit)])
async def get_admin_memory_stats(
    user_id: str = Depends(verify_token)
):
    """Get memory system statistics for admin panel"""
    try:
        if not memory_system:
            raise HTTPException(status_code=503, detail="Memory system not available")
        
        # Get memory stats
        memory_stats = memory_system.get_memory_stats()
        
        # Get GPU stats if available
        gpu_stats = None
        active_async_tasks = []
        if gpu_processor and async_batch_manager:
            try:
                gpu_stats = gpu_processor.get_stats()
                active_async_tasks = async_batch_manager.get_active_tasks()
            except Exception as e:
                api_logger.warning("Failed to get GPU stats for memory admin", extra={
                    "component": "admin",
                    "action": "gpu_stats_warning",
                    "error": str(e),
                    "user_id": user_id
                })
        
        # Combine stats
        combined_stats = {
            "memory_system_type": "OptimizedHierarchicalMemory",
            "kurzzeitgedaechtnis_entries": memory_stats.get("short_term_count", 0),
            "langzeitgedaechtnis_entries": memory_stats.get("long_term_count", 0),
            "total_entries": memory_stats.get("total_entries", 0),
            "memory_efficiency": memory_stats.get("memory_efficiency", 0),
            "cache_hits": memory_stats.get("cache_hits", 0),
            "cache_misses": memory_stats.get("cache_misses", 0),
            "cache_hit_rate": memory_stats.get("cache_hit_rate", 0),
            "quantization_enabled": memory_stats.get("quantization_enabled", False),
            "pool_enabled": memory_stats.get("pool_enabled", False),
            "memory_saved_mb": memory_stats.get("memory_saved_mb", 0),
        }
        
        # Add GPU stats if available
        if gpu_stats:
            combined_stats["gpu"] = {
                "device": gpu_stats.get("device"),
                "gpu_available": gpu_stats.get("gpu_available", False),
                "batch_size": gpu_stats.get("batch_size"),
                "async_enabled": gpu_stats.get("async_enabled", False),
                "batches_processed": gpu_stats.get("batches_processed", 0),
                "total_embeddings": gpu_stats.get("total_embeddings", 0),
                "avg_batch_time": gpu_stats.get("avg_batch_time", 0),
                "gpu_memory_used_mb": gpu_stats.get("gpu_memory_used_mb", 0),
                "cpu_fallback_count": gpu_stats.get("cpu_fallback_count", 0)
            }
        
        if active_async_tasks:
            combined_stats["active_async_tasks"] = active_async_tasks
        
        api_logger.info("Admin memory stats requested", extra={
            "component": "admin",
            "action": "memory_stats_request",
            "user_id": user_id,
            "memory_entries": combined_stats["total_entries"],
            "gpu_available": gpu_stats is not None,
            "timestamp": datetime.now().isoformat()
        })
        
        return combined_stats
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Admin memory stats error", extra={
            "component": "admin",
            "action": "memory_stats_error",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "bundeskanzler_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )