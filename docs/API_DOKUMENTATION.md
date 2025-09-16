# 📡 API Dokumentation - Bundeskanzler KI

**Version 4.0.0** - RESTful API für RTX 2070 GPU-optimierte Bundeskanzler KI

## 🚀 **API Übersicht**

Die Bundeskanzler KI bietet eine vollständige RESTful API für die Integration in externe Anwendungen. Die API ist FastAPI-basiert und bietet automatische Dokumentation mit RTX 2070 GPU-Optimierung.

### 🌐 **Base URL**
```
http://localhost:8000
```

### 📋 **Verfügbare Endpunkte**
- **POST** `/chat` - Haupt-Chat-Endpunkt (RTX 2070 KI)
- **POST** `/api/chat` - Alternative Chat-Route
- **GET** `/health` - System-Status & GPU-Monitoring
- **GET** `/docs` - Automatische API-Dokumentation
- **GET** `/redoc` - Alternative API-Dokumentation
- **GET** `/system/status` - Detaillierte System-Informationen

## 🔧 **API starten**

### 💻 **Via Start-Script**
```bash
./start_ki.sh
# Wählen Sie Option 3: API Server
```

### ⚡ **Direkt starten**
```bash
# RTX 2070 API-Server starten
uvicorn core/bundeskanzler_api.py:app --host 0.0.0.0 --port 8000 --reload

# Mit GPU-Monitoring
uvicorn core/bundeskanzler_api.py:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

### 🌐 **Zugriff**
- **API**: http://localhost:8000
- **Dokumentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **System-Status**: http://localhost:8000/system/status

## 📋 **Endpunkt-Details**

### 💬 **Chat Endpunkt**

#### **POST /chat**
Hauptendpunkt für Chat-Anfragen mit RTX 2070 GPU-optimierter KI.

**Request Body:**
```json
{
  "message": "Was ist die Klimapolitik der Bundesregierung?",
  "use_rag": true,
  "max_tokens": 500,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "Deutschland setzt sich für innovative Klimaschutzmaßnahmen ein...",
  "confidence": 74.2,
  "sources": 3,
  "processing_time": 2.1,
  "topic": "klimapolitik",
  "timestamp": "2025-09-15T20:30:45Z"
}
```

#### **curl Beispiel:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Was ist die Klimapolitik der Bundesregierung?",
    "use_rag": true
  }'
```

#### **Python Beispiel:**
```python
import requests
import json

url = "http://localhost:8000/chat"
data = {
    "message": "Was ist die Klimapolitik der Bundesregierung?",
    "use_rag": True
}

response = requests.post(url, json=data)
result = response.json()

print(f"Antwort: {result['response']}")
print(f"Konfidenz: {result['confidence']}%")
```

### 📊 **Health Check Endpunkt**

#### **GET /health**
System-Status und Verfügbarkeit prüfen.

**Response:**
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "rag_system": "operational",
  "gpu_available": true,
  "corpus_size": 75,
  "uptime": "0:15:30"
}
```

#### **curl Beispiel:**
```bash
curl -X GET "http://localhost:8000/health"
```

### 🔍 **RAG Search Endpunkt**

#### **POST /search**
Direkte Suche in der Wissensbasis.

**Request Body:**
```json
{
  "query": "Klimapolitik",
  "top_k": 5,
  "min_score": 0.5
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Deutschland setzt sich für innovative Klimaschutz ein.",
      "score": 0.742,
      "source": "bundestag",
      "topic": "klimapolitik"
    }
  ],
  "total_results": 5,
  "processing_time": 0.3
}
```

## 🛠️ **Erweiterte API-Nutzung**

### 🔧 **Mit Authentifizierung**
```python
import requests

headers = {
    "Authorization": "Bearer YOUR_API_TOKEN",
    "Content-Type": "application/json"
}

data = {"message": "Ihre Frage"}
response = requests.post("http://localhost:8000/chat", json=data, headers=headers)
```

### 📊 **Batch-Anfragen**
```python
import requests
import asyncio
import aiohttp

async def batch_questions(questions):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for question in questions:
            task = session.post(
                "http://localhost:8000/chat",
                json={"message": question}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return [await resp.json() for resp in responses]

# Verwendung
questions = [
    "Was ist die Klimapolitik?",
    "Welche Wirtschaftsreformen sind geplant?",
    "Wie steht Deutschland zur EU?"
]

results = asyncio.run(batch_questions(questions))
```

### 🔄 **WebSocket Support**
```python
import websocket

def on_message(ws, message):
    print(f"Empfangen: {message}")

def on_error(ws, error):
    print(f"Fehler: {error}")

# WebSocket-Verbindung
ws = websocket.WebSocketApp(
    "ws://localhost:8000/ws",
    on_message=on_message,
    on_error=on_error
)

ws.run_forever()
```

## 📋 **Request/Response Schema**

### 💬 **Chat Request Schema**
```json
{
  "type": "object",
  "properties": {
    "message": {
      "type": "string",
      "description": "Die Benutzerfrage",
      "maxLength": 1000
    },
    "use_rag": {
      "type": "boolean",
      "description": "RAG-System verwenden",
      "default": true
    },
    "max_tokens": {
      "type": "integer",
      "description": "Maximale Antwort-Länge",
      "default": 500,
      "minimum": 50,
      "maximum": 1000
    },
    "temperature": {
      "type": "number",
      "description": "Kreativität der Antwort",
      "default": 0.7,
      "minimum": 0.0,
      "maximum": 1.0
    }
  },
  "required": ["message"]
}
```

### 💡 **Chat Response Schema**
```json
{
  "type": "object",
  "properties": {
    "response": {
      "type": "string",
      "description": "KI-Antwort"
    },
    "confidence": {
      "type": "number",
      "description": "Vertrauenswert in %",
      "minimum": 0,
      "maximum": 100
    },
    "sources": {
      "type": "integer",
      "description": "Anzahl verwendeter Quellen"
    },
    "processing_time": {
      "type": "number",
      "description": "Verarbeitungszeit in Sekunden"
    },
    "topic": {
      "type": "string",
      "description": "Erkanntes Thema"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "Zeitstempel der Antwort"
    }
  }
}
```

## 🚦 **HTTP Status Codes**

| Code | Bedeutung | Beschreibung |
|------|-----------|--------------|
| **200** | OK | Erfolgreiche Anfrage |
| **400** | Bad Request | Ungültige Anfrage |
| **404** | Not Found | Endpunkt nicht gefunden |
| **422** | Unprocessable Entity | Validation Error |
| **500** | Internal Server Error | Server-Fehler |
| **503** | Service Unavailable | Service nicht verfügbar |

## 🔧 **Fehlerbehandlung**

### ❌ **Error Response Format**
```json
{
  "error": "Bad Request",
  "message": "Message field is required",
  "code": 400,
  "timestamp": "2025-09-15T20:30:45Z"
}
```

### 🛠️ **Python Error Handling**
```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": "Test"}
    )
    response.raise_for_status()  # Wirft Exception bei HTTP-Fehlern
    
    data = response.json()
    print(data["response"])
    
except requests.exceptions.HTTPError as e:
    print(f"HTTP-Fehler: {e}")
except requests.exceptions.ConnectionError:
    print("Verbindungsfehler: API-Server nicht erreichbar")
except requests.exceptions.Timeout:
    print("Timeout: Anfrage dauerte zu lange")
except Exception as e:
    print(f"Unbekannter Fehler: {e}")
```

## 📊 **Performance & Limits**

### ⚡ **Rate Limiting**
- **Requests**: 100 pro Minute pro IP
- **Concurrent**: 10 gleichzeitige Verbindungen
- **Message Size**: Max 1000 Zeichen

### 📈 **Performance Metriken**
- **Antwortzeit**: ~2-5 Sekunden
- **Durchsatz**: ~20 Anfragen/Minute
- **Verfügbarkeit**: 99.9% Uptime

### 🔧 **Optimierung**
```python
# Session für bessere Performance
import requests

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

# Wiederverwendung der Session
for question in questions:
    response = session.post(
        "http://localhost:8000/chat",
        json={"message": question}
    )
```

## 🧪 **Testing**

### 🔧 **Unit Tests**
```python
import pytest
import requests

def test_chat_endpoint():
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": "Test"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "confidence" in data

def test_health_endpoint():
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
```

### 📊 **Load Testing**
```bash
# Mit curl und parallel
echo "Was ist die Klimapolitik?" | parallel -j 10 -n 0 \
  curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "{}"}'
```

## 🌐 **Integration Beispiele**

### 🐍 **Python Client**
```python
class BundeskanzlerKIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def ask(self, question, use_rag=True):
        response = self.session.post(
            f"{self.base_url}/chat",
            json={"message": question, "use_rag": use_rag}
        )
        return response.json()
    
    def health(self):
        response = self.session.get(f"{self.base_url}/health")
        return response.json()

# Verwendung
client = BundeskanzlerKIClient()
result = client.ask("Was ist die Klimapolitik?")
print(result["response"])
```

### 🌐 **JavaScript/Node.js**
```javascript
const axios = require('axios');

class BundeskanzlerKIClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async ask(question, useRag = true) {
        try {
            const response = await axios.post(`${this.baseUrl}/chat`, {
                message: question,
                use_rag: useRag
            });
            return response.data;
        } catch (error) {
            throw new Error(`API Error: ${error.message}`);
        }
    }
    
    async health() {
        const response = await axios.get(`${this.baseUrl}/health`);
        return response.data;
    }
}

// Verwendung
const client = new BundeskanzlerKIClient();
client.ask('Was ist die Klimapolitik?')
    .then(result => console.log(result.response))
    .catch(error => console.error(error));
```

### 🌐 **Frontend Integration (React)**
```jsx
import { useState, useEffect } from 'react';

function BundeskanzlerKIChat() {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    
    const askQuestion = async () => {
        setLoading(true);
        try {
            const result = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: question })
            });
            const data = await result.json();
            setResponse(data.response);
        } catch (error) {
            setResponse('Fehler beim Laden der Antwort');
        }
        setLoading(false);
    };
    
    return (
        <div>
            <input 
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ihre Frage an die Bundeskanzler KI"
            />
            <button onClick={askQuestion} disabled={loading}>
                {loading ? 'Lädt...' : 'Fragen'}
            </button>
            {response && <div>{response}</div>}
        </div>
    );
}
```

---

**Entwickelt für die deutsche politische Bildung 🇩🇪**

*Letztes Update: 15. September 2025*