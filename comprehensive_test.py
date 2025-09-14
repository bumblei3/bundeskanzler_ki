"""
Umfassender Test der Bundeskanzler KI
Testet alle Hauptfunktionen, Integrationsfähigkeiten und Edge Cases
"""
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import sys


API_BASE_URL = "http://localhost:8000"
USERNAME = "bundeskanzler"
PASSWORD = "ki2025"


class BundeskanzlerKITester:
    """Comprehensive Tester für die Bundeskanzler KI"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.access_token = None
        self.session = None
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Loggt Testergebnis"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"
        
        result = f"{status} {test_name}"
        if details:
            result += f" - {details}"
        
        print(result)
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    async def authenticate(self) -> bool:
        """Authentifizierung bei der API"""
        print("\n🔐 AUTHENTIFIZIERUNG")
        print("=" * 50)
        
        try:
            data = aiohttp.FormData()
            data.add_field('username', USERNAME)
            data.add_field('password', PASSWORD)
            
            async with self.session.post(f"{self.base_url}/auth/token", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.access_token = result["access_token"]
                    self.log_test("Authentifizierung", True, f"Token erhalten: {self.access_token[:20]}...")
                    return True
                else:
                    self.log_test("Authentifizierung", False, f"Status: {response.status}")
                    return False
        except Exception as e:
            self.log_test("Authentifizierung", False, f"Exception: {e}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Erstellt Authorization Headers"""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def test_api_health(self):
        """Test API-Gesundheit und Status"""
        print("\n🏥 API-GESUNDHEIT & STATUS")
        print("=" * 50)
        
        # Health Check
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test("Health Check", True, f"Status: {data['status']}")
                else:
                    self.log_test("Health Check", False, f"Status: {response.status}")
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {e}")
        
        # API Status
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test("API Status", True, f"Version: {data['version']}, Uptime: {data['uptime']:.1f}s")
                else:
                    self.log_test("API Status", False, f"Status: {response.status}")
        except Exception as e:
            self.log_test("API Status", False, f"Exception: {e}")
    
    async def test_chat_conversations(self):
        """Test verschiedene Chat-Szenarien"""
        print("\n💬 CHAT-KONVERSATIONEN")
        print("=" * 50)
        
        test_conversations = [
            {
                "name": "Klimapolitik Grundfrage",
                "message": "Wie steht es um die Klimapolitik in Deutschland?",
                "expected_keywords": ["klima", "deutschland", "politik"]
            },
            {
                "name": "Wirtschaftliche Lage",
                "message": "Wie entwickelt sich die deutsche Wirtschaft?",
                "expected_keywords": ["wirtschaft", "deutschland"]
            },
            {
                "name": "Energiewende",
                "message": "Können Sie mir Details zur Energiewende erklären?",
                "expected_keywords": ["energie", "erneuerbar"]
            },
            {
                "name": "Digitalisierung",
                "message": "Welche Digitalisierungsmaßnahmen plant die Regierung?",
                "expected_keywords": ["digital"]
            },
            {
                "name": "Soziale Themen",
                "message": "Was unternimmt die Regierung für soziale Gerechtigkeit?",
                "expected_keywords": ["sozial"]
            },
            {
                "name": "Komplexe Frage",
                "message": "Wie hängen Klimaschutz, Wirtschaft und soziale Gerechtigkeit zusammen?",
                "expected_keywords": ["klima", "wirtschaft", "sozial"]
            }
        ]
        
        for conv in test_conversations:
            await self.test_single_chat(conv)
            await asyncio.sleep(0.5)  # Rate limiting respect
    
    async def test_single_chat(self, conversation: Dict[str, Any]):
        """Test einzelne Chat-Konversation"""
        chat_request = {
            "message": conversation["message"],
            "user_id": "test_user",
            "max_length": 300,
            "include_sources": True
        }
        
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/chat",
                headers=self.get_headers(),
                json=chat_request
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '').lower()
                    
                    # Check if response contains expected keywords
                    keyword_found = any(
                        keyword.lower() in response_text 
                        for keyword in conversation.get("expected_keywords", [])
                    )
                    
                    confidence = data.get('confidence', 0)
                    
                    if keyword_found and confidence > 0.5:
                        self.log_test(
                            conversation["name"], 
                            True, 
                            f"Konfidenz: {confidence:.2f}, Zeit: {response_time:.2f}s"
                        )
                    else:
                        self.log_test(
                            conversation["name"], 
                            False, 
                            f"Keywords fehlen oder niedrige Konfidenz: {confidence:.2f}"
                        )
                else:
                    self.log_test(conversation["name"], False, f"HTTP {response.status}")
        
        except Exception as e:
            self.log_test(conversation["name"], False, f"Exception: {e}")
    
    async def test_memory_system(self):
        """Test Memory-System Funktionalität"""
        print("\n🧠 MEMORY-SYSTEM")
        print("=" * 50)
        
        # Memory hinzufügen
        test_memories = [
            {
                "content": "Deutschland hat sich verpflichtet, bis 2045 klimaneutral zu werden",
                "importance": 0.9,
                "tags": ["klimaschutz", "2045", "klimaneutralität"],
                "metadata": {"source": "test", "year": 2025}
            },
            {
                "content": "Die Bundesregierung investiert 200 Milliarden Euro in erneuerbare Energien",
                "importance": 0.8,
                "tags": ["energie", "investition", "erneuerbar"],
                "metadata": {"source": "test", "amount": 200000000000}
            },
            {
                "content": "Neue Elektromobilitäts-Initiative mit 100.000 Ladesäulen geplant",
                "importance": 0.7,
                "tags": ["elektromobilität", "ladesäulen", "initiative"],
                "metadata": {"source": "test", "count": 100000}
            }
        ]
        
        # Memories hinzufügen
        for i, memory in enumerate(test_memories):
            try:
                async with self.session.post(
                    f"{self.base_url}/memory/add",
                    headers=self.get_headers(),
                    json=memory
                ) as response:
                    
                    success = response.status == 200
                    self.log_test(
                        f"Memory hinzufügen #{i+1}", 
                        success,
                        memory["content"][:50] + "..." if success else f"Status: {response.status}"
                    )
            except Exception as e:
                self.log_test(f"Memory hinzufügen #{i+1}", False, f"Exception: {e}")
        
        # Memory-Suche
        search_queries = [
            {"query": "Klimaschutz 2045", "expected_results": 1},
            {"query": "erneuerbare Energie Investition", "expected_results": 1},
            {"query": "Elektromobilität Ladesäulen", "expected_results": 1},
            {"query": "Nachhaltigkeit", "expected_results": 2}  # Sollte mehrere finden
        ]
        
        for search in search_queries:
            await self.test_memory_search(search)
        
        # Memory-Statistiken
        await self.test_memory_stats()
    
    async def test_memory_search(self, search_config: Dict[str, Any]):
        """Test Memory-Suche"""
        search_request = {
            "query": search_config["query"],
            "top_k": 5,
            "min_similarity": 0.1
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/memory/search",
                headers=self.get_headers(),
                json=search_request
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    found_count = data.get('total_found', 0)
                    expected = search_config.get('expected_results', 1)
                    
                    success = found_count >= expected
                    self.log_test(
                        f"Memory-Suche: '{search_config['query']}'",
                        success,
                        f"Gefunden: {found_count}, Erwartet: ≥{expected}"
                    )
                else:
                    self.log_test(
                        f"Memory-Suche: '{search_config['query']}'",
                        False,
                        f"HTTP {response.status}"
                    )
        except Exception as e:
            self.log_test(
                f"Memory-Suche: '{search_config['query']}'",
                False,
                f"Exception: {e}"
            )
    
    async def test_memory_stats(self):
        """Test Memory-Statistiken"""
        try:
            async with self.session.get(
                f"{self.base_url}/memory/stats",
                headers=self.get_headers()
            ) as response:
                
                if response.status == 200:
                    stats = await response.json()
                    total = stats.get('total_memories', 0)
                    short_term = stats.get('short_term_count', 0)
                    long_term = stats.get('long_term_count', 0)
                    
                    success = total > 0
                    self.log_test(
                        "Memory-Statistiken",
                        success,
                        f"Gesamt: {total}, Kurz: {short_term}, Lang: {long_term}"
                    )
                else:
                    self.log_test("Memory-Statistiken", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Memory-Statistiken", False, f"Exception: {e}")
    
    async def test_user_management(self):
        """Test Nutzerprofil-Management"""
        print("\n👤 NUTZERPROFIL-MANAGEMENT")
        print("=" * 50)
        
        try:
            async with self.session.get(
                f"{self.base_url}/user/profile",
                headers=self.get_headers()
            ) as response:
                
                if response.status == 200:
                    profile = await response.json()
                    user_id = profile.get('user_id')
                    interactions = profile.get('interaction_count', 0)
                    
                    success = user_id is not None
                    self.log_test(
                        "Nutzerprofil abrufen",
                        success,
                        f"User: {user_id}, Interaktionen: {interactions}"
                    )
                else:
                    self.log_test("Nutzerprofil abrufen", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Nutzerprofil abrufen", False, f"Exception: {e}")
    
    async def test_corpus_management(self):
        """Test Korpus-Management"""
        print("\n📚 KORPUS-MANAGEMENT")
        print("=" * 50)
        
        # Sätze hinzufügen
        test_sentences = [
            ("Deutschland führt bei der Entwicklung von Wasserstofftechnologie", "technologie", "de"),
            ("Die Bildungsreform stärkt digitale Kompetenzen in Schulen", "bildung", "de"),
            ("Internationale Zusammenarbeit beim Klimaschutz wird ausgebaut", "international", "de")
        ]
        
        for sentence, category, language in test_sentences:
            try:
                params = {
                    "sentence": sentence,
                    "category": category,
                    "language": language
                }
                
                async with self.session.post(
                    f"{self.base_url}/corpus/add",
                    headers=self.get_headers(),
                    params=params
                ) as response:
                    
                    success = response.status == 200
                    self.log_test(
                        f"Korpus-Satz hinzufügen ({category})",
                        success,
                        sentence[:40] + "..." if success else f"HTTP {response.status}"
                    )
            except Exception as e:
                self.log_test(f"Korpus-Satz hinzufügen ({category})", False, f"Exception: {e}")
        
        # Kategorien abrufen
        try:
            async with self.session.get(
                f"{self.base_url}/corpus/categories",
                headers=self.get_headers()
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    categories = data.get('categories', [])
                    
                    success = len(categories) > 0
                    self.log_test(
                        "Korpus-Kategorien abrufen",
                        success,
                        f"Gefunden: {len(categories)} Kategorien: {', '.join(categories[:5])}"
                    )
                else:
                    self.log_test("Korpus-Kategorien abrufen", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Korpus-Kategorien abrufen", False, f"Exception: {e}")
    
    async def test_webhooks(self):
        """Test Webhook-Funktionalität"""
        print("\n🔗 WEBHOOK-INTEGRATION")
        print("=" * 50)
        
        # News Update Webhook
        news_payload = {
            "content": "Bundeskanzler kündigt neue KI-Strategie für Deutschland an",
            "source": "Test News Agency",
            "timestamp": datetime.now().isoformat(),
            "importance": "high"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/webhook/news_update",
                headers=self.get_headers(),
                json=news_payload
            ) as response:
                
                success = response.status == 200
                webhook_id = ""
                if success:
                    data = await response.json()
                    webhook_id = data.get('webhook_id', '')
                
                self.log_test(
                    "News-Update Webhook",
                    success,
                    f"Webhook ID: {webhook_id}" if success else f"HTTP {response.status}"
                )
        except Exception as e:
            self.log_test("News-Update Webhook", False, f"Exception: {e}")
        
        # Policy Change Webhook
        policy_payload = {
            "policy": "Neue KI-Regulierung für den öffentlichen Sektor",
            "effective_date": "2025-01-01",
            "department": "Digitalisierungsministerium",
            "budget": 500000000
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/webhook/policy_change",
                headers=self.get_headers(),
                json=policy_payload
            ) as response:
                
                success = response.status == 200
                webhook_id = ""
                if success:
                    data = await response.json()
                    webhook_id = data.get('webhook_id', '')
                
                self.log_test(
                    "Policy-Change Webhook",
                    success,
                    f"Webhook ID: {webhook_id}" if success else f"HTTP {response.status}"
                )
        except Exception as e:
            self.log_test("Policy-Change Webhook", False, f"Exception: {e}")
    
    async def test_error_handling(self):
        """Test Error-Handling und Edge Cases"""
        print("\n⚠️ ERROR-HANDLING & EDGE CASES")
        print("=" * 50)
        
        # Ungültiger Token
        invalid_headers = {
            "Authorization": "Bearer invalid_token_12345",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat",
                headers=invalid_headers,
                json={"message": "Test mit ungültigem Token"}
            ) as response:
                
                # Erwarten 401 Unauthorized
                success = response.status == 401
                self.log_test(
                    "Ungültiger Token abgelehnt",
                    success,
                    f"Status: {response.status} (erwartet: 401)"
                )
        except Exception as e:
            self.log_test("Ungültiger Token abgelehnt", False, f"Exception: {e}")
        
        # Leere Chat-Nachricht
        try:
            async with self.session.post(
                f"{self.base_url}/chat",
                headers=self.get_headers(),
                json={"message": ""}
            ) as response:
                
                # Erwarten 422 Validation Error
                success = response.status == 422
                self.log_test(
                    "Leere Nachricht validiert",
                    success,
                    f"Status: {response.status} (erwartet: 422)"
                )
        except Exception as e:
            self.log_test("Leere Nachricht validiert", False, f"Exception: {e}")
        
        # Zu lange Nachricht
        try:
            async with self.session.post(
                f"{self.base_url}/chat",
                headers=self.get_headers(),
                json={"message": "x" * 1001}  # Über 1000 Zeichen
            ) as response:
                
                # Erwarten 422 Validation Error
                success = response.status == 422
                self.log_test(
                    "Zu lange Nachricht abgelehnt",
                    success,
                    f"Status: {response.status} (erwartet: 422)"
                )
        except Exception as e:
            self.log_test("Zu lange Nachricht abgelehnt", False, f"Exception: {e}")
    
    async def test_performance(self):
        """Test Performance und Antwortzeiten"""
        print("\n⚡ PERFORMANCE & ANTWORTZEITEN")
        print("=" * 50)
        
        response_times = []
        
        # Multiple Chat-Requests für Performance-Test
        test_messages = [
            "Was ist die aktuelle Klimapolitik?",
            "Wie entwickelt sich die Wirtschaft?",
            "Welche Digitalisierungsmaßnahmen gibt es?",
            "Was plant die Regierung für Bildung?",
            "Wie steht es um die Energiewende?"
        ]
        
        for i, message in enumerate(test_messages):
            start_time = time.time()
            
            try:
                async with self.session.post(
                    f"{self.base_url}/chat",
                    headers=self.get_headers(),
                    json={"message": message, "max_length": 100}
                ) as response:
                    
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    success = response.status == 200 and response_time < 5.0  # Max 5 Sekunden
                    self.log_test(
                        f"Performance Test #{i+1}",
                        success,
                        f"Zeit: {response_time:.2f}s"
                    )
            except Exception as e:
                self.log_test(f"Performance Test #{i+1}", False, f"Exception: {e}")
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Durchschnittliche Antwortzeit
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            success = avg_time < 3.0  # Durchschnitt unter 3 Sekunden
            self.log_test(
                "Durchschnittliche Antwortzeit",
                success,
                f"Ø: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s"
            )
    
    async def run_all_tests(self):
        """Führt alle Tests aus"""
        print("🎯 BUNDESKANZLER KI - UMFASSENDER TEST")
        print("=" * 80)
        print(f"Startzeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API Endpoint: {self.base_url}")
        print("=" * 80)
        
        # Authentifizierung
        if not await self.authenticate():
            print("❌ Authentifizierung fehlgeschlagen - Tests beendet")
            return
        
        # Alle Testsuiten
        await self.test_api_health()
        await self.test_chat_conversations()
        await self.test_memory_system()
        await self.test_user_management()
        await self.test_corpus_management()
        await self.test_webhooks()
        await self.test_error_handling()
        await self.test_performance()
        
        # Zusammenfassung
        self.print_summary()
    
    def print_summary(self):
        """Druckt Test-Zusammenfassung"""
        print("\n" + "=" * 80)
        print("📊 TEST-ZUSAMMENFASSUNG")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"📈 Gesamtergebnis:")
        print(f"   Gesamt:        {self.total_tests} Tests")
        print(f"   ✅ Erfolgreich: {self.passed_tests} Tests")
        print(f"   ❌ Fehlgeschlagen: {self.failed_tests} Tests")
        print(f"   📊 Erfolgsquote: {success_rate:.1f}%")
        
        if success_rate >= 90:
            overall_status = "🎉 EXZELLENT"
        elif success_rate >= 80:
            overall_status = "✅ GUT"
        elif success_rate >= 70:
            overall_status = "⚠️ AKZEPTABEL"
        else:
            overall_status = "❌ PROBLEMATISCH"
        
        print(f"\n🏆 Gesamtbewertung: {overall_status}")
        
        # Fehlerhafte Tests anzeigen
        if self.failed_tests > 0:
            print(f"\n❌ Fehlgeschlagene Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   • {result['test']}: {result['details']}")
        
        print("\n" + "=" * 80)
        print(f"🏁 Test beendet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


async def main():
    """Hauptfunktion"""
    print("🚀 Starte umfassenden Test der Bundeskanzler KI...")
    
    try:
        async with BundeskanzlerKITester() as tester:
            await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n👋 Test durch Nutzer beendet")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())