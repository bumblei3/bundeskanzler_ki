#!/usr/bin/env python3
"""
🚀 RTX 2070-Optimierte Bundeskanzler-KI Integration
==================================================

Integriert RTX 2070-optimierte LLM und RAG-Komponenten
in das bestehende Multi-Agent-System

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

# RTX 2070 optimierte Komponenten
from .rtx2070_llm_manager import generate_llm_response, get_rtx2070_llm_manager
from .rtx2070_rag_system import create_rtx2070_rag_system, rtx2070_rag_query

# Bestehende Komponenten
try:
    from .gpu_manager import RTX2070Manager

    GPUManager = RTX2070Manager
except ImportError:
    try:
        from gpu_manager import RTX2070Manager

        GPUManager = RTX2070Manager
    except ImportError:
        GPUManager = None

# Multilingual Support
try:
    from .multilingual_manager import get_multilingual_manager

    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False

try:
    from .multi_agent_system import AgentType, MultiAgentSystem
except ImportError:
    from multi_agent_system import AgentType, MultiAgentSystem

try:
    from .rag_system import RAGSystem
except ImportError:
    from rag_system import RAGSystem

# Faktencheck-System
try:
    from .fact_checker import FactChecker, get_fact_checker

    FACT_CHECK_AVAILABLE = True
except ImportError:
    FACT_CHECK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RTX2070Config:
    """Konfiguration für RTX 2070 optimierte KI"""

    enable_llm_integration: bool = True
    enable_rag_optimization: bool = True
    llm_model_preference: str = "auto"  # "mistral_7b", "llama2_7b", "german_gpt2", "auto"
    rag_hybrid_search: bool = True
    dynamic_model_loading: bool = True
    vram_safety_margin_gb: float = 1.0  # Sicherheitsreserve für VRAM


class RTX2070BundeskanzlerKI:
    """
    RTX 2070 optimierte Bundeskanzler-KI mit integrierten LLM und RAG
    """

    def __init__(self, config: RTX2070Config = None):
        self.config = config or RTX2070Config()

        # RTX 2070 GPU Manager
        try:
            self.gpu_manager = GPUManager()
        except Exception as e:
            logger.warning(f"GPU Manager nicht verfügbar: {e}")
            self.gpu_manager = None

        # RTX 2070 optimierte Komponenten
        self.llm_manager = None
        self.rag_system = None

        # Bestehende Komponenten (Fallback)
        self.multi_agent_system = MultiAgentSystem()
        self.fallback_rag = RAGSystem()

        # Multilingual Support
        self.multilingual_manager = None
        if MULTILINGUAL_AVAILABLE:
            try:
                self.multilingual_manager = get_multilingual_manager()
                logger.info("🌍 Multilingual Support aktiviert")
            except Exception as e:
                logger.warning(f"❌ Multilingual Support fehlgeschlagen: {e}")

        # Faktencheck-System
        self.fact_checker = None
        if FACT_CHECK_AVAILABLE:
            try:
                self.fact_checker = get_fact_checker()
                logger.info("🔍 Faktencheck-System aktiviert")
            except Exception as e:
                logger.warning(f"❌ Faktencheck-System fehlgeschlagen: {e}")

        # Initialisierung
        self._initialize_rtx2070_components()

        logger.info("🚀 RTX 2070 Bundeskanzler-KI initialisiert")

    def _initialize_rtx2070_components(self):
        """Initialisiert RTX 2070 optimierte Komponenten"""
        try:
            # VRAM prüfen
            if self.gpu_manager:
                gpu_stats = self.gpu_manager.get_gpu_stats()
                if gpu_stats:
                    available_vram = gpu_stats.memory_total_gb - gpu_stats.memory_used_gb

                    if available_vram < self.config.vram_safety_margin_gb:
                        logger.warning(f"⚠️ Wenig VRAM verfügbar: {available_vram:.1f} GB")
                        self.config.enable_llm_integration = False
                else:
                    logger.warning("⚠️ GPU Stats nicht verfügbar - deaktiviere LLM Integration")
                    self.config.enable_llm_integration = False

            # LLM Manager initialisieren
            if self.config.enable_llm_integration:
                self.llm_manager = get_rtx2070_llm_manager(self.gpu_manager)
                logger.info("✅ RTX 2070 LLM Manager initialisiert")

            # RAG System initialisieren
            if self.config.enable_rag_optimization:
                self.rag_system = create_rtx2070_rag_system()
                if not self.rag_system.build_index():
                    logger.warning(
                        "⚠️ RTX 2070 RAG konnte nicht initialisiert werden, verwende Fallback"
                    )
                    self.rag_system = None
                else:
                    logger.info("✅ RTX 2070 RAG System initialisiert")

        except Exception as e:
            logger.error(f"❌ Fehler bei RTX 2070 Initialisierung: {e}")
            # Fallback auf bestehende Komponenten
            self.config.enable_llm_integration = False
            self.config.enable_rag_optimization = False

    def query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Kompatibilitätsmethode für bestehende Aufrufe"""
        return self.process_query(query, context)

    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        RTX 2070 optimierte Query-Verarbeitung mit Multilingual Support
        """
        context = context or {}
        original_query = query

        try:
            # Mehrsprachige Verarbeitung
            query_language = "de"  # Standard: Deutsch
            translated_query = query

            if self.multilingual_manager:
                try:
                    # Sprache der Query erkennen (vereinfacht ohne Übersetzung)
                    detected_lang, confidence = self.multilingual_manager.detect_language(query)

                    # Keine Übersetzung mehr - DeepL wurde entfernt
                    query_language = detected_lang.value if detected_lang else "de"
                    translated_query = query  # Keine Übersetzung

                    context["original_query"] = original_query
                    context["query_language"] = query_language
                    context["translated_query"] = translated_query

                    logger.info(
                        f"🌍 Sprache erkannt: {query_language} (Vertrauen: {confidence:.2f})"
                    )

                except Exception as e:
                    logger.warning(f"⚠️ Mehrsprachige Verarbeitung fehlgeschlagen: {e}")
                    query_language = "de"
                    translated_query = query

            # Query-Komplexität analysieren (mit übersetzter Query)
            query_complexity = self._analyze_query_complexity(translated_query)

            # RAG-Kontext abrufen
            rag_context = self._get_rag_context(translated_query, query_complexity)

            # Agenten-Routing (mit übersetzter Query)
            agent_response = self._route_to_agents(translated_query, rag_context, query_complexity)

            # LLM-Enhancement falls aktiviert
            if self.config.enable_llm_integration and query_complexity in ["complex", "expert"]:
                enhanced_response = self._enhance_with_llm(
                    translated_query, agent_response, rag_context
                )
            else:
                enhanced_response = agent_response

            # Antwort zurück in Originalsprache übersetzen falls nötig
            final_response = enhanced_response
            # Übersetzung nicht mehr verfügbar - DeepL wurde entfernt
            if query_language != "de":
                logger.info(f"🌍 Antwort bleibt auf Deutsch (Übersetzung nicht verfügbar)")

            return {
                "query": original_query,
                "translated_query": (
                    translated_query if translated_query != original_query else None
                ),
                "response": final_response,
                "original_response": (
                    enhanced_response if final_response != enhanced_response else None
                ),
                "rag_context": rag_context,
                "query_complexity": query_complexity,
                "query_language": query_language,
                "components_used": self._get_components_status(),
                "processing_time": None,  # Könnte später hinzugefügt werden
            }

        except Exception as e:
            logger.error(f"❌ Fehler bei Query-Verarbeitung: {e}")
            return {
                "query": query,
                "response": f"❌ Verarbeitungsfehler: {str(e)}",
                "error": str(e),
                "components_used": self._get_components_status(),
            }

    def process_query_with_fact_check(
        self, query: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        RTX 2070 optimierte Query-Verarbeitung mit integriertem Faktencheck
        """
        context = context or {}
        original_query = query

        try:
            # Normale Query-Verarbeitung durchführen
            base_result = self.process_query(query, context)

            # Faktencheck nur durchführen wenn verfügbar und erfolgreiche Antwort
            if self.fact_checker and "error" not in base_result:
                response_text = base_result.get("response", "")

                # Query-Kategorie bestimmen
                query_category = self._determine_query_category(query)

                # Antwort validieren
                fact_check_result = self.fact_checker.validate_response(
                    response_text, query_category
                )

                # Faktencheck-Ergebnisse zur Antwort hinzufügen
                base_result.update(
                    {
                        "fact_check": {
                            "overall_confidence": fact_check_result["overall_confidence"],
                            "total_statements": fact_check_result["total_statements"],
                            "accurate_statements": fact_check_result["accurate_statements"],
                            "sources_used": fact_check_result["sources_used"],
                            "validation_details": fact_check_result["validation_details"],
                        },
                        "confidence_score": fact_check_result["overall_confidence"],
                        "sources_cited": fact_check_result["sources_used"],
                    }
                )

                # Warnung hinzufügen bei niedriger Konfidenz
                if fact_check_result["overall_confidence"] < 0.7:
                    base_result["response"] += (
                        "\n\n⚠️ **Hinweis:** Diese Information hat eine moderate Konfidenz. "
                        "Bitte konsultieren Sie die angegebenen Quellen für aktuelle Details."
                    )

                logger.info(
                    f"🔍 Faktencheck abgeschlossen: {fact_check_result['overall_confidence']:.2f} Konfidenz"
                )

            else:
                # Fallback ohne Faktencheck
                base_result.update(
                    {
                        "fact_check": None,
                        "confidence_score": 0.5,  # Standard-Konfidenz
                        "sources_cited": 0,
                    }
                )
                logger.info("⚠️ Faktencheck nicht verfügbar - verwende Standard-Konfidenz")

            return base_result

        except Exception as e:
            logger.error(f"❌ Fehler bei faktencheck-validierter Query-Verarbeitung: {e}")
            return {
                "query": query,
                "response": f"❌ Verarbeitungsfehler: {str(e)}",
                "error": str(e),
                "fact_check": None,
                "confidence_score": 0.0,
                "components_used": self._get_components_status(),
            }

    def _determine_query_category(self, query: str) -> str:
        """Bestimmt die Kategorie einer Query für den Faktencheck"""
        query_lower = query.lower()

        if any(
            word in query_lower for word in ["klima", "umwelt", "energie", "co2", "erneuerbare"]
        ):
            return "climate"
        elif any(
            word in query_lower
            for word in ["wirtschaft", "finanzen", "arbeitsmarkt", "inflation", "eu"]
        ):
            return "economy"
        elif any(
            word in query_lower for word in ["politik", "bundestag", "gesetz", "regierung", "wahl"]
        ):
            return "politics"
        else:
            return "general"

    def _analyze_query_complexity(self, query: str) -> str:
        """Analysiert die Komplexität der Query"""
        query_length = len(query.split())

        # Einfache Heuristiken für RTX 2070
        if query_length < 5:
            return "simple"
        elif query_length < 15:
            return "moderate"
        elif any(
            keyword in query.lower() for keyword in ["politik", "gesetz", "bundestag", "regierung"]
        ):
            return "expert"
        else:
            return "complex"

    def _get_rag_context(self, query: str, complexity: str) -> str:
        """Ruft RAG-Kontext ab (RTX 2070 optimiert oder Fallback)"""
        try:
            if self.rag_system and self.config.enable_rag_optimization:
                # RTX 2070 optimierte RAG-Abfrage
                rag_result = self.rag_system.rag_query(
                    query, top_k=3 if complexity == "simple" else 5
                )
                return rag_result.get("context", "")
            else:
                # Fallback auf bestehendes RAG-System
                rag_result = self.fallback_rag.rag_answer(query)
                return rag_result.get("context", "")

        except Exception as e:
            logger.error(f"❌ RAG-Fehler: {e}")
            return ""

    def _route_to_agents(self, query: str, context: str, complexity: str) -> str:
        """Routed Query an entsprechende Agenten"""
        try:
            # Multi-Agent-System verwenden (async call)
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Context als Dict übergeben
            context_dict = {"rag_context": context, "complexity": complexity} if context else None

            response = loop.run_until_complete(self.multi_agent_system.query(query, context_dict))
            loop.close()

            return response.primary_response

        except Exception as e:
            logger.error(f"❌ Agent-Routing Fehler: {e}")
            return f"Fehler bei der Agent-Verarbeitung: {e}"

    def _select_agent(self, query: str) -> AgentType:
        """Wählt passenden Agenten basierend auf Query-Inhalt"""
        query_lower = query.lower()

        # Einfache Keyword-basierte Agenten-Auswahl
        if any(
            word in query_lower for word in ["bundestag", "gesetz", "politik", "wahl", "partei"]
        ):
            return AgentType.POLITIK
        elif any(
            word in query_lower
            for word in ["wirtschaft", "finanzen", "arbeitsmarkt", "inflation", "eu"]
        ):
            return AgentType.WIRTSCHAFT
        elif any(
            word in query_lower for word in ["klima", "energie", "umwelt", "co2", "erneuerbare"]
        ):
            return AgentType.KLIMA
        else:
            return AgentType.COORDINATOR  # Koordinator für allgemeine Fragen

    def _enhance_with_llm(self, query: str, agent_response: str, rag_context: str) -> str:
        """Erweitert Antwort mit RTX 2070 LLM"""
        try:
            if not self.llm_manager:
                return agent_response

            # LLM Prompt zusammenstellen
            llm_prompt = f"""
Basierend auf der folgenden Kontext-Information und Agenten-Antwort,
formuliere eine kohärente und hilfreiche Antwort auf die Frage.

Kontext: {rag_context[:1000]}...

Agenten-Antwort: {agent_response}

Frage: {query}

Antwort:
"""

            # RTX 2070 optimierte LLM-Generierung
            llm_response = generate_llm_response(
                prompt=llm_prompt,
                context=rag_context,
                model_preference=self.config.llm_model_preference,
            )

            # Kombiniere Agenten- und LLM-Antwort
            if llm_response and not llm_response.startswith("❌"):
                return f"{agent_response}\n\n🤖 KI-Erweiterung: {llm_response}"
            else:
                return agent_response

        except Exception as e:
            logger.error(f"❌ LLM Enhancement Fehler: {e}")
            return agent_response

    def _get_components_status(self) -> Dict[str, bool]:
        """Gibt Status der verwendeten Komponenten zurück"""
        return {
            "rtx2070_llm": self.llm_manager is not None and self.config.enable_llm_integration,
            "rtx2070_rag": self.rag_system is not None and self.config.enable_rag_optimization,
            "multilingual_support": self.multilingual_manager is not None,
            "fact_check_system": self.fact_checker is not None,
            "deepl_available": False,  # DeepL wurde entfernt
            "google_translate_available": False,  # Google Translate wurde entfernt
            "multi_agent_system": True,  # Immer verfügbar
            "fallback_rag": True,  # Immer verfügbar
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Gibt System-Informationen zurück"""
        gpu_info = {}
        if self.gpu_manager and hasattr(self.gpu_manager, "get_gpu_stats"):
            gpu_stats = self.gpu_manager.get_gpu_stats()
            if gpu_stats:
                gpu_info = gpu_stats.__dict__
            else:
                gpu_info = {"error": "GPU stats nicht verfügbar"}
        elif torch.cuda.is_available():
            gpu_info = {
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "device": "cuda",
            }
        else:
            gpu_info = {"device": "cpu"}

        llm_info = {}
        if self.llm_manager:
            llm_info = self.llm_manager.get_model_info()

        rag_info = {}
        if self.rag_system:
            rag_info = self.rag_system.get_system_info()
        elif self.fallback_rag:
            # Fallback RAG Info
            rag_info = {
                "corpus_loaded": (
                    len(self.fallback_rag.corpus_entries) > 0
                    if hasattr(self.fallback_rag, "corpus_entries")
                    else False
                ),
                "corpus_entries": (
                    len(self.fallback_rag.corpus_entries)
                    if hasattr(self.fallback_rag, "corpus_entries")
                    else 0
                ),
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "embedding_dimension": 384,
            }

        multi_agent_info = {}
        if hasattr(self.multi_agent_system, "get_system_info"):
            multi_agent_info = self.multi_agent_system.get_system_info()
        else:
            # Basis-Info für Multi-Agent-System
            multi_agent_info = {"agents_count": 0, "agents_initialized": 0, "system_active": True}

        multilingual_info = {}
        if self.multilingual_manager:
            # Vereinfachte Info für SimpleMultilingualManager
            multilingual_info = {
                "supported_languages": ["de", "en"],
                "translation_available": False,
                "deepl_available": False,
            }

        fact_check_info = {}
        if self.fact_checker:
            fact_check_info = self.fact_checker.get_system_info()

        return {
            "config": self.config.__dict__,
            "gpu_info": gpu_info,
            "llm_info": llm_info,
            "rag_info": rag_info,
            "multi_agent_info": multi_agent_info,
            "multilingual_info": multilingual_info,
            "fact_check_info": fact_check_info,
            "components_status": self._get_components_status(),
            "rtx2070_optimized": True,
        }


# Globale Instanz für einfachen Zugriff
_rtx2070_ki_instance = None


def get_rtx2070_bundeskanzler_ki(config: RTX2070Config = None) -> RTX2070BundeskanzlerKI:
    """Factory-Funktion für RTX 2070 KI"""
    global _rtx2070_ki_instance
    if _rtx2070_ki_instance is None:
        _rtx2070_ki_instance = RTX2070BundeskanzlerKI(config)
    return _rtx2070_ki_instance


def rtx2070_process_query(query: str, **kwargs) -> Dict[str, Any]:
    """Kompatibilitätsfunktion für bestehende Aufrufe"""
    ki = get_rtx2070_bundeskanzler_ki()
    return ki.process_query(query, kwargs)


def rtx2070_process_query_with_fact_check(query: str, **kwargs) -> Dict[str, Any]:
    """Kompatibilitätsfunktion für faktencheck-validierte Queries"""
    ki = get_rtx2070_bundeskanzler_ki()
    return ki.process_query_with_fact_check(query, kwargs)


if __name__ == "__main__":
    # Test der RTX 2070 Bundeskanzler-KI
    print("🚀 RTX 2070 Bundeskanzler-KI Test")

    # KI initialisieren
    ki = get_rtx2070_bundeskanzler_ki()

    # System-Info anzeigen
    info = ki.get_system_info()
    print(f"GPU VRAM: {info['gpu_info'].get('memory_total_gb', 'N/A')} GB")
    print(f"LLM aktiviert: {info['components_status']['rtx2070_llm']}")
    print(f"RAG optimiert: {info['components_status']['rtx2070_rag']}")

    # Test-Abfrage
    test_query = "Was ist die Bedeutung der Energiewende für Deutschland?"
    result = ki.process_query(test_query)

    print(f"\nTest-Abfrage: {test_query}")
    print(f"Komplexität: {result['query_complexity']}")
    print(f"Antwort-Länge: {len(result['response'])} Zeichen")
    print(f"Kontext-Länge: {len(result['rag_context'])} Zeichen")

    print(f"\nAntwort (gekürzt): {result['response'][:300]}...")
