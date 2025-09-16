#!/usr/bin/env python3
"""
🚀 RTX 2070-Optimierte KI Tests
==============================

Tests für RTX 2070-optimierte LLM und RAG-Komponenten

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from core.rtx2070_bundeskanzler_ki import (
    RTX2070BundeskanzlerKI,
    RTX2070Config,
    get_rtx2070_bundeskanzler_ki,
)

# RTX 2070 Komponenten
from core.rtx2070_llm_manager import RTX2070LLMManager, get_rtx2070_llm_manager
from core.rtx2070_rag_system import RTX2070OptimizedRAG, create_rtx2070_rag_system


class TestRTX2070LLMManager:
    """Tests für RTX 2070 LLM Manager"""

    @pytest.fixture
    def mock_gpu_manager(self):
        """Mock GPU Manager für Tests"""
        mock_manager = MagicMock()
        mock_stats = MagicMock()
        mock_stats.memory_total_gb = 8.0
        mock_stats.memory_used_gb = 2.0
        mock_manager.get_gpu_stats.return_value = mock_stats
        return mock_manager

    def test_llm_manager_initialization(self, mock_gpu_manager):
        """Test LLM Manager Initialisierung"""
        manager = RTX2070LLMManager(mock_gpu_manager)

        assert manager.gpu_manager == mock_gpu_manager
        assert manager.device in ["cuda", "cpu"]
        assert manager.current_model is None

    def test_model_selection_simple(self, mock_gpu_manager):
        """Test automatische Modell-Auswahl für einfache Queries"""
        manager = RTX2070LLMManager(mock_gpu_manager)

        model = manager.select_optimal_model("simple")
        assert model in ["german_gpt2", "cpu_fallback"]

    def test_model_selection_complex(self, mock_gpu_manager):
        """Test automatische Modell-Auswahl für komplexe Queries"""
        manager = RTX2070LLMManager(mock_gpu_manager)

        model = manager.select_optimal_model("complex")
        assert model in ["mistral_7b", "llama2_7b", "cpu_fallback"]

    @patch("rtx2070_llm_manager.torch.cuda.is_available", return_value=False)
    def test_cpu_fallback(self, mock_cuda, mock_gpu_manager):
        """Test CPU-Fallback bei fehlender GPU"""
        manager = RTX2070LLMManager(mock_gpu_manager)

        # CPU-Modell sollte ausgewählt werden
        model = manager.select_optimal_model("complex")
        assert model == "cpu_fallback"

    def test_get_model_info(self, mock_gpu_manager):
        """Test Modell-Info Abruf"""
        manager = RTX2070LLMManager(mock_gpu_manager)

        info = manager.get_model_info()
        assert "available_vram_gb" in info
        assert "device" in info


class TestRTX2070RAGSystem:
    """Tests für RTX 2070 RAG System"""

    @pytest.fixture
    def temp_corpus(self):
        """Temporärer Test-Corpus"""
        corpus_data = [
            {"text": "Deutschland hat ehrgeizige Klimaziele bis 2030.", "id": 1},
            {"text": "Die Energiewende ist ein zentrales Projekt der Regierung.", "id": 2},
            {"text": "Die Bundesregierung setzt auf erneuerbare Energien.", "id": 3},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(corpus_data, f)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_rag_initialization(self, temp_corpus):
        """Test RAG System Initialisierung"""
        rag = RTX2070OptimizedRAG(corpus_path=temp_corpus)

        assert rag.corpus_path == temp_corpus
        assert rag.embeddings is not None
        assert rag.index is None  # Noch nicht geladen

    def test_corpus_loading(self, temp_corpus):
        """Test Corpus Laden"""
        rag = RTX2070OptimizedRAG(corpus_path=temp_corpus)

        success = rag.load_corpus()
        assert success
        assert len(rag.corpus) == 3
        assert len(rag.id_to_text) == 3

    @patch("rtx2070_rag_system.torch.cuda.is_available", return_value=False)
    def test_rag_index_building_cpu(self, mock_cuda, temp_corpus):
        """Test Index-Erstellung auf CPU"""
        rag = RTX2070OptimizedRAG(corpus_path=temp_corpus)

        success = rag.build_index(force_rebuild=True)
        assert success
        assert rag.index is not None

    def test_hybrid_search(self, temp_corpus):
        """Test Hybrid Search Funktionalität"""
        rag = RTX2070OptimizedRAG(corpus_path=temp_corpus)

        # Index erstellen
        rag.build_index(force_rebuild=True)

        # Suche durchführen
        results = rag.hybrid_search("Energiewende", top_k=2)

        assert isinstance(results, list)
        if results:  # Nur prüfen wenn Ergebnisse vorhanden
            assert len(results) <= 2
            for result in results:
                assert "id" in result
                assert "text" in result
                assert "score" in result

    def test_rag_query(self, temp_corpus):
        """Test vollständige RAG-Abfrage"""
        rag = RTX2070OptimizedRAG(corpus_path=temp_corpus)

        # Index erstellen
        rag.build_index(force_rebuild=True)

        # Abfrage durchführen
        result = rag.rag_query("Was ist die Energiewende?")

        assert "query" in result
        assert "context" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)


class TestRTX2070BundeskanzlerKI:
    """Tests für RTX 2070 Bundeskanzler-KI Integration"""

    @pytest.fixture
    def rtx_config(self):
        """RTX 2070 Konfiguration für Tests"""
        return RTX2070Config(
            enable_llm_integration=False,  # Für Tests deaktivieren
            enable_rag_optimization=True,
            vram_safety_margin_gb=0.5,
        )

    def test_ki_initialization(self, rtx_config):
        """Test KI Initialisierung"""
        ki = RTX2070BundeskanzlerKI(rtx_config)

        assert ki.config == rtx_config
        assert ki.multi_agent_system is not None
        assert ki.fallback_rag is not None

    def test_query_complexity_analysis(self, rtx_config):
        """Test Query-Komplexitätsanalyse"""
        ki = RTX2070BundeskanzlerKI(rtx_config)

        # Einfache Query
        complexity = ki._analyze_query_complexity("Hallo")
        assert complexity == "simple"

        # Komplexe Query
        complexity = ki._analyze_query_complexity(
            "Was ist die Bedeutung der Klimapolitik für Deutschland?"
        )
        assert complexity in ["moderate", "complex", "expert"]

    def test_agent_selection(self, rtx_config):
        """Test Agenten-Auswahl"""
        ki = RTX2070BundeskanzlerKI(rtx_config)

        # Politik Query
        agent = ki._select_agent("Was sagt der Bundestag?")
        assert str(agent) == "AgentType.POLITIK"

        # Wirtschaft Query
        agent = ki._select_agent("Wie steht die Inflation?")
        assert str(agent) == "AgentType.WIRTSCHAFT"

        # Klima Query
        agent = ki._select_agent("Energiewende Pläne")
        assert str(agent) == "AgentType.KLIMA"

    def test_process_query_basic(self, rtx_config):
        """Test grundlegende Query-Verarbeitung"""
        ki = RTX2070BundeskanzlerKI(rtx_config)

        result = ki.process_query("Testfrage")

        assert "query" in result
        assert "response" in result
        assert "query_complexity" in result
        assert "components_used" in result
        assert result["query"] == "Testfrage"

    def test_system_info(self, rtx_config):
        """Test System-Info Abruf"""
        ki = RTX2070BundeskanzlerKI(rtx_config)

        info = ki.get_system_info()

        assert "config" in info
        assert "gpu_info" in info
        assert "components_status" in info
        assert info["rtx2070_optimized"] == True

    def test_components_status(self, rtx_config):
        """Test Komponenten-Status"""
        ki = RTX2070BundeskanzlerKI(rtx_config)

        status = ki._get_components_status()

        assert "rtx2070_llm" in status
        assert "rtx2070_rag" in status
        assert "multi_agent_system" in status
        assert "fallback_rag" in status

        # Bei Test-Konfiguration sollte LLM deaktiviert sein
        assert status["rtx2070_llm"] == False


class TestRTX2070Integration:
    """Integrationstests für RTX 2070 Komponenten"""

    def test_factory_functions(self):
        """Test Factory-Funktionen"""
        # LLM Manager
        llm_manager = get_rtx2070_llm_manager()
        assert isinstance(llm_manager, RTX2070LLMManager)

        # RAG System
        rag_system = create_rtx2070_rag_system()
        assert isinstance(rag_system, RTX2070OptimizedRAG)

        # KI System
        ki_system = get_rtx2070_bundeskanzler_ki()
        assert isinstance(ki_system, RTX2070BundeskanzlerKI)

    def test_memory_management(self):
        """Test Speicher-Management"""
        # Mehrere Instanzen sollten dieselbe globale Instanz verwenden
        ki1 = get_rtx2070_bundeskanzler_ki()
        ki2 = get_rtx2070_bundeskanzler_ki()

        assert ki1 is ki2  # Selbe Instanz

    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test vollständige Pipeline (nur bei verfügbarer Hardware)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA nicht verfügbar")

        # Vollständige Pipeline testen
        ki = get_rtx2070_bundeskanzler_ki()

        query = "Was ist die Bedeutung der Digitalisierung?"
        result = ki.process_query(query)

        assert result["query"] == query
        assert "response" in result
        assert len(result["response"]) > 0


# Benchmarks für RTX 2070 Performance
class TestRTX2070Performance:
    """Performance-Tests für RTX 2070 Optimierungen"""

    @pytest.fixture
    def performance_config(self):
        """Performance-Test Konfiguration"""
        return RTX2070Config(
            enable_llm_integration=True, enable_rag_optimization=True, dynamic_model_loading=True
        )

    @pytest.mark.benchmark
    def test_rag_query_performance(self, temp_corpus):
        """Benchmark RAG-Abfrage Performance"""
        rag = RTX2070OptimizedRAG(corpus_path=temp_corpus)
        rag.build_index(force_rebuild=True)

        import time

        start_time = time.time()

        for _ in range(10):
            result = rag.rag_query("Energiewende")

        end_time = time.time()
        avg_time = (end_time - start_time) / 10

        # Sollte unter 1 Sekunde pro Abfrage liegen
        assert avg_time < 1.0, f"Durchschnittliche Abfragezeit: {avg_time:.3f}s"

    @pytest.mark.benchmark
    def test_model_loading_performance(self):
        """Benchmark Modell-Ladezeiten"""
        manager = RTX2070LLMManager()

        import time

        start_time = time.time()

        # Schnelles Modell laden (German GPT-2)
        success = manager.load_model("german_gpt2")

        end_time = time.time()
        load_time = end_time - start_time

        # Sollte unter 30 Sekunden laden
        assert load_time < 30.0, f"Ladezeit: {load_time:.1f}s"
        assert success


if __name__ == "__main__":
    # Direkter Test-Launch
    pytest.main([__file__, "-v"])
