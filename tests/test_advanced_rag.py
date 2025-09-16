"""
Test Suite für Advanced RAG System 2.0
Umfassende Tests für die neuen Hybrid Search Features
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Project root setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.advanced_rag_system import AdvancedRAGSystem


class TestAdvancedRAGSystem(unittest.TestCase):
    """Test Suite für Advanced RAG System 2.0"""

    def setUp(self):
        """Setup für jeden Test"""
        # Temporärer Corpus für Tests
        self.temp_dir = tempfile.mkdtemp()
        self.corpus_path = os.path.join(self.temp_dir, "test_corpus.json")

        # Test Corpus erstellen
        test_corpus = [
            {
                "text": "Deutschland hat sich ehrgeizige Klimaziele bis 2030 gesetzt.",
                "source": "Klimaschutzgesetz",
                "category": "Klimapolitik",
            },
            {
                "text": "Die Energiewende ist ein zentraler Baustein der deutschen Klimapolitik.",
                "source": "Energiewende-Programm",
                "category": "Energie",
            },
            {
                "text": "Der Kohleausstieg ist bis 2038 geplant.",
                "source": "Kohleausstiegsgesetz",
                "category": "Energiepolitik",
            },
        ]

        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(test_corpus, f, ensure_ascii=False)

    def tearDown(self):
        """Cleanup nach jedem Test"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_system_initialization(self):
        """Test Advanced RAG System Initialisierung"""
        rag = AdvancedRAGSystem(
            corpus_path=self.corpus_path,
            use_hybrid_search=True,
            bm25_weight=0.3,
            semantic_weight=0.7,
        )

        # Basic properties
        assert rag.corpus_path == self.corpus_path
        assert rag.use_hybrid_search == True
        assert rag.bm25_weight == 0.3
        assert rag.semantic_weight == 0.7
        assert len(rag.corpus_entries) == 3

        # Models loaded
        assert rag.embedding_model is not None
        assert rag.semantic_index is not None
        assert rag.bm25_index is not None

    def test_hybrid_search_functionality(self):
        """Test Hybrid Search (BM25 + Semantic)"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path, use_hybrid_search=True)

        # Test Query
        query = "Klimaziele Deutschland"
        results = rag.retrieve_relevant_documents(query, top_k=3)

        # Assertions
        assert len(results) > 0
        assert len(results) <= 3

        # Check result structure
        for result in results:
            assert "text" in result
            assert "score" in result
            assert "rank" in result
            assert "search_type" in result
            assert result["search_type"] == "hybrid"

    def test_semantic_search_only(self):
        """Test reine semantische Suche"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path, use_hybrid_search=False)

        query = "Klimapolitik"
        results = rag.retrieve_relevant_documents(query, top_k=2)

        assert len(results) > 0
        assert len(results) <= 2

        for result in results:
            assert result["search_type"] == "semantic"

    def test_query_expansion(self):
        """Test Query Expansion mit Synonymen"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path)

        # Test Query Expansion
        original_query = "klima"
        expanded_query = rag._expand_query(original_query)

        # Expanded query sollte mehr Begriffe enthalten
        original_words = set(original_query.split())
        expanded_words = set(expanded_query.split())

        assert len(expanded_words) >= len(original_words)
        assert "klima" in expanded_query

    def test_tokenization(self):
        """Test deutsche Tokenisierung"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path)

        text = "Deutschland hat ehrgeizige Klimaziele und Energiewende."
        tokens = rag._tokenize_text(text)

        # Assertions
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "deutschland" in tokens
        assert "klimaziele" in tokens

        # Stoppwörter sollten entfernt sein
        assert "hat" not in tokens
        assert "und" not in tokens

    def test_bm25_search(self):
        """Test BM25 Keyword Search"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path, use_hybrid_search=True)

        query = "kohleausstieg"
        results = rag._bm25_search(query, top_k=2)

        assert len(results) > 0

        for result in results:
            assert "score" in result
            assert result["score"] > 0
            assert result["search_type"] == "bm25"

    def test_reranking(self):
        """Test Intelligent Reranking"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path)

        # Mock results
        mock_results = [
            {"text": "Deutschland hat Klimaziele gesetzt", "score": 0.5, "rank": 1},
            {"text": "Energiewende ist wichtig", "score": 0.6, "rank": 2},
        ]

        query = "klimaziele deutschland"
        reranked = rag._rerank_results(query, mock_results.copy())

        # Check reranking logic
        assert len(reranked) == 2
        assert all("word_overlap" in result for result in reranked)

        # Erstes Ergebnis sollte höhere Überlappung haben
        assert reranked[0]["word_overlap"] >= reranked[1]["word_overlap"]

    def test_system_info(self):
        """Test System Information"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path)

        info = rag.get_system_info()

        # Required fields
        required_fields = [
            "version",
            "model",
            "corpus_size",
            "hybrid_search",
            "indexes",
            "config",
            "stats",
            "weights",
        ]

        for field in required_fields:
            assert field in info

        assert info["version"] == "2.0"
        assert info["corpus_size"] == 3
        assert isinstance(info["stats"], dict)

    def test_benchmark_functionality(self):
        """Test Benchmark Features"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path)

        test_queries = ["klimaziele", "energiewende"]
        benchmark_results = rag.benchmark_search(test_queries, iterations=2)

        # Check structure
        assert "queries" in benchmark_results
        assert "iterations" in benchmark_results
        assert "detailed" in benchmark_results
        assert "summary" in benchmark_results

        # Check that all methods were tested
        assert "semantic" in benchmark_results["summary"]
        assert "hybrid" in benchmark_results["summary"]

    def test_performance_stats_tracking(self):
        """Test Performance Statistics"""
        rag = AdvancedRAGSystem(corpus_path=self.corpus_path)

        initial_queries = rag.stats["queries_processed"]

        # Perform some queries
        rag.retrieve_relevant_documents("test query 1", top_k=1)
        rag.retrieve_relevant_documents("test query 2", top_k=1)

        # Check stats updated
        assert rag.stats["queries_processed"] == initial_queries + 2
        assert rag.stats["avg_response_time"] > 0

    def test_error_handling(self):
        """Test Error Handling"""
        # Test mit non-existentem Corpus
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_corpus = os.path.join(temp_dir, "nonexistent.json")

            # Should not crash, should create default corpus
            rag = AdvancedRAGSystem(corpus_path=fake_corpus)
            assert len(rag.corpus_entries) > 0  # Default corpus created

    def test_backward_compatibility(self):
        """Test Backward Compatibility"""
        # Test dass RAGSystemV2 Alias funktioniert
        from core.advanced_rag_system import RAGSystemV2

        rag = RAGSystemV2(corpus_path=self.corpus_path)
        assert isinstance(rag, AdvancedRAGSystem)

    def test_german_model_fallback(self):
        """Test German Model Fallback"""
        # Test mit nicht-existentem German Model
        rag = AdvancedRAGSystem(
            corpus_path=self.corpus_path,
            german_model="nonexistent/model",
            fallback_model="paraphrase-multilingual-MiniLM-L12-v2",
        )

        # Should fallback to multilingual model
        assert rag.current_model == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_config_customization(self):
        """Test Configuration Customization"""
        custom_config = {
            "top_k": 10,
            "similarity_threshold": 0.5,
            "enable_query_expansion": False,
            "enable_reranking": False,
        }

        rag = AdvancedRAGSystem(corpus_path=self.corpus_path)

        # Update config
        rag.config.update(custom_config)

        # Test dass Config angewendet wird
        query = "test"
        results = rag.retrieve_relevant_documents(query)

        # Mit disabled reranking sollten keine word_overlap keys da sein
        for result in results:
            assert "word_overlap" not in result


class TestAdvancedRAGIntegration(unittest.TestCase):
    """Integration Tests für Advanced RAG System"""

    def test_full_pipeline_integration(self):
        """Test vollständige Pipeline Integration"""

        # Create real corpus
        temp_dir = tempfile.mkdtemp()
        corpus_path = os.path.join(temp_dir, "integration_corpus.json")

        corpus = [
            {
                "text": "Die Bundesregierung hat ehrgeizige Klimaziele für 2030 beschlossen. CO2-Emissionen sollen um 65% reduziert werden.",
                "source": "Regierungsbeschluss 2021",
                "category": "Klimapolitik",
            },
            {
                "text": "Erneuerbare Energien werden massiv ausgebaut. Windkraft und Solarenergie sind zentrale Säulen der Energiewende.",
                "source": "EEG-Novelle 2023",
                "category": "Energiepolitik",
            },
            {
                "text": "Der Kohleausstieg erfolgt bis spätestens 2038. Betroffene Regionen erhalten Strukturhilfen in Milliardenhöhe.",
                "source": "Kohleausstiegsgesetz",
                "category": "Strukturwandel",
            },
        ]

        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)

        try:
            # Initialize system
            rag = AdvancedRAGSystem(
                corpus_path=corpus_path,
                use_hybrid_search=True,
                bm25_weight=0.4,
                semantic_weight=0.6,
            )

            # Test comprehensive queries
            test_cases = [
                {
                    "query": "Was sind die deutschen Klimaziele bis 2030?",
                    "expected_keywords": ["klimaziele", "2030", "co2", "emissionen"],
                },
                {
                    "query": "Energiewende und erneuerbare Energien",
                    "expected_keywords": ["erneuerbare", "energien", "energiewende"],
                },
                {
                    "query": "Kohleausstieg Deutschland Strukturwandel",
                    "expected_keywords": ["kohleausstieg", "strukturhilfen"],
                },
            ]

            for test_case in test_cases:
                results = rag.retrieve_relevant_documents(test_case["query"], top_k=3)

                # Basic assertions
                assert len(results) > 0, f"No results for query: {test_case['query']}"
                assert len(results) <= 3

                # Check result quality
                best_result = results[0]
                assert best_result["score"] > 0
                assert best_result["search_type"] == "hybrid"

                # Check that relevant keywords appear
                result_text = best_result["text"].lower()
                keyword_found = any(
                    keyword in result_text for keyword in test_case["expected_keywords"]
                )
                assert (
                    keyword_found
                ), f"No relevant keywords found in result for query: {test_case['query']}"

            # Test performance
            info = rag.get_system_info()
            assert info["stats"]["queries_processed"] >= len(test_cases)
            assert info["stats"]["avg_response_time"] > 0

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
