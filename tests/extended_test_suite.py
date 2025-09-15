"""
Comprehensive Test Suite für erhöhte Test Coverage
Erweiterte Tests für alle Core-Module
"""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import pytest
import sys


# Zusätzliche Tests für Security Utils
class TestSecurityUtils:
    """Tests für Security Utilities"""

    def test_validate_user_input_normal(self):
        """Test normale Eingabe-Validation"""
        from utils.security import validate_user_input
        
        normal_input = "Was ist die Klimapolitik der Bundesregierung?"
        result = validate_user_input(normal_input)
        assert result == normal_input
        
    def test_validate_user_input_sql_injection(self):
        """Test SQL Injection Schutz"""
        from utils.security import validate_user_input
        
        malicious_input = "'; DROP TABLE users; --"
        result = validate_user_input(malicious_input)
        assert "DROP TABLE" not in result
        assert ";" not in result
        assert "'" not in result
        
    def test_validate_user_input_xss(self):
        """Test XSS Schutz"""
        from utils.security import validate_user_input
        
        xss_input = '<script>alert("hack")</script>'
        result = validate_user_input(xss_input)
        assert "<script>" not in result
        assert "alert" not in result
        
    def test_validate_user_input_too_long(self):
        """Test zu lange Eingaben"""
        from utils.security import validate_user_input
        
        long_input = "x" * 2000
        with pytest.raises(ValueError, match="Input too long"):
            validate_user_input(long_input, max_length=1000)
            
    def test_validate_user_input_invalid_type(self):
        """Test ungültiger Eingabe-Typ"""
        from utils.security import validate_user_input
        
        with pytest.raises(ValueError, match="Input must be a string"):
            validate_user_input(123)
            
    def test_validate_file_path_normal(self):
        """Test normale Datei-Pfad Validation"""
        from utils.security import validate_file_path
        
        normal_path = "data/corpus.json"
        result = validate_file_path(normal_path)
        assert result == normal_path
        
    def test_validate_file_path_traversal(self):
        """Test Directory Traversal Schutz"""
        from utils.security import validate_file_path
        
        with pytest.raises(ValueError, match="Dangerous path pattern"):
            validate_file_path("../../../etc/passwd")
            
    def test_sanitize_log_message(self):
        """Test Log Message Sanitization"""
        from utils.security import sanitize_log_message
        
        dangerous_log = "User input: \nADMIN\tPASSWORD\r\n"
        result = sanitize_log_message(dangerous_log)
        assert "\n" not in result
        assert "\r" not in result
        assert "\t" not in result


class TestSmartCache:
    """Tests für Smart Cache System"""

    def test_smart_cache_initialization(self):
        """Test Smart Cache Initialisierung"""
        from utils.smart_cache import SmartCache
        
        cache = SmartCache(max_size=100, default_ttl=3600)
        assert cache.max_size == 100
        assert cache.default_ttl == 3600
        assert len(cache.cache) == 0
        
    def test_cache_set_and_get(self):
        """Test Cache Set und Get Operationen"""
        from utils.smart_cache import SmartCache
        
        cache = SmartCache()
        test_data = {"answer": "Klimaziele sind wichtig"}
        
        cache.set("Klimaziele", test_data)
        result = cache.get("Klimaziele")
        
        assert result == test_data
        
    def test_cache_expiration(self):
        """Test Cache Expiration"""
        from utils.smart_cache import SmartCache
        import time
        
        cache = SmartCache(default_ttl=1)  # 1 second TTL
        cache.set("test", {"data": "expires"})
        
        # Sofortiger Zugriff sollte funktionieren
        assert cache.get("test") is not None
        
        # Nach TTL sollte expired sein
        time.sleep(1.1)
        assert cache.get("test") is None
        
    def test_cache_eviction(self):
        """Test Cache Eviction bei Überschreitung der Max-Size"""
        from utils.smart_cache import SmartCache
        
        cache = SmartCache(max_size=2)
        
        cache.set("key1", {"data": "1"})
        cache.set("key2", {"data": "2"})
        cache.set("key3", {"data": "3"})  # Sollte key1 evicten
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        
    def test_cache_stats(self):
        """Test Cache Statistiken"""
        from utils.smart_cache import SmartCache
        
        cache = SmartCache()
        cache.set("test1", {"data": "1"})
        cache.set("test2", {"data": "2"})
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 2
        assert stats['max_size'] > 0
        assert 'memory_usage_mb' in stats
        
    def test_embedding_cache(self):
        """Test Embedding Cache"""
        from utils.smart_cache import EmbeddingCache
        import numpy as np
        
        cache = EmbeddingCache(max_size=10)
        embedding = np.array([0.1, 0.2, 0.3])
        
        # Cache embedding
        cache.cache_embedding("test text", embedding)
        
        # Retrieve embedding
        result = cache.get_embedding("test text")
        np.testing.assert_array_equal(result, embedding)
        
    def test_embedding_cache_eviction(self):
        """Test Embedding Cache Eviction"""
        from utils.smart_cache import EmbeddingCache
        import numpy as np
        
        cache = EmbeddingCache(max_size=2)
        
        # Fill cache
        cache.cache_embedding("text1", np.array([1]))
        cache.cache_embedding("text2", np.array([2]))
        
        # Access text1 to increase its count
        cache.get_embedding("text1")
        cache.get_embedding("text1")
        
        # Add text3 - should evict text2 (less accessed)
        cache.cache_embedding("text3", np.array([3]))
        
        assert cache.get_embedding("text1") is not None
        assert cache.get_embedding("text2") is None  # Evicted
        assert cache.get_embedding("text3") is not None


class TestPerformanceKIExtended:
    """Erweiterte Tests für Performance KI"""

    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG System für Tests"""
        with patch('core.performance_ki.RAGSystem') as mock:
            mock_instance = Mock()
            mock_instance.retrieve_relevant_documents.return_value = [
                {"text": "Deutschland hat ehrgeizige Klimaziele.", "score": 0.9}
            ]
            mock.return_value = mock_instance
            yield mock_instance

    def test_performance_ki_with_security(self, mock_rag_system):
        """Test Performance KI mit Security Integration"""
        from core.performance_ki import PerformanceOptimizedKI
        
        ki = PerformanceOptimizedKI()
        
        # Test mit gefährlicher Eingabe
        malicious_input = '<script>alert("hack")</script>'
        response = ki.antwort(malicious_input)
        
        assert "ungültige Zeichen" in response or "zu lang" in response
        
    def test_performance_ki_caching_hit(self, mock_rag_system):
        """Test Performance KI Cache Hit"""
        from core.performance_ki import PerformanceOptimizedKI
        
        ki = PerformanceOptimizedKI()
        
        # Erste Anfrage - Cache Miss
        frage = "Was sind die Klimaziele?"
        response1 = ki.antwort(frage)
        
        # Zweite Anfrage - Cache Hit
        response2 = ki.antwort(frage)
        
        assert response1 == response2
        assert ki.stats["cache_hits"] >= 1
        
    def test_performance_ki_theme_recognition(self):
        """Test Themen-Erkennung"""
        from core.performance_ki import PerformanceOptimizedKI
        
        ki = PerformanceOptimizedKI()
        
        climate_theme = ki.erkenne_thema("Klimaziele und Energiewende")
        assert climate_theme == "Klimaschutz"
        
        politics_theme = ki.erkenne_thema("Bundestag und Wahlen")
        assert politics_theme == "Politik"
        
        economics_theme = ki.erkenne_thema("Wirtschaftswachstum und Inflation")
        assert economics_theme == "Wirtschaft"
        
    def test_performance_ki_stats_tracking(self, mock_rag_system):
        """Test Performance Statistiken Tracking"""
        from core.performance_ki import PerformanceOptimizedKI
        
        ki = PerformanceOptimizedKI()
        
        initial_queries = ki.stats["total_queries"]
        ki.antwort("Test Frage")
        
        assert ki.stats["total_queries"] == initial_queries + 1
        assert "avg_response_time" in ki.get_performance_stats()
        
    def test_performance_ki_cache_optimization(self):
        """Test Cache Optimierung"""
        from core.performance_ki import PerformanceOptimizedKI
        
        ki = PerformanceOptimizedKI()
        
        # Fülle Cache mit Test-Daten
        for i in range(150):  # Mehr als max_cache_size
            ki.response_cache[f"test_key_{i}"] = {
                "answer": f"Test Antwort {i}",
                "timestamp": 1000000 + i
            }
        
        # Optimiere Cache
        ki.optimize_cache(max_size=100)
        
        assert len(ki.response_cache) <= 100


class TestRAGSystemExtended:
    """Erweiterte Tests für RAG System"""

    @pytest.fixture
    def temp_corpus_file(self):
        """Temporäre Corpus-Datei für Tests"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_corpus = [
                {"text": "Deutschland hat ehrgeizige Klimaziele bis 2030."},
                {"text": "Die Energiewende ist ein wichtiger Baustein."},
                {"text": "Erneuerbare Energien werden massiv ausgebaut."}
            ]
            json.dump(test_corpus, f, ensure_ascii=False)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_rag_system_config_validation(self, temp_corpus_file):
        """Test RAG System Konfiguration"""
        from core.rag_system import RAGSystem
        
        # Test mit custom config
        custom_config = {
            "embedding_model": "test-model",
            "max_corpus_size": 1000,
            "top_k": 3
        }
        
        with patch('sentence_transformers.SentenceTransformer'), \
             patch('faiss.IndexFlatIP'):
            rag = RAGSystem(corpus_path=temp_corpus_file, config=custom_config)
            assert rag.config["top_k"] == 3
            assert rag.config["max_corpus_size"] == 1000

    def test_rag_system_corpus_loading(self, temp_corpus_file):
        """Test Corpus Loading"""
        from core.rag_system import RAGSystem
        
        with patch('sentence_transformers.SentenceTransformer'), \
             patch('faiss.IndexFlatIP'):
            rag = RAGSystem(corpus_path=temp_corpus_file)
            assert len(rag.corpus_entries) == 3
            assert "Klimaziele" in rag.corpus_entries[0]["text"]

    def test_rag_system_empty_corpus(self):
        """Test RAG mit leerem Corpus"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            f.flush()
            
            with patch('sentence_transformers.SentenceTransformer'), \
                 patch('faiss.IndexFlatIP'):
                from core.rag_system import RAGSystem
                rag = RAGSystem(corpus_path=f.name)
                assert len(rag.corpus_entries) == 0
                
        os.unlink(f.name)

    def test_rag_system_invalid_corpus_format(self):
        """Test RAG mit ungültigem Corpus Format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            
            with patch('sentence_transformers.SentenceTransformer'), \
                 patch('faiss.IndexFlatIP'):
                from core.rag_system import RAGSystem
                # Sollte Exception abfangen und graceful degradieren
                rag = RAGSystem(corpus_path=f.name)
                assert hasattr(rag, 'corpus_entries')
                
        os.unlink(f.name)


class TestAdvancedMonitorExtended:
    """Erweiterte Tests für Advanced Monitor"""

    def test_monitor_metric_types(self):
        """Test verschiedene Metrik-Typen"""
        from monitoring.advanced_monitor import AdvancedMonitor
        
        monitor = AdvancedMonitor()
        
        # Test Counter Metrik
        monitor.increment_counter("test_counter")
        monitor.increment_counter("test_counter", value=5)
        
        # Test Gauge Metrik
        monitor.set_gauge("test_gauge", 42.5)
        
        # Test Histogram Metrik
        monitor.record_histogram("test_histogram", 1.23)
        
        metrics = monitor.get_metrics()
        assert "test_counter" in str(metrics)
        assert "test_gauge" in str(metrics)
        assert "test_histogram" in str(metrics)

    def test_monitor_alert_rules(self):
        """Test Alert Rules"""
        from monitoring.advanced_monitor import AdvancedMonitor
        
        monitor = AdvancedMonitor()
        
        # Füge Alert Rule hinzu
        monitor.add_alert_rule(
            name="high_response_time",
            condition="response_time > 5.0",
            severity="warning",
            message="Response time too high"
        )
        
        assert len(monitor.alert_rules) == 1
        assert monitor.alert_rules[0]["name"] == "high_response_time"

    def test_monitor_health_check(self):
        """Test Health Check"""
        from monitoring.advanced_monitor import AdvancedMonitor
        
        monitor = AdvancedMonitor()
        health = monitor.health_check()
        
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy", "degraded"]
        assert "checks" in health
        assert "timestamp" in health

    def test_monitor_export_functionality(self):
        """Test Export Funktionen"""
        from monitoring.advanced_monitor import AdvancedMonitor
        
        monitor = AdvancedMonitor()
        
        # Füge Test-Metriken hinzu
        monitor.increment_counter("test_requests")
        monitor.set_gauge("test_memory", 512)
        
        # Test JSON Export
        json_export = monitor.export_json()
        assert isinstance(json_export, str)
        
        # Test Prometheus Export  
        prometheus_export = monitor.export_prometheus()
        assert isinstance(prometheus_export, str)
        assert "test_requests" in prometheus_export


class TestCodeQualityExtended:
    """Erweiterte Tests für Code Quality"""

    def test_code_quality_manager_initialization(self):
        """Test Code Quality Manager Initialisierung"""
        from utils.code_quality import CodeQualityManager
        
        manager = CodeQualityManager()
        assert hasattr(manager, 'run_black')
        assert hasattr(manager, 'run_isort')
        assert hasattr(manager, 'run_pylint')

    def test_code_quality_black_formatting(self):
        """Test Black Code Formatting"""
        from utils.code_quality import CodeQualityManager
        
        manager = CodeQualityManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test():    return 'test'")  # Schlecht formatiert
            f.flush()
            
            # Formatiere mit Black
            result = manager.run_black(f.name)
            assert result["success"] == True
            
        os.unlink(f.name)

    def test_code_quality_isort_imports(self):
        """Test isort Import Sortierung"""
        from utils.code_quality import CodeQualityManager
        
        manager = CodeQualityManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import os\nimport sys\nimport json")  # Unsortierte Imports
            f.flush()
            
            # Sortiere mit isort
            result = manager.run_isort(f.name)
            assert result["success"] == True
            
        os.unlink(f.name)

    def test_code_quality_pylint_check(self):
        """Test Pylint Code Check"""
        from utils.code_quality import CodeQualityManager
        
        manager = CodeQualityManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def good_function():
    '''A well documented function.'''
    return 'Hello World'
""")
            f.flush()
            
            # Prüfe mit Pylint
            result = manager.run_pylint(f.name)
            assert "score" in result
            
        os.unlink(f.name)


# Integration Tests
class TestSystemIntegrationExtended:
    """Erweiterte System Integration Tests"""

    def test_end_to_end_workflow(self):
        """Test kompletter End-to-End Workflow"""
        from core.performance_ki import PerformanceOptimizedKI
        from utils.security import validate_user_input
        from utils.smart_cache import SmartCache
        
        # 1. Input Validation
        user_input = "Was sind die deutschen Klimaziele?"
        validated_input = validate_user_input(user_input)
        assert validated_input == user_input
        
        # 2. Smart Cache Check
        cache = SmartCache()
        cached_result = cache.get(validated_input)
        assert cached_result is None  # Erstes Mal
        
        # 3. KI Processing würde hier stattfinden
        # (Mocked für Test)
        mock_response = {"answer": "Deutschland hat ehrgeizige Klimaziele bis 2030"}
        
        # 4. Cache Update
        cache.set(validated_input, mock_response)
        
        # 5. Verify Cache Hit
        cached_result = cache.get(validated_input)
        assert cached_result == mock_response

    def test_security_integration(self):
        """Test Security Integration in allen Komponenten"""
        from utils.security import validate_user_input, sanitize_log_message
        
        # Test gefährliche Inputs werden abgefangen
        dangerous_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "../../../etc/passwd"
        ]
        
        for dangerous_input in dangerous_inputs:
            try:
                result = validate_user_input(dangerous_input)
                # Sollte sanitized sein
                assert "<script>" not in result
                assert "DROP TABLE" not in result
                assert "../" not in result
            except ValueError:
                # Oder Exception werfen
                pass

    def test_monitoring_integration(self):
        """Test Monitoring Integration"""
        from monitoring.advanced_monitor import AdvancedMonitor
        
        monitor = AdvancedMonitor()
        
        # Simuliere System-Aktivität
        monitor.increment_counter("requests_total")
        monitor.set_gauge("active_users", 42)
        monitor.record_histogram("response_time", 0.25)
        
        # Prüfe Health Check
        health = monitor.health_check()
        assert health["status"] in ["healthy", "unhealthy", "degraded"]
        
        # Prüfe Metrics Export
        metrics = monitor.get_metrics()
        assert len(str(metrics)) > 0


# Performance Tests
class TestPerformanceBenchmarks:
    """Performance Benchmark Tests"""

    @pytest.mark.performance
    def test_cache_performance(self):
        """Test Cache Performance"""
        from utils.smart_cache import SmartCache
        import time
        
        cache = SmartCache(max_size=1000)
        
        # Measure cache operations
        start_time = time.time()
        
        for i in range(1000):
            cache.set(f"key_{i}", {"data": f"value_{i}"})
            
        for i in range(1000):
            result = cache.get(f"key_{i}")
            assert result is not None
            
        end_time = time.time()
        
        # Cache operations should be fast
        assert end_time - start_time < 1.0  # Under 1 second for 2000 ops

    @pytest.mark.performance  
    def test_security_validation_performance(self):
        """Test Security Validation Performance"""
        from utils.security import validate_user_input
        import time
        
        test_inputs = [
            "Was ist die Klimapolitik?",
            "Wie funktioniert die Energiewende?",
            "Welche Ziele hat Deutschland?",
        ] * 100  # 300 inputs
        
        start_time = time.time()
        
        for test_input in test_inputs:
            result = validate_user_input(test_input)
            assert len(result) > 0
            
        end_time = time.time()
        
        # Validation should be fast
        assert end_time - start_time < 0.5  # Under 0.5 seconds for 300 validations


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core", "--cov=utils", "--cov=monitoring"])