"""
Multi-Agent System Test Suite
============================

Comprehensive unit and integration tests for the Multi-Agent Intelligence System
including individual agent tests, coordinator tests, and system integration tests.

Author: Advanced RAG Team
Date: September 15, 2025
"""

import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import time
from typing import Dict, List, Any

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.multi_agent_system import (
    MultiAgentSystem, CoordinatorAgent, PolitikAgent, WirtschaftAgent, KlimaAgent,
    QueryClassifier, AgentType, QueryComplexity, AgentCapability,
    QueryClassification, AgentResponse, MultiAgentResponse
)

class TestQueryClassifier(unittest.TestCase):
    """Test suite for QueryClassifier"""
    
    def setUp(self):
        self.classifier = QueryClassifier()
    
    def test_politik_classification(self):
        """Test politik domain classification"""
        test_queries = [
            "Wer ist der Bundeskanzler?",
            "Wie funktioniert der Bundestag?",
            "Welche Parteien sind in der Koalition?",
            "Was sagt das Grundgesetz über Wahlen?"
        ]
        
        for query in test_queries:
            classification = self.classifier.classify_query(query)
            self.assertEqual(classification.primary_domain, 'politik',
                           f"Query '{query}' should be classified as politik")
            self.assertIn(AgentType.POLITIK, classification.routing_decision)
    
    def test_wirtschaft_classification(self):
        """Test wirtschaft domain classification"""
        test_queries = [
            "Wie hoch ist die Inflation?",
            "Was macht die EZB mit den Zinsen?",
            "Wie entwickelt sich der Arbeitsmarkt?",
            "Welche Steuern plant die Regierung?"
        ]
        
        for query in test_queries:
            classification = self.classifier.classify_query(query)
            self.assertEqual(classification.primary_domain, 'wirtschaft',
                           f"Query '{query}' should be classified as wirtschaft")
            self.assertIn(AgentType.WIRTSCHAFT, classification.routing_decision)
    
    def test_klima_classification(self):
        """Test klima domain classification"""
        test_queries = [
            "Was sind die CO2-Ziele?",
            "Wie läuft die Energiewende?",
            "Wann steigt Deutschland aus der Kohle aus?",
            "Welche Rolle spielt Wasserstoff?"
        ]
        
        for query in test_queries:
            classification = self.classifier.classify_query(query)
            self.assertEqual(classification.primary_domain, 'klima',
                           f"Query '{query}' should be classified as klima")
            self.assertIn(AgentType.KLIMA, classification.routing_decision)
    
    def test_complex_multi_domain_classification(self):
        """Test complex multi-domain classification"""
        complex_queries = [
            "Wie beeinflusst die Klimapolitik die Wirtschaft?",
            "Welche politischen Maßnahmen gibt es für den Arbeitsmarkt und Klimaschutz?",
            "Wie wirken sich CO2-Steuern auf Unternehmen und Wahlen aus?"
        ]
        
        for query in complex_queries:
            classification = self.classifier.classify_query(query)
            self.assertGreaterEqual(len(classification.secondary_domains), 1,
                                  f"Complex query '{query}' should have secondary domains")
            self.assertEqual(classification.complexity, QueryComplexity.COMPLEX,
                           f"Query '{query}' should be classified as complex")
    
    def test_confidence_scoring(self):
        """Test confidence scoring for classifications"""
        high_confidence_query = "Wer ist der aktuelle Bundeskanzler von Deutschland?"
        low_confidence_query = "Was denkst du über die Situation?"
        
        high_conf_result = self.classifier.classify_query(high_confidence_query)
        low_conf_result = self.classifier.classify_query(low_confidence_query)
        
        self.assertGreater(high_conf_result.confidence, low_conf_result.confidence,
                          "High confidence query should have higher confidence score")

class TestAgentCapabilities(unittest.TestCase):
    """Test suite for individual agent capabilities"""
    
    def setUp(self):
        # Mock RAG system for testing
        self.mock_rag = Mock()
        self.mock_rag.search.return_value = [
            {
                'text': 'Bundeskanzler Olaf Scholz führt die Bundesregierung.',
                'score': 0.95,
                'metadata': {'source': 'politik_test'}
            },
            {
                'text': 'Die SPD ist die stärkste Partei in der Koalition.',
                'score': 0.88,
                'metadata': {'source': 'politik_test2'}
            }
        ]
        
        self.politik_agent = PolitikAgent(self.mock_rag)
        self.wirtschaft_agent = WirtschaftAgent(self.mock_rag)
        self.klima_agent = KlimaAgent(self.mock_rag)
    
    def test_politik_agent_relevance_calculation(self):
        """Test politik agent relevance calculation"""
        high_relevance_query = "Bundestag Gesetz Koalition"
        low_relevance_query = "Inflation Zinsen Wirtschaft"
        
        high_relevance = self.politik_agent.calculate_query_relevance(high_relevance_query)
        low_relevance = self.politik_agent.calculate_query_relevance(low_relevance_query)
        
        self.assertGreater(high_relevance, low_relevance,
                          "Politik agent should rate politik queries higher")
        self.assertGreater(high_relevance, 0.5,
                          "High relevance politik query should score > 0.5")
    
    def test_wirtschaft_agent_relevance_calculation(self):
        """Test wirtschaft agent relevance calculation"""
        high_relevance_query = "Inflation Arbeitsmarkt Wirtschaft"
        low_relevance_query = "Bundestag Gesetz Politik"
        
        high_relevance = self.wirtschaft_agent.calculate_query_relevance(high_relevance_query)
        low_relevance = self.wirtschaft_agent.calculate_query_relevance(low_relevance_query)
        
        self.assertGreater(high_relevance, low_relevance,
                          "Wirtschaft agent should rate wirtschaft queries higher")
    
    def test_klima_agent_relevance_calculation(self):
        """Test klima agent relevance calculation"""
        high_relevance_query = "Klimaschutz CO2 Energiewende"
        low_relevance_query = "Bundestag Inflation Politik"
        
        high_relevance = self.klima_agent.calculate_query_relevance(high_relevance_query)
        low_relevance = self.klima_agent.calculate_query_relevance(low_relevance_query)
        
        self.assertGreater(high_relevance, low_relevance,
                          "Klima agent should rate klima queries higher")
    
    def test_agent_capabilities_structure(self):
        """Test agent capabilities are properly structured"""
        agents = [self.politik_agent, self.wirtschaft_agent, self.klima_agent]
        
        for agent in agents:
            self.assertIsInstance(agent.capability.domain, str)
            self.assertIsInstance(agent.capability.expertise_areas, list)
            self.assertIsInstance(agent.capability.keywords, list)
            self.assertIsInstance(agent.capability.confidence_threshold, float)
            
            # Check that expertise areas and keywords are not empty
            self.assertGreater(len(agent.capability.expertise_areas), 0)
            self.assertGreater(len(agent.capability.keywords), 0)

class TestCoordinatorAgent(unittest.IsolatedAsyncioTestCase):
    """Test suite for CoordinatorAgent"""
    
    async def asyncSetUp(self):
        # Mock RAG system
        self.mock_rag = Mock()
        self.mock_rag.search.return_value = [
            {
                'text': 'Test document content for coordinator testing.',
                'score': 0.90,
                'metadata': {'source': 'test'}
            }
        ]
        
        self.coordinator = CoordinatorAgent(self.mock_rag)
    
    async def test_single_agent_response(self):
        """Test coordinator with single agent response"""
        query = "Wer ist der Bundeskanzler?"
        
        # Mock agent response
        mock_response = AgentResponse(
            agent_type=AgentType.POLITIK,
            content="Olaf Scholz ist der aktuelle Bundeskanzler.",
            confidence=0.95,
            sources=[],
            processing_time=0.1,
            metadata={}
        )
        
        with patch.object(self.coordinator.agents[AgentType.POLITIK], 'process_query', 
                         return_value=mock_response):
            result = await self.coordinator.process_query(query)
            
            self.assertIsInstance(result, MultiAgentResponse)
            self.assertEqual(len(result.contributing_agents), 1)
            self.assertEqual(result.contributing_agents[0], AgentType.POLITIK)
            self.assertGreater(result.confidence, 0.0)
    
    async def test_multi_agent_response_synthesis(self):
        """Test coordinator synthesizing multiple agent responses"""
        query = "Wie beeinflusst Klimapolitik die Wirtschaft?"
        
        # Mock responses from multiple agents
        politik_response = AgentResponse(
            agent_type=AgentType.POLITIK,
            content="Politische Perspektive zum Klimaschutz.",
            confidence=0.85,
            sources=[],
            processing_time=0.1,
            metadata={}
        )
        
        wirtschaft_response = AgentResponse(
            agent_type=AgentType.WIRTSCHAFT,
            content="Wirtschaftliche Auswirkungen des Klimaschutzes.",
            confidence=0.80,
            sources=[],
            processing_time=0.12,
            metadata={}
        )
        
        klima_response = AgentResponse(
            agent_type=AgentType.KLIMA,
            content="Klima-Perspektive zur Wirtschaftspolitik.",
            confidence=0.90,
            sources=[],
            processing_time=0.08,
            metadata={}
        )
        
        with patch.object(self.coordinator.agents[AgentType.POLITIK], 'process_query', 
                         return_value=politik_response), \
             patch.object(self.coordinator.agents[AgentType.WIRTSCHAFT], 'process_query', 
                         return_value=wirtschaft_response), \
             patch.object(self.coordinator.agents[AgentType.KLIMA], 'process_query', 
                         return_value=klima_response):
            
            result = await self.coordinator.process_query(query)
            
            self.assertIsInstance(result, MultiAgentResponse)
            self.assertGreaterEqual(len(result.contributing_agents), 2)
            self.assertIn("Multi-Agent Analyse", result.primary_response)
            
            # Check that response contains content from multiple agents
            self.assertIn("Politik", result.primary_response)
            self.assertIn("Wirtschaft", result.primary_response)
            self.assertIn("Klima", result.primary_response)
    
    async def test_coordinator_stats_update(self):
        """Test coordinator statistics updates"""
        initial_queries = self.coordinator.stats['queries_processed']
        
        query = "Test query for stats"
        
        # Mock simple response
        mock_response = AgentResponse(
            agent_type=AgentType.POLITIK,
            content="Test response",
            confidence=0.8,
            sources=[],
            processing_time=0.1,
            metadata={}
        )
        
        with patch.object(self.coordinator.agents[AgentType.POLITIK], 'process_query', 
                         return_value=mock_response):
            await self.coordinator.process_query(query)
            
            # Check stats were updated
            self.assertEqual(self.coordinator.stats['queries_processed'], initial_queries + 1)
            self.assertEqual(self.coordinator.stats['successful_responses'], initial_queries + 1)
            self.assertGreater(self.coordinator.stats['avg_response_time'], 0.0)

class TestMultiAgentSystem(unittest.IsolatedAsyncioTestCase):
    """Integration tests for complete MultiAgentSystem"""
    
    async def asyncSetUp(self):
        # Mock RAG system initialization
        with patch('core.multi_agent_system.AdvancedRAGSystem') as mock_rag_class:
            mock_rag_instance = Mock()
            mock_rag_instance.search.return_value = [
                {
                    'text': 'Integration test document content.',
                    'score': 0.85,
                    'metadata': {'source': 'integration_test'}
                }
            ]
            mock_rag_class.return_value = mock_rag_instance
            
            self.system = MultiAgentSystem()
    
    async def test_system_initialization(self):
        """Test system initializes correctly"""
        self.assertIsNotNone(self.system.coordinator)
        self.assertIsNotNone(self.system.rag_system)
        
        # Test agent capabilities
        capabilities = self.system.get_agent_capabilities()
        self.assertIn('politik', capabilities)
        self.assertIn('wirtschaft', capabilities)
        self.assertIn('klima', capabilities)
    
    async def test_query_interface(self):
        """Test main query interface"""
        query = "Test query for system interface"
        
        # Mock coordinator response
        mock_response = MultiAgentResponse(
            primary_response="Test system response",
            contributing_agents=[AgentType.POLITIK],
            confidence=0.8,
            sources=[],
            synthesis_metadata={},
            total_processing_time=0.1
        )
        
        with patch.object(self.system.coordinator, 'process_query', 
                         return_value=mock_response):
            result = await self.system.query(query)
            
            self.assertIsInstance(result, MultiAgentResponse)
            self.assertEqual(result.primary_response, "Test system response")
    
    async def test_system_status(self):
        """Test system status reporting"""
        status = self.system.get_status()
        
        self.assertIn('coordinator_stats', status)
        self.assertIn('agent_stats', status)
        self.assertIn('system_info', status)
        
        # Check system info
        system_info = status['system_info']
        self.assertEqual(system_info['total_agents'], 3)
        self.assertIn('classification_domains', system_info)

class TestPerformance(unittest.IsolatedAsyncioTestCase):
    """Performance and stress tests"""
    
    async def asyncSetUp(self):
        with patch('core.multi_agent_system.AdvancedRAGSystem') as mock_rag_class:
            mock_rag_instance = Mock()
            mock_rag_instance.search.return_value = [
                {
                    'text': 'Performance test document.',
                    'score': 0.9,
                    'metadata': {'source': 'perf_test'}
                }
            ]
            mock_rag_class.return_value = mock_rag_instance
            
            self.system = MultiAgentSystem()
    
    async def test_response_time_performance(self):
        """Test response time performance"""
        queries = [
            "Wer ist der Bundeskanzler?",
            "Wie hoch ist die Inflation?", 
            "Was sind die CO2-Ziele?",
            "Wie funktioniert der Bundestag?",
            "Was macht die EZB?"
        ]
        
        # Mock quick responses
        mock_response = MultiAgentResponse(
            primary_response="Quick test response",
            contributing_agents=[AgentType.POLITIK],
            confidence=0.8,
            sources=[],
            synthesis_metadata={},
            total_processing_time=0.05
        )
        
        with patch.object(self.system.coordinator, 'process_query', 
                         return_value=mock_response):
            
            start_time = time.time()
            
            # Process queries sequentially
            for query in queries:
                result = await self.system.query(query)
                self.assertIsInstance(result, MultiAgentResponse)
            
            total_time = time.time() - start_time
            avg_time_per_query = total_time / len(queries)
            
            # Performance assertions
            self.assertLess(avg_time_per_query, 1.0,  # Should be < 1 second per query
                           f"Average response time {avg_time_per_query:.3f}s too slow")
    
    async def test_concurrent_query_handling(self):
        """Test handling concurrent queries"""
        queries = [
            "Politik Test 1",
            "Wirtschaft Test 2", 
            "Klima Test 3"
        ]
        
        mock_response = MultiAgentResponse(
            primary_response="Concurrent test response",
            contributing_agents=[AgentType.POLITIK],
            confidence=0.8,
            sources=[],
            synthesis_metadata={},
            total_processing_time=0.1
        )
        
        with patch.object(self.system.coordinator, 'process_query', 
                         return_value=mock_response):
            
            # Execute queries concurrently
            tasks = [self.system.query(query) for query in queries]
            results = await asyncio.gather(*tasks)
            
            # Verify all queries completed
            self.assertEqual(len(results), len(queries))
            for result in results:
                self.assertIsInstance(result, MultiAgentResponse)

def run_tests():
    """Run all test suites"""
    print("🧪 MULTI-AGENT SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestQueryClassifier,
        TestAgentCapabilities,
        TestCoordinatorAgent,
        TestMultiAgentSystem,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)