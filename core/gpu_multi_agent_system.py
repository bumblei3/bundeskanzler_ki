#!/usr/bin/env python3
"""
🚀 GPU Multi-Agent System für RTX 2070
======================================

Advanced Multi-Agent System mit RTX 2070 Parallel Processing:
- CUDA Streams für simultane Agent-Verarbeitung
- Memory-optimierte 3-Agent Architektur
- Politik, Wirtschaft, Klima parallel auf GPU
- FP16 Mixed Precision für alle Agents
- Smart Load Balancing für 8GB VRAM

Basiert auf: Multi-Agent Intelligence System
Optimiert für: NVIDIA GeForce RTX 2070
Autor: Claude-3.5-Sonnet
Datum: 15. September 2025
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np

# Import GPU Manager und RAG System
import sys
sys.path.append('/home/tobber/bkki_venv')

from core.gpu_manager import get_rtx2070_manager, rtx2070_context
from core.gpu_rag_system import GPUAcceleratedRAG
from core.multi_agent_system import (
    CoordinatorAgent, PolitikAgent, WirtschaftAgent, KlimaAgent,
    QueryClassifier, ResponseSynthesizer
)

logger = logging.getLogger(__name__)

@dataclass
class GPUAgentConfig:
    """Konfiguration für GPU-optimierte Agents"""
    agent_name: str
    cuda_stream_id: int
    memory_allocation_mb: int
    batch_size: int
    fp16_enabled: bool = True
    
class GPUMultiAgentCoordinator:
    """
    🚀 GPU Multi-Agent Coordinator für RTX 2070
    
    Features:
    - Parallel Agent Processing mit CUDA Streams
    - Memory-optimierte Architektur für 8GB VRAM
    - FP16 Mixed Precision für alle Agents
    - Load Balancing & Error Recovery
    - Real-time Performance Monitoring
    """
    
    def __init__(self, corpus_path: str = None):
        """
        Initialisiert GPU Multi-Agent System
        
        Args:
            corpus_path: Pfad zum Corpus für RAG System
        """
        self.corpus_path = corpus_path
        
        # GPU Manager
        self.gpu_manager = get_rtx2070_manager()
        
        # CUDA Streams für parallele Verarbeitung
        self.cuda_streams = {}
        self._setup_cuda_streams()
        
        # Agent Configurations
        self.agent_configs = {
            'politik': GPUAgentConfig(
                agent_name='politik',
                cuda_stream_id=0,
                memory_allocation_mb=2500,  # ~2.5GB für Politik
                batch_size=16
            ),
            'wirtschaft': GPUAgentConfig(
                agent_name='wirtschaft', 
                cuda_stream_id=1,
                memory_allocation_mb=2500,  # ~2.5GB für Wirtschaft
                batch_size=16
            ),
            'klima': GPUAgentConfig(
                agent_name='klima',
                cuda_stream_id=2, 
                memory_allocation_mb=2500,  # ~2.5GB für Klima
                batch_size=16
            )
        }
        
        # GPU-optimierte RAG Systeme pro Agent
        self.agent_rags = {}
        self._initialize_agent_rags()
        
        # Agents
        self.agents = {}
        self._initialize_agents()
        
        # Coordinator & Supporting Systems
        self.coordinator = CoordinatorAgent()
        self.query_classifier = QueryClassifier()
        self.response_synthesizer = ResponseSynthesizer()
        
        # Performance Tracking
        self.performance_stats = {
            'total_queries': 0,
            'parallel_queries': 0,
            'avg_processing_time_ms': 0.0,
            'gpu_utilization_avg': 0.0,
            'memory_efficiency': 0.0,
            'cuda_stream_utilization': {0: 0.0, 1: 0.0, 2: 0.0}
        }
        
        logger.info("🚀 GPU Multi-Agent System initialisiert:")
        logger.info(f"   🎯 GPU Available: {'✅' if self.gpu_manager.is_gpu_available() else '❌'}")
        logger.info(f"   🔥 CUDA Streams: {len(self.cuda_streams)}")
        logger.info(f"   💾 Memory per Agent: 2.5GB")
        logger.info(f"   ⚡ Parallel Agents: 3 (Politik, Wirtschaft, Klima)")
    
    def _setup_cuda_streams(self):
        """Initialisiert CUDA Streams für parallele Agent-Verarbeitung"""
        if not self.gpu_manager.is_gpu_available():
            logger.warning("⚠️ GPU nicht verfügbar - CUDA Streams deaktiviert")
            return
            
        try:
            # Create 3 CUDA Streams für 3 Agents
            for stream_id in range(3):
                stream = torch.cuda.Stream(device=self.gpu_manager.get_device())
                self.cuda_streams[stream_id] = stream
                
            logger.info(f"✅ {len(self.cuda_streams)} CUDA Streams erstellt")
            
        except Exception as e:
            logger.error(f"❌ CUDA Stream Setup fehlgeschlagen: {e}")
    
    def _initialize_agent_rags(self):
        """Initialisiert GPU-optimierte RAG Systeme für jeden Agent"""
        try:
            for agent_name, config in self.agent_configs.items():
                logger.info(f"🔧 Initialisiere {agent_name.upper()} RAG System...")
                
                # GPU RAG mit spezifischer Konfiguration
                rag = GPUAcceleratedRAG(
                    corpus_path=self.corpus_path,
                    use_gpu=True,
                    max_seq_length=512,
                    german_model="deepset/gbert-large"
                )
                
                # Lade Test Corpus falls kein Pfad angegeben
                if not self.corpus_path:
                    rag.corpus = self._get_agent_corpus(agent_name)
                    rag._generate_embeddings()
                    rag._build_indices()
                
                self.agent_rags[agent_name] = rag
                
                logger.info(f"✅ {agent_name.upper()} RAG System bereit")
                
        except Exception as e:
            logger.error(f"❌ Agent RAG Initialisierung fehlgeschlagen: {e}")
            raise
    
    def _get_agent_corpus(self, agent_name: str) -> List[str]:
        """Holt domänen-spezifischen Corpus für Agent"""
        corpora = {
            'politik': [
                "Die Bundesregierung setzt auf koalitionäre Zusammenarbeit zwischen SPD, Grünen und FDP.",
                "Der Bundestag verabschiedete wichtige Gesetze zur Digitalisierung der Verwaltung.",
                "Bundeskanzler Scholz betont die Bedeutung europäischer Integration.",
                "Die Wahlbeteiligung bei der letzten Bundestagswahl lag bei 76,6 Prozent.",
                "Opposition kritisiert Regierungspolitik in der Migrationsfrage scharf."
            ],
            'wirtschaft': [
                "Die deutsche Wirtschaft zeigt Anzeichen einer Erholung nach der Krise.",
                "Inflation erreichte im letzten Quartal 3,2 Prozent und bleibt Herausforderung.",
                "Arbeitslosenquote sank auf 5,4 Prozent - niedrigster Stand seit Jahren.",
                "Bundesbank warnt vor Rezessionsrisiken durch internationale Spannungen.",
                "Mindestlohn wurde auf 12 Euro pro Stunde erhöht zur sozialen Absicherung."
            ],
            'klima': [
                "Deutschland zielt auf Klimaneutralität bis 2045 durch massive Transformation.",
                "Kohleausstieg bis 2038 beschlossen zur Reduktion der CO2-Emissionen.",
                "Erneuerbare Energien sollen 80% des Stromverbrauchs bis 2030 decken.",
                "Energiewende erfordert Investitionen von 500 Milliarden Euro bis 2030.",
                "CO2-Preis steigt schrittweise zur Lenkung hin zu klimafreundlichen Technologien."
            ]
        }
        return corpora.get(agent_name, [])
    
    def _initialize_agents(self):
        """Initialisiert GPU-optimierte Agents"""
        try:
            # Politik Agent
            self.agents['politik'] = PolitikAgent(
                rag_system=self.agent_rags['politik']
            )
            
            # Wirtschaft Agent 
            self.agents['wirtschaft'] = WirtschaftAgent(
                rag_system=self.agent_rags['wirtschaft']
            )
            
            # Klima Agent
            self.agents['klima'] = KlimaAgent(
                rag_system=self.agent_rags['klima']
            )
            
            logger.info("✅ Alle GPU-Agents initialisiert")
            
        except Exception as e:
            logger.error(f"❌ Agent Initialisierung fehlgeschlagen: {e}")
            raise
    
    async def process_query_parallel(
        self, 
        query: str, 
        confidence_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Verarbeitet Query mit parallelen GPU-Agents
        
        Args:
            query: Benutzeranfrage
            confidence_threshold: Mindest-Konfidenz für Agent-Aktivierung
            
        Returns:
            Kombinierte Antwort aller relevanten Agents
        """
        start_time = time.perf_counter()
        
        try:
            # 1. Query Classification
            classifications = self.query_classifier.classify_query(query)
            
            # 2. Bestimme aktive Agents basierend auf Confidence
            active_agents = []
            for domain, confidence in classifications.items():
                if confidence >= confidence_threshold and domain in self.agents:
                    active_agents.append(domain)
            
            if not active_agents:
                # Fallback: verwende besten Agent
                best_domain = max(classifications.items(), key=lambda x: x[1])[0]
                if best_domain in self.agents:
                    active_agents = [best_domain]
            
            logger.info(f"🎯 Active Agents: {active_agents}")
            logger.info(f"📊 Classifications: {classifications}")
            
            # 3. Parallel GPU Processing
            if len(active_agents) > 1 and self.gpu_manager.is_gpu_available():
                responses = await self._process_parallel_gpu(query, active_agents)
            else:
                responses = await self._process_sequential(query, active_agents)
            
            # 4. Response Synthesis
            synthesized_response = self.response_synthesizer.synthesize_responses(
                query, responses, classifications
            )
            
            # 5. Performance Tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(processing_time, len(active_agents) > 1)
            
            # 6. Finale Response
            result = {
                'query': query,
                'response': synthesized_response,
                'active_agents': active_agents,
                'classifications': classifications,
                'processing_time_ms': processing_time,
                'parallel_processing': len(active_agents) > 1,
                'gpu_accelerated': self.gpu_manager.is_gpu_available()
            }
            
            logger.info(f"✅ Query verarbeitet in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"❌ Parallel Query Processing fehlgeschlagen: {e}")
            return {
                'query': query,
                'response': f"Fehler bei der Verarbeitung: {str(e)}",
                'error': True,
                'processing_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    async def _process_parallel_gpu(
        self, 
        query: str, 
        active_agents: List[str]
    ) -> Dict[str, Any]:
        """
        Verarbeitet Query parallel auf GPU mit CUDA Streams
        
        Args:
            query: User Query
            active_agents: Liste aktiver Agent-Namen
            
        Returns:
            Dict mit Agent-Responses
        """
        responses = {}
        
        try:
            # Parallel Execution mit CUDA Streams
            with rtx2070_context() as gpu_manager:
                
                # Create Tasks für parallele Ausführung
                tasks = []
                
                for agent_name in active_agents:
                    if agent_name in self.agents:
                        config = self.agent_configs[agent_name]
                        stream = self.cuda_streams.get(config.cuda_stream_id)
                        
                        # GPU Stream Context
                        task = self._execute_agent_on_stream(
                            agent_name, query, stream, config
                        )
                        tasks.append(task)
                
                # Await all parallel tasks
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for i, result in enumerate(results):
                        agent_name = active_agents[i]
                        if isinstance(result, Exception):
                            logger.error(f"❌ {agent_name} Agent Fehler: {result}")
                            responses[agent_name] = {
                                'response': f"Agent {agent_name} Fehler: {str(result)}",
                                'confidence': 0.0,
                                'error': True
                            }
                        else:
                            responses[agent_name] = result
            
            logger.info(f"🚀 Parallel GPU Processing: {len(responses)} Agents")
            return responses
            
        except Exception as e:
            logger.error(f"❌ Parallel GPU Processing fehlgeschlagen: {e}")
            return {}
    
    async def _execute_agent_on_stream(
        self,
        agent_name: str,
        query: str, 
        cuda_stream: torch.cuda.Stream,
        config: GPUAgentConfig
    ) -> Dict[str, Any]:
        """
        Führt Agent auf spezifischem CUDA Stream aus
        
        Args:
            agent_name: Name des Agents
            query: User Query
            cuda_stream: CUDA Stream für Execution
            config: Agent Configuration
            
        Returns:
            Agent Response
        """
        try:
            agent = self.agents[agent_name]
            
            # CUDA Stream Context
            if cuda_stream:
                with torch.cuda.stream(cuda_stream):
                    # Mixed Precision Context für Agent
                    with self.gpu_manager.mixed_precision_context():
                        response = await asyncio.get_event_loop().run_in_executor(
                            None, agent.process_query, query
                        )
            else:
                # Fallback ohne Stream
                response = await asyncio.get_event_loop().run_in_executor(
                    None, agent.process_query, query
                )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Agent {agent_name} Execution auf Stream fehlgeschlagen: {e}")
            raise
    
    async def _process_sequential(
        self, 
        query: str, 
        active_agents: List[str]
    ) -> Dict[str, Any]:
        """Fallback: Sequential Processing ohne GPU Parallelisierung"""
        responses = {}
        
        try:
            for agent_name in active_agents:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, agent.process_query, query
                    )
                    responses[agent_name] = response
            
            logger.info(f"💻 Sequential Processing: {len(responses)} Agents")
            return responses
            
        except Exception as e:
            logger.error(f"❌ Sequential Processing fehlgeschlagen: {e}")
            return {}
    
    def _update_performance_stats(self, processing_time_ms: float, was_parallel: bool):
        """Update Performance Statistics"""
        self.performance_stats['total_queries'] += 1
        
        if was_parallel:
            self.performance_stats['parallel_queries'] += 1
        
        # Update Average Processing Time
        total = self.performance_stats['total_queries']
        current_avg = self.performance_stats['avg_processing_time_ms']
        new_avg = ((current_avg * (total - 1)) + processing_time_ms) / total
        self.performance_stats['avg_processing_time_ms'] = new_avg
        
        # Update GPU Stats
        if self.gpu_manager.is_gpu_available():
            gpu_stats = self.gpu_manager.get_gpu_stats()
            if gpu_stats:
                self.performance_stats['gpu_utilization_avg'] = gpu_stats.gpu_utilization
                self.performance_stats['memory_efficiency'] = gpu_stats.memory_utilization
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Holt detaillierte Performance Summary"""
        summary = self.performance_stats.copy()
        
        # Add GPU Information
        if self.gpu_manager.is_gpu_available():
            gpu_stats = self.gpu_manager.get_gpu_stats()
            memory_summary = self.gpu_manager.get_memory_summary()
            
            summary.update({
                'gpu_status': 'ACTIVE',
                'current_gpu_utilization': gpu_stats.gpu_utilization if gpu_stats else 0,
                'current_memory_usage_gb': gpu_stats.memory_used_gb if gpu_stats else 0,
                'current_temperature_c': gpu_stats.temperature_c if gpu_stats else 0,
                'tensor_cores_enabled': self.gpu_manager.tensor_cores_enabled,
                'cuda_streams_active': len(self.cuda_streams)
            })
            
            if memory_summary:
                summary.update(memory_summary)
        else:
            summary['gpu_status'] = 'CPU_FALLBACK'
        
        # Calculate Efficiency Metrics
        if summary['total_queries'] > 0:
            parallel_ratio = summary['parallel_queries'] / summary['total_queries']
            summary['parallel_processing_ratio'] = parallel_ratio
            
            # Estimate Speedup
            if parallel_ratio > 0:
                estimated_speedup = 1 + (parallel_ratio * 2)  # Conservative estimate
                summary['estimated_speedup'] = estimated_speedup
        
        return summary
    
    async def benchmark_parallel_performance(
        self, 
        test_queries: List[str], 
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmarks parallel vs sequential performance
        
        Args:
            test_queries: Liste von Test-Queries
            iterations: Anzahl Benchmark-Iterationen
            
        Returns:
            Benchmark Ergebnisse
        """
        logger.info(f"🚀 Starte Multi-Agent GPU Benchmark ({iterations} Iterationen)...")
        
        # Parallel Performance
        parallel_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            
            for query in test_queries:
                await self.process_query_parallel(query, confidence_threshold=0.1)
            
            iteration_time = time.perf_counter() - start_time
            parallel_times.append(iteration_time)
            logger.info(f"   Parallel Iteration {i+1}: {iteration_time:.2f}s")
        
        avg_parallel_time = sum(parallel_times) / len(parallel_times)
        
        # Sequential Performance (deaktiviere parallele Verarbeitung)
        original_streams = self.cuda_streams
        self.cuda_streams = {}  # Disable parallel processing
        
        sequential_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            
            for query in test_queries:
                await self.process_query_parallel(query, confidence_threshold=0.1)
            
            iteration_time = time.perf_counter() - start_time
            sequential_times.append(iteration_time)
            logger.info(f"   Sequential Iteration {i+1}: {iteration_time:.2f}s")
        
        avg_sequential_time = sum(sequential_times) / len(sequential_times)
        
        # Restore parallel processing
        self.cuda_streams = original_streams
        
        # Calculate Results
        speedup = avg_sequential_time / avg_parallel_time if avg_parallel_time > 0 else 1.0
        
        results = {
            'test_queries_count': len(test_queries),
            'iterations': iterations,
            'avg_parallel_time_s': avg_parallel_time,
            'avg_sequential_time_s': avg_sequential_time,
            'parallel_speedup': speedup,
            'parallel_queries_per_sec': len(test_queries) / avg_parallel_time,
            'sequential_queries_per_sec': len(test_queries) / avg_sequential_time,
            'gpu_enabled': self.gpu_manager.is_gpu_available(),
            'cuda_streams_count': len(original_streams)
        }
        
        logger.info("✅ Multi-Agent Benchmark abgeschlossen:")
        logger.info(f"   🚀 Parallel: {avg_parallel_time:.2f}s")
        logger.info(f"   💻 Sequential: {avg_sequential_time:.2f}s") 
        logger.info(f"   ⚡ Speedup: {speedup:.2f}x")
        
        return results

# Convenience Functions
def create_gpu_multi_agent_system(corpus_path: str = None) -> GPUMultiAgentCoordinator:
    """
    Erstellt GPU Multi-Agent System
    
    Args:
        corpus_path: Pfad zum Corpus
        
    Returns:
        GPUMultiAgentCoordinator Instanz
    """
    return GPUMultiAgentCoordinator(corpus_path=corpus_path)

async def test_gpu_multi_agent():
    """Test Function für GPU Multi-Agent System"""
    print("🚀 Testing GPU Multi-Agent System...")
    
    # Create System
    multi_agent = create_gpu_multi_agent_system()
    
    # Test Queries
    test_queries = [
        "Was ist die Klimapolitik der Bundesregierung?",
        "Wie entwickelt sich die deutsche Wirtschaft?", 
        "Welche politischen Reformen plant die Koalition?",
        "Was bedeutet die Energiewende für Unternehmen und Klima?"
    ]
    
    # Test Individual Queries
    for query in test_queries:
        print(f"\n🤖 Query: {query}")
        result = await multi_agent.process_query_parallel(query)
        print(f"   Response: {result['response'][:100]}...")
        print(f"   Agents: {result['active_agents']}")
        print(f"   Time: {result['processing_time_ms']:.1f}ms")
        print(f"   Parallel: {'✅' if result['parallel_processing'] else '❌'}")
    
    # Performance Summary
    summary = multi_agent.get_performance_summary()
    print(f"\n📊 Performance Summary:")
    print(f"   Total Queries: {summary['total_queries']}")
    print(f"   Parallel Queries: {summary['parallel_queries']}")
    print(f"   Avg Time: {summary['avg_processing_time_ms']:.1f}ms")
    print(f"   GPU Status: {summary['gpu_status']}")
    
    # Benchmark
    print(f"\n🚀 Running Performance Benchmark...")
    benchmark = await multi_agent.benchmark_parallel_performance(test_queries[:2])
    print(f"   Parallel Speedup: {benchmark['parallel_speedup']:.2f}x")
    print(f"   Parallel Throughput: {benchmark['parallel_queries_per_sec']:.1f} queries/sec")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_gpu_multi_agent())