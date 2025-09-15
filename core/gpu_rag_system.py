#!/usr/bin/env python3
"""
🚀 GPU-Accelerated RAG System für RTX 2070
============================================

Advanced RAG mit RTX 2070 GPU-Optimierung:
- CUDA-optimierte sentence-transformers
- GPU-accelerated vector search
- Mixed Precision (FP16) für Tensor Cores
- Smart Memory Management für 8GB VRAM
- Batch Processing Optimization

Kompatibel mit Advanced RAG System 2.0
Autor: Claude-3.5-Sonnet
Datum: 15. September 2025
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict

# Import unserer RTX 2070 GPU Manager
from core.gpu_manager import (
    get_rtx2070_manager, 
    rtx2070_context, 
    move_to_rtx2070,
    get_optimal_batch_size,
    is_rtx2070_available
)

logger = logging.getLogger(__name__)

class GPUAcceleratedRAG:
    """
    🚀 GPU-Accelerated RAG System für RTX 2070
    
    Optimiert für NVIDIA GeForce RTX 2070:
    - 8GB VRAM optimal genutzt
    - Tensor Cores für FP16 Mixed Precision  
    - CUDA-accelerated embeddings
    - Batch processing für hohen Durchsatz
    - Fallback zu CPU bei GPU-Problemen
    """

    def __init__(
        self,
        corpus_path: str = None,
        german_model: str = "deepset/gbert-large",
        fallback_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        use_hybrid_search: bool = True,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        use_gpu: bool = True,
        max_seq_length: int = 512,  # Optimiert für RTX 2070
    ):
        """
        Initialisiert GPU-Accelerated RAG System
        
        Args:
            corpus_path: Pfad zur Corpus-Datei
            german_model: Primäres deutsches GPU-Modell
            fallback_model: CPU Fallback Modell
            use_hybrid_search: Hybrid Search aktivieren
            bm25_weight: BM25 Gewichtung (0.0-1.0)
            semantic_weight: Semantic Search Gewichtung (0.0-1.0)
            use_gpu: RTX 2070 GPU verwenden
            max_seq_length: Maximale Sequenzlänge (512 optimal für RTX 2070)
        """
        self.corpus_path = corpus_path
        self.german_model_name = german_model
        self.fallback_model_name = fallback_model
        self.use_hybrid_search = use_hybrid_search
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.use_gpu = use_gpu and is_rtx2070_available()
        self.max_seq_length = max_seq_length
        
        # RTX 2070 Manager
        self.gpu_manager = get_rtx2070_manager()
        
        # Model Storage
        self.model = None
        self.corpus = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        
        # Performance Tracking
        self.performance_stats = {
            'embeddings_generated': 0,
            'searches_performed': 0,
            'average_search_time_ms': 0.0,
            'gpu_utilization_avg': 0.0,
            'cache_hits': 0
        }
        
        # Embedding Cache für bessere Performance
        self.embedding_cache = {}
        self.cache_max_size = 1000
        
        logger.info(f"🚀 GPU-Accelerated RAG initialisiert:")
        logger.info(f"   🎯 GPU Support: {'✅ RTX 2070' if self.use_gpu else '❌ CPU Only'}")
        logger.info(f"   📏 Max Sequence Length: {self.max_seq_length}")
        logger.info(f"   🔄 Hybrid Search: {'✅' if self.use_hybrid_search else '❌'}")
        
        # Initialize System
        self._initialize_models()
        if corpus_path:
            self._load_corpus()
    
    def _initialize_models(self):
        """Initialisiert German Language Model mit RTX 2070 Optimierung"""
        try:
            logger.info("🚀 Initialisiere GPU-optimiertes German Model...")
            
            if self.use_gpu:
                # RTX 2070 GPU Model Loading
                with rtx2070_context() as gpu_manager:
                    # Load model mit GPU-Optimierung
                    self.model = SentenceTransformer(
                        self.german_model_name,
                        device=gpu_manager.get_device()
                    )
                    
                    # RTX 2070 Optimierung
                    self.model.max_seq_length = self.max_seq_length
                    
                    # Mixed Precision für Tensor Cores
                    if gpu_manager.tensor_cores_enabled:
                        self.model = gpu_manager.optimize_model_for_rtx2070(
                            self.model, 
                            enable_fp16=True
                        )
                        logger.info("✅ Model mit FP16 Mixed Precision optimiert")
                    
                    logger.info(f"✅ German Model auf RTX 2070 geladen: {self.german_model_name}")
                    
            else:
                # CPU Fallback
                logger.warning("⚠️ GPU nicht verfügbar - verwende CPU Model")
                self.model = SentenceTransformer(self.fallback_model_name)
                self.model.max_seq_length = self.max_seq_length
                
        except Exception as e:
            logger.error(f"❌ Model Loading fehlgeschlagen: {e}")
            # Emergency CPU Fallback
            try:
                logger.info("🔄 Versuche CPU Fallback Model...")
                self.model = SentenceTransformer(self.fallback_model_name)
                self.use_gpu = False
                logger.info("✅ CPU Fallback Model geladen")
            except Exception as fallback_error:
                logger.error(f"❌ Auch CPU Fallback fehlgeschlagen: {fallback_error}")
                raise
    
    def _load_corpus(self):
        """Lädt und indexiert Corpus mit GPU-Acceleration"""
        try:
            logger.info(f"📚 Lade Corpus: {self.corpus_path}")
            
            # Load Raw Corpus
            if self.corpus_path.endswith('.json'):
                with open(self.corpus_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, list):
                        self.corpus = raw_data
                    else:
                        self.corpus = [raw_data]
            else:
                with open(self.corpus_path, 'r', encoding='utf-8') as f:
                    self.corpus = [line.strip() for line in f if line.strip()]
            
            logger.info(f"📊 Corpus geladen: {len(self.corpus)} Dokumente")
            
            # Generate Embeddings mit GPU-Acceleration
            self._generate_embeddings()
            
            # Build Search Indices
            self._build_indices()
            
            logger.info("✅ Corpus erfolgreich indexiert")
            
        except Exception as e:
            logger.error(f"❌ Corpus Loading fehlgeschlagen: {e}")
            raise
    
    def _generate_embeddings(self):
        """Generiert Embeddings mit RTX 2070 GPU-Acceleration"""
        logger.info("🚀 Generiere GPU-accelerated Embeddings...")
        
        start_time = time.time()
        
        try:
            if self.use_gpu:
                # RTX 2070 Batch Processing
                batch_size = get_optimal_batch_size('embedding_generation')
                
                with rtx2070_context() as gpu_manager:
                    logger.info(f"⚡ GPU Batch Size: {batch_size}")
                    
                    # Generate embeddings in batches
                    all_embeddings = []
                    
                    for i in range(0, len(self.corpus), batch_size):
                        batch = self.corpus[i:i + batch_size]
                        
                        # Mixed Precision Context für Tensor Cores
                        with gpu_manager.mixed_precision_context():
                            batch_embeddings = self.model.encode(
                                batch,
                                batch_size=batch_size,
                                show_progress_bar=True,
                                convert_to_numpy=True,  # Force NumPy conversion
                                normalize_embeddings=True
                            )
                        
                        # Convert zu NumPy
                        if isinstance(batch_embeddings, torch.Tensor):
                            batch_embeddings = batch_embeddings.cpu().numpy()
                        elif hasattr(batch_embeddings, 'cpu'):
                            # Handle andere GPU Tensor Typen
                            batch_embeddings = batch_embeddings.cpu().numpy()
                        
                        all_embeddings.append(batch_embeddings)
                        
                        # Memory Management
                        if i % (batch_size * 4) == 0:
                            gpu_manager.clear_memory()
                            progress = (i + batch_size) / len(self.corpus) * 100
                            logger.info(f"   Progress: {progress:.1f}%")
                    
                    # Combine alle Embeddings
                    self.embeddings = np.vstack(all_embeddings)
                    
            else:
                # CPU Fallback
                logger.info("💻 CPU Embedding Generation...")
                self.embeddings = self.model.encode(
                    self.corpus,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )
            
            generation_time = time.time() - start_time
            
            # Update Performance Stats
            self.performance_stats['embeddings_generated'] = len(self.corpus)
            avg_time_per_doc = (generation_time / len(self.corpus)) * 1000
            
            logger.info(f"✅ Embeddings generiert:")
            logger.info(f"   📊 Dokumente: {len(self.corpus)}")
            logger.info(f"   ⏱️ Gesamt: {generation_time:.2f}s")
            logger.info(f"   ⚡ Pro Dokument: {avg_time_per_doc:.2f}ms")
            logger.info(f"   📐 Shape: {self.embeddings.shape}")
            
            # Save embeddings
            self._save_embeddings()
            
        except Exception as e:
            logger.error(f"❌ Embedding Generation fehlgeschlagen: {e}")
            raise
    
    def _build_indices(self):
        """Baut Search-Indices für Hybrid Search"""
        try:
            logger.info("🔧 Baue Search-Indices...")
            
            # 1. FAISS Index für Semantic Search
            dimension = self.embeddings.shape[1]
            
            if self.use_gpu and is_rtx2070_available():
                # GPU FAISS Index (falls verfügbar)
                try:
                    # Versuche GPU FAISS 
                    import faiss
                    
                    # Standard FAISS CPU Index (FAISS-GPU nicht über pip verfügbar)
                    self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product
                    self.faiss_index.add(self.embeddings.astype('float32'))
                    
                    logger.info(f"✅ FAISS Index erstellt: {dimension}D, {len(self.corpus)} docs")
                    
                except Exception as gpu_faiss_error:
                    logger.warning(f"⚠️ GPU FAISS nicht verfügbar: {gpu_faiss_error}")
                    # Fallback zu CPU FAISS
                    self.faiss_index = faiss.IndexFlatIP(dimension)
                    self.faiss_index.add(self.embeddings.astype('float32'))
            else:
                # CPU FAISS Index
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(self.embeddings.astype('float32'))
                logger.info(f"✅ CPU FAISS Index erstellt: {dimension}D")
            
            # 2. BM25 Index für Keyword Search
            if self.use_hybrid_search:
                # Preprocess für BM25
                preprocessed_corpus = [self._preprocess_text(doc) for doc in self.corpus]
                tokenized_corpus = [doc.split() for doc in preprocessed_corpus]
                
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info("✅ BM25 Index erstellt")
            
            logger.info("✅ Alle Indices erfolgreich erstellt")
            
        except Exception as e:
            logger.error(f"❌ Index Building fehlgeschlagen: {e}")
            raise
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        top_k: int = 10,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevante Dokumente mit GPU-Acceleration
        
        Args:
            query: Suchanfrage
            top_k: Anzahl zurückzugebender Dokumente
            use_cache: Embedding Cache verwenden
            
        Returns:
            Liste relevanter Dokumente mit Scores
        """
        start_time = time.time()
        
        try:
            # Cache Check
            cache_key = f"{query}_{top_k}"
            if use_cache and cache_key in self.embedding_cache:
                self.performance_stats['cache_hits'] += 1
                logger.debug(f"✅ Cache Hit für Query: {query[:50]}...")
                return self.embedding_cache[cache_key]
            
            if self.use_hybrid_search and self.bm25:
                # Hybrid Search: BM25 + Semantic
                results = self._hybrid_search(query, top_k)
            else:
                # Nur Semantic Search
                results = self._semantic_search(query, top_k)
            
            # Cache Management
            if use_cache:
                self._update_cache(cache_key, results)
            
            # Performance Tracking
            search_time = (time.time() - start_time) * 1000
            self.performance_stats['searches_performed'] += 1
            
            # Update average
            current_avg = self.performance_stats['average_search_time_ms']
            searches = self.performance_stats['searches_performed']
            new_avg = ((current_avg * (searches - 1)) + search_time) / searches
            self.performance_stats['average_search_time_ms'] = new_avg
            
            logger.debug(f"🔍 Search completed in {search_time:.2f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Document Retrieval fehlgeschlagen: {e}")
            return []
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """GPU-accelerated Semantic Search"""
        try:
            # Generate Query Embedding mit GPU
            if self.use_gpu:
                with rtx2070_context() as gpu_manager:
                    with gpu_manager.mixed_precision_context():
                        query_embedding = self.model.encode(
                            [query], 
                            normalize_embeddings=True,
                            convert_to_numpy=True
                        )
            else:
                query_embedding = self.model.encode(
                    [query], 
                    normalize_embeddings=True
                )
            
            # FAISS Search
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                top_k
            )
            
            # Format Results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.corpus):
                    results.append({
                        'text': self.corpus[idx],
                        'score': float(score),
                        'rank': i + 1,
                        'method': 'semantic'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic Search fehlgeschlagen: {e}")
            return []
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """GPU-accelerated Hybrid Search (BM25 + Semantic)"""
        try:
            # 1. Semantic Search
            semantic_results = self._semantic_search(query, top_k * 2)
            
            # 2. BM25 Search
            preprocessed_query = self._preprocess_text(query)
            tokenized_query = preprocessed_query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Top BM25 Results
            bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
            bm25_results = []
            
            for i, idx in enumerate(bm25_indices):
                if idx < len(self.corpus):
                    bm25_results.append({
                        'text': self.corpus[idx],
                        'score': float(bm25_scores[idx]),
                        'rank': i + 1,
                        'method': 'bm25'
                    })
            
            # 3. Reciprocal Rank Fusion (RRF)
            combined_results = self._reciprocal_rank_fusion(
                semantic_results, 
                bm25_results, 
                top_k
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid Search fehlgeschlagen: {e}")
            return self._semantic_search(query, top_k)
    
    def _reciprocal_rank_fusion(
        self, 
        semantic_results: List[Dict], 
        bm25_results: List[Dict], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion für Hybrid Search"""
        rrf_scores = defaultdict(float)
        doc_info = {}
        k = 60  # RRF parameter
        
        # Semantic Scores
        for rank, result in enumerate(semantic_results, 1):
            doc_text = result['text']
            rrf_scores[doc_text] += self.semantic_weight / (k + rank)
            doc_info[doc_text] = result
        
        # BM25 Scores
        for rank, result in enumerate(bm25_results, 1):
            doc_text = result['text']
            rrf_scores[doc_text] += self.bm25_weight / (k + rank)
            if doc_text not in doc_info:
                doc_info[doc_text] = result
        
        # Sort by RRF Score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format Final Results
        final_results = []
        for rank, (doc_text, rrf_score) in enumerate(sorted_docs[:top_k], 1):
            result = doc_info[doc_text].copy()
            result.update({
                'score': rrf_score,
                'rank': rank,
                'method': 'hybrid_rrf'
            })
            final_results.append(result)
        
        return final_results
    
    def _preprocess_text(self, text: str) -> str:
        """Text Preprocessing für Deutsche Texte"""
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep German umlauts
        text = re.sub(r'[^\w\säöüß]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _update_cache(self, cache_key: str, results: List[Dict]):
        """Update Embedding Cache mit LRU-ähnlicher Logik"""
        if len(self.embedding_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[cache_key] = results
    
    def _save_embeddings(self):
        """Speichert Embeddings für spätere Verwendung"""
        try:
            embeddings_path = 'gpu_rag_embeddings.pkl'
            with open(embeddings_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'corpus': self.corpus,
                    'model_name': self.german_model_name,
                    'max_seq_length': self.max_seq_length,
                    'created_at': datetime.now().isoformat()
                }, f)
            logger.info(f"💾 Embeddings gespeichert: {embeddings_path}")
        except Exception as e:
            logger.warning(f"⚠️ Embeddings speichern fehlgeschlagen: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Holt Performance-Statistiken"""
        stats = self.performance_stats.copy()
        
        if self.use_gpu:
            gpu_stats = self.gpu_manager.get_gpu_stats()
            if gpu_stats:
                stats.update({
                    'gpu_utilization_current': gpu_stats.gpu_utilization,
                    'gpu_memory_used_gb': gpu_stats.memory_used_gb,
                    'gpu_memory_utilization': gpu_stats.memory_utilization,
                    'gpu_temperature_c': gpu_stats.temperature_c,
                    'tensor_core_enabled': self.gpu_manager.tensor_cores_enabled
                })
        
        stats['model_name'] = self.german_model_name
        stats['use_gpu'] = self.use_gpu
        stats['corpus_size'] = len(self.corpus)
        stats['cache_size'] = len(self.embedding_cache)
        
        return stats
    
    def benchmark_performance(self, test_queries: List[str], iterations: int = 5) -> Dict[str, Any]:
        """
        Performance Benchmark für RTX 2070 vs CPU
        
        Args:
            test_queries: Liste von Test-Queries
            iterations: Anzahl Benchmark-Iterationen
            
        Returns:
            Benchmark-Ergebnisse
        """
        logger.info(f"🚀 Starte RAG Performance Benchmark ({iterations} Iterationen)...")
        
        results = {
            'gpu_enabled': self.use_gpu,
            'query_times_ms': [],
            'embedding_times_ms': [],
            'search_times_ms': [],
            'total_queries': len(test_queries) * iterations
        }
        
        try:
            for iteration in range(iterations):
                logger.info(f"   Iteration {iteration + 1}/{iterations}")
                
                for query in test_queries:
                    # Measure Query Processing
                    start_time = time.perf_counter()
                    
                    documents = self.retrieve_relevant_documents(
                        query, 
                        top_k=10,
                        use_cache=False  # Disable cache für accurate timing
                    )
                    
                    query_time = (time.perf_counter() - start_time) * 1000
                    results['query_times_ms'].append(query_time)
            
            # Calculate Statistics
            query_times = results['query_times_ms']
            avg_time = sum(query_times) / len(query_times)
            min_time = min(query_times)
            max_time = max(query_times)
            
            benchmark_summary = {
                'avg_query_time_ms': avg_time,
                'min_query_time_ms': min_time,
                'max_query_time_ms': max_time,
                'queries_per_second': 1000 / avg_time,
                'gpu_enabled': self.use_gpu,
                'model_name': self.german_model_name,
                'corpus_size': len(self.corpus)
            }
            
            # GPU Stats hinzufügen
            if self.use_gpu:
                gpu_stats = self.gpu_manager.get_gpu_stats()
                if gpu_stats:
                    benchmark_summary.update({
                        'gpu_utilization': gpu_stats.gpu_utilization,
                        'gpu_memory_usage_gb': gpu_stats.memory_used_gb,
                        'gpu_temperature': gpu_stats.temperature_c
                    })
            
            logger.info("✅ RAG Benchmark abgeschlossen:")
            logger.info(f"   ⚡ Avg Query Time: {avg_time:.1f}ms")
            logger.info(f"   🚀 Queries/Second: {1000/avg_time:.1f}")
            logger.info(f"   📊 Total Queries: {len(query_times)}")
            
            return benchmark_summary
            
        except Exception as e:
            logger.error(f"❌ Benchmark fehlgeschlagen: {e}")
            return {'error': str(e)}

# Convenience Functions
def create_gpu_rag(
    corpus_path: str = None,
    use_gpu: bool = True,
    **kwargs
) -> GPUAcceleratedRAG:
    """
    Erstellt GPU-Accelerated RAG System
    
    Args:
        corpus_path: Pfad zum Corpus
        use_gpu: RTX 2070 GPU verwenden
        **kwargs: Weitere Parameter
        
    Returns:
        GPUAcceleratedRAG Instanz
    """
    return GPUAcceleratedRAG(
        corpus_path=corpus_path,
        use_gpu=use_gpu,
        **kwargs
    )

if __name__ == "__main__":
    # Test GPU-Accelerated RAG
    print("🚀 Testing GPU-Accelerated RAG System...")
    
    # Create test corpus
    test_corpus = [
        "Die Klimapolitik der Bundesregierung zielt auf Klimaneutralität bis 2045 ab.",
        "Deutschland plant den Kohleausstieg bis 2038 zur Reduktion der CO2-Emissionen.",
        "Erneuerbare Energien sollen bis 2030 80% des Stromverbrauchs decken.",
        "Die Energiewende erfordert massive Investitionen in Wind- und Solarenergie.",
        "Das Klimaschutzgesetz definiert verbindliche Sektorziele für CO2-Reduktion."
    ]
    
    # Test Embeddings direkt
    rag = GPUAcceleratedRAG()
    rag.corpus = test_corpus
    rag._generate_embeddings()
    rag._build_indices()
    
    # Test Query
    test_query = "Was sind die Klimaziele Deutschlands?"
    results = rag.retrieve_relevant_documents(test_query, top_k=3)
    
    print(f"\nQuery: {test_query}")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
    
    # Performance Stats
    stats = rag.get_performance_stats()
    print(f"\nPerformance: GPU={stats['use_gpu']}, Avg Time={stats['average_search_time_ms']:.1f}ms")