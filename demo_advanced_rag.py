#!/usr/bin/env python3
"""
Advanced RAG System 2.0 - Demo und Test Script
Zeigt die neuen Hybrid Search Capabilities
"""

import sys
import os

# Pfad Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.advanced_rag_system import AdvancedRAGSystem
import json
from datetime import datetime


def demo_advanced_rag():
    """Demo der Advanced RAG System 2.0 Features"""
    
    print("🚀 ADVANCED RAG SYSTEM 2.0 DEMO")
    print("=" * 50)
    
    # Initialisiere System
    print("\n📡 Initialisiere Advanced RAG System...")
    try:
        rag = AdvancedRAGSystem(
            use_hybrid_search=True,
            bm25_weight=0.3,
            semantic_weight=0.7
        )
        print("✅ System erfolgreich initialisiert!")
        
        # System Info
        info = rag.get_system_info()
        print(f"\n📊 SYSTEM INFO:")
        print(f"   Model: {info['model']}")
        print(f"   Corpus Size: {info['corpus_size']} Dokumente")
        print(f"   Hybrid Search: {info['hybrid_search']}")
        print(f"   Indexes: Semantic={info['indexes']['semantic']}, BM25={info['indexes']['bm25']}")
        
    except Exception as e:
        print(f"❌ Fehler bei Initialisierung: {e}")
        return
    
    # Test Queries
    test_queries = [
        "Was sind die Klimaziele von Deutschland?",
        "Energiewende und erneuerbare Energien",
        "Kohleausstieg Deutschland",
        "Elektromobilität Förderung",
        "CO2 Emissionen reduzieren"
    ]
    
    print(f"\n🤖 TESTE {len(test_queries)} QUERIES:")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. QUERY: '{query}'")
        
        try:
            start_time = datetime.now()
            results = rag.retrieve_relevant_documents(query, top_k=3)
            response_time = (datetime.now() - start_time).total_seconds()
            
            print(f"   ⏱️ Response Time: {response_time:.3f}s")
            print(f"   📋 Ergebnisse: {len(results)}")
            
            for j, result in enumerate(results, 1):
                print(f"   {j}. Score: {result['score']:.3f} | Type: {result['search_type']}")
                print(f"      Text: {result['text'][:80]}...")
                if 'word_overlap' in result:
                    print(f"      Word Overlap: {result['word_overlap']}")
                    
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
    
    # Benchmark verschiedener Suchmethoden
    print(f"\n🔬 BENCHMARK VERSCHIEDENER SUCHMETHODEN:")
    print("-" * 45)
    
    try:
        benchmark_results = rag.benchmark_search(test_queries[:3], iterations=3)
        
        print("PERFORMANCE VERGLEICH:")
        for method, stats in benchmark_results['summary'].items():
            print(f"  {method.upper()}:")
            print(f"    Avg Time: {stats['avg_time']:.4f}s")
            print(f"    Avg Results: {stats['avg_results']:.1f}")
            
    except Exception as e:
        print(f"❌ Benchmark Fehler: {e}")
    
    # Finale Statistiken
    print(f"\n📊 FINALE SYSTEM STATISTIKEN:")
    print("-" * 30)
    final_stats = rag.stats
    print(f"Queries Processed: {final_stats['queries_processed']}")
    print(f"Hybrid Searches: {final_stats['hybrid_searches']}")
    print(f"Semantic Searches: {final_stats['semantic_searches']}")
    print(f"Avg Response Time: {final_stats['avg_response_time']:.3f}s")


def compare_rag_versions():
    """Vergleicht altes vs. neues RAG System"""
    
    print("\n🔄 RAG SYSTEM VERGLEICH: V1 vs V2")
    print("=" * 40)
    
    test_query = "Was sind die deutschen Klimaziele?"
    
    try:
        # Old RAG System
        print("\n📡 Teste Original RAG System...")
        from core.rag_system import RAGSystem
        
        old_rag = RAGSystem()
        start_time = datetime.now()
        old_results = old_rag.retrieve_relevant_documents(test_query, top_k=3)
        old_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Original RAG: {len(old_results)} Ergebnisse in {old_time:.3f}s")
        
        # New RAG System
        print("\n🚀 Teste Advanced RAG System 2.0...")
        new_rag = AdvancedRAGSystem()
        start_time = datetime.now()
        new_results = new_rag.retrieve_relevant_documents(test_query, top_k=3)
        new_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Advanced RAG: {len(new_results)} Ergebnisse in {new_time:.3f}s")
        
        # Vergleich
        print(f"\n📊 PERFORMANCE VERGLEICH:")
        print(f"   Speed Improvement: {((old_time - new_time) / old_time * 100):+.1f}%")
        print(f"   Results Quality: Enhanced with hybrid search & reranking")
        
        print(f"\n🏆 VERBESSERUNGEN IN V2:")
        print("   ✅ Hybrid Search (BM25 + Semantic)")
        print("   ✅ German Language Model Support")
        print("   ✅ Query Expansion mit Synonymen")
        print("   ✅ Intelligent Reranking")
        print("   ✅ Performance Monitoring")
        print("   ✅ Advanced Configuration")
        
    except Exception as e:
        print(f"❌ Vergleich fehlgeschlagen: {e}")


def interactive_mode():
    """Interaktiver Modus für Testing"""
    
    print("\n💬 INTERAKTIVER MODUS")
    print("=" * 25)
    print("Gib deine Fragen ein (oder 'quit' zum Beenden)")
    
    try:
        rag = AdvancedRAGSystem()
        
        while True:
            query = input("\n🤖 Deine Frage: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
            
            try:
                start_time = datetime.now()
                results = rag.retrieve_relevant_documents(query, top_k=3)
                response_time = (datetime.now() - start_time).total_seconds()
                
                print(f"\n📋 {len(results)} Ergebnisse in {response_time:.3f}s:")
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Score: {result['score']:.3f} | {result['search_type'].upper()}")
                    print(f"   {result['text']}")
                    
            except Exception as e:
                print(f"❌ Fehler: {e}")
                
    except Exception as e:
        print(f"❌ Initialisierung fehlgeschlagen: {e}")
    
    print("\n👋 Tschüss!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_rag_versions()
    else:
        demo_advanced_rag()
        
        # Optional: Vergleich auch ausführen
        if input("\n🔄 RAG Vergleich ausführen? (y/n): ").lower() == 'y':
            compare_rag_versions()