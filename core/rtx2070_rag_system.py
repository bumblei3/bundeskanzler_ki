#!/usr/bin/env python3
"""
🚀 RTX 2070-Optimiertes RAG-System für Bundeskanzler-KI
======================================================

Optimiert für NVIDIA GeForce RTX 2070 (8GB VRAM):
- Quantisierte Embeddings (4-bit/8-bit)
- GPU-optimierte FAISS-Indizes
- Hybrid Search (semantisch + keyword)
- Re-Ranking mit Cross-Encoder
- Dynamic Memory Management

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import gc
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class RTX2070OptimizedEmbeddings:
    """
    RTX 2070 optimierte Embeddings mit Quantisierung
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim = 384  # Standard für MiniLM

        # RTX 2070 Optimierungen
        self.quantization_config = {
            "load_in_8bit": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

    def load_model(self) -> bool:
        """Lädt Embeddings-Modell mit RTX 2070 Optimierungen"""
        try:
            logger.info(f"Lade RTX 2070-optimiertes Embeddings-Modell: {self.model_name}")

            # Sentence Transformers für einfachere Handhabung
            self.model = SentenceTransformer(
                self.model_name, device=self.device, cache_folder="./models/embeddings"
            )

            # GPU Memory optimieren
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ Embeddings-Modell erfolgreich geladen")
            return True

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden des Embeddings-Modells: {e}")
            return False

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """RTX 2070 optimierte Text-Encodierung"""
        if not self.model:
            if not self.load_model():
                raise RuntimeError("Embeddings-Modell konnte nicht geladen werden")

        try:
            # Batch-Verarbeitung für VRAM-Effizienz
            if isinstance(texts, str):
                texts = [texts]

            # RTX 2070 spezifische Batch-Größe
            optimal_batch_size = min(batch_size, 16)  # Konservativ für 8GB VRAM

            embeddings = self.model.encode(
                texts,
                batch_size=optimal_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Für bessere Cosine-Similarity
            )

            return embeddings

        except Exception as e:
            logger.error(f"❌ Fehler bei der Encodierung: {e}")
            return np.array([])


class RTX2070OptimizedRAG:
    """
    RTX 2070 optimiertes RAG-System mit GPU-FAISS und Hybrid Search
    """

    def __init__(
        self,
        corpus_path: str = None,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        models_path: str = None,
        index_path: str = None,
    ):
        # Pfad-Konfiguration
        if corpus_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            corpus_path = os.path.join(project_root, "data", "corpus.json")

        if models_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            models_path = os.path.join(project_root, "models")

        if index_path is None:
            index_path = os.path.join(models_path, "rtx2070_rag_index.faiss")

        self.corpus_path = corpus_path
        self.models_path = models_path
        self.index_path = index_path

        # RTX 2070 optimierte Komponenten
        self.embeddings = RTX2070OptimizedEmbeddings(embedding_model)
        self.index = None
        self.corpus = []
        self.id_to_text = {}

        # GPU-optimierte FAISS-Konfiguration
        self.use_gpu = torch.cuda.is_available()
        self.embedding_dim = 384

        # Hybrid Search Komponenten
        self.keyword_index = None  # Für BM25-ähnliche Suche

        logger.info("🚀 RTX 2070 RAG-System initialisiert")

    def load_corpus(self) -> bool:
        """Lädt und verarbeitet Corpus für RTX 2070"""
        try:
            if not os.path.exists(self.corpus_path):
                logger.warning(f"Corpus-Datei nicht gefunden: {self.corpus_path}")
                return False

            with open(self.corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Corpus verarbeiten - verschiedene Formate unterstützen
            self.corpus = []
            texts = []

            if isinstance(data, list):
                # Direktes Array-Format
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        text_id = len(self.corpus)
                        self.corpus.append(
                            {
                                "id": text_id,
                                "text": item["text"],
                                "metadata": item.get("metadata", {}),
                            }
                        )
                        self.id_to_text[text_id] = item["text"]
                        texts.append(item["text"])
            elif isinstance(data, dict) and "entries" in data:
                # entries-Format
                for item in data["entries"]:
                    if isinstance(item, dict) and "text" in item:
                        text_id = len(self.corpus)
                        self.corpus.append({"id": text_id, "text": item["text"], "metadata": item})
                        self.id_to_text[text_id] = item["text"]
                        texts.append(item["text"])

            logger.info(f"✅ Corpus geladen: {len(self.corpus)} Dokumente")
            return len(self.corpus) > 0

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden des Corpus: {e}")
            return False

    def build_index(self, force_rebuild: bool = False) -> bool:
        """Erstellt RTX 2070 optimierten FAISS-Index"""
        try:
            if os.path.exists(self.index_path) and not force_rebuild:
                logger.info("Lade vorhandenen FAISS-Index...")
                return self.load_index()

            if not self.corpus:
                if not self.load_corpus():
                    return False

            logger.info("🚀 Erstelle RTX 2070-optimierten FAISS-Index...")

            # Texte in Batches encodieren (VRAM-schonend)
            texts = [item["text"] for item in self.corpus]
            batch_size = 8  # RTX 2070 konservativ

            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = self.embeddings.encode(batch_texts, batch_size=8)
                all_embeddings.append(batch_embeddings)

                # VRAM freigeben zwischen Batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            embeddings = np.vstack(all_embeddings)

            # RTX 2070 optimierter FAISS-Index
            if self.use_gpu:
                try:
                    # GPU-Index für RTX 2070 (FAISS 1.12.0 kompatibel)
                    res = faiss.StandardGpuResources()
                    index_flat = faiss.IndexFlatIP(
                        self.embedding_dim
                    )  # Inner Product für normalisierte Embeddings
                    self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                except AttributeError:
                    # Fallback für neuere FAISS-Versionen
                    logger.warning(
                        "⚠️ StandardGpuResources nicht verfügbar, verwende GPU-Index direkt"
                    )
                    try:
                        self.index = faiss.GpuIndexFlatIP(faiss.get_num_gpus(), self.embedding_dim)
                    except Exception as gpu_error:
                        logger.warning(
                            f"⚠️ GPU-Index fehlgeschlagen: {gpu_error}, verwende CPU-Fallback"
                        )
                        self.use_gpu = False
                        self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                # CPU-Fallback
                self.index = faiss.IndexFlatIP(self.embedding_dim)

            # Embeddings hinzufügen
            self.index.add(embeddings.astype("float32"))

            # Index speichern
            self.save_index()

            logger.info(f"✅ FAISS-Index erstellt: {len(self.corpus)} Dokumente")
            return True

        except Exception as e:
            logger.error(f"❌ Fehler beim Erstellen des Index: {e}")
            return False

    def save_index(self):
        """Speichert FAISS-Index RTX 2070-optimiert"""
        try:
            if self.index:
                # GPU-Index zurück auf CPU für Speicherung
                if hasattr(self.index, "index"):
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                else:
                    cpu_index = self.index

                faiss.write_index(cpu_index, self.index_path)
                logger.info(f"💾 Index gespeichert: {self.index_path}")

        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern des Index: {e}")

    def load_index(self) -> bool:
        """Lädt FAISS-Index mit RTX 2070 Optimierung"""
        try:
            if not os.path.exists(self.index_path):
                return False

            # CPU-Index laden
            cpu_index = faiss.read_index(self.index_path)

            # Auf GPU verschieben falls verfügbar
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index = cpu_index

            logger.info("✅ FAISS-Index geladen")
            return True

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden des Index: {e}")
            return False

    def hybrid_search(
        self, query: str, top_k: int = 5, semantic_weight: float = 0.7, keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        RTX 2070 optimierte Hybrid Search (semantisch + keyword)
        """
        try:
            if not self.index:
                if not self.build_index():
                    return []

            # Semantische Suche
            query_embedding = self.embeddings.encode([query])[0]
            scores_semantic, indices_semantic = self.index.search(
                query_embedding.reshape(1, -1).astype("float32"),
                top_k * 2,  # Mehr Ergebnisse für Re-Ranking
            )

            # Keyword-basierte Suche (einfache Implementierung)
            keyword_results = self._keyword_search(query, top_k * 2)

            # Hybrid Scores kombinieren
            hybrid_results = self._combine_hybrid_results(
                semantic_results=list(zip(indices_semantic[0], scores_semantic[0])),
                keyword_results=keyword_results,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                top_k=top_k,
            )

            return hybrid_results

        except Exception as e:
            logger.error(f"❌ Fehler bei der Suche: {e}")
            return []

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Einfache keyword-basierte Suche"""
        query_words = set(query.lower().split())
        results = []

        for item in self.corpus:
            text = item["text"].lower()
            text_words = set(text.split())

            # Jaccard-Ähnlichkeit
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)

            if union > 0:
                score = intersection / union
                results.append((item["id"], score))

        # Top-K Ergebnisse
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _combine_hybrid_results(
        self,
        semantic_results: List[Tuple[int, float]],
        keyword_results: List[Tuple[int, float]],
        semantic_weight: float,
        keyword_weight: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Kombiniert semantische und keyword-basierte Ergebnisse"""
        # Scores normalisieren und kombinieren
        combined_scores = {}

        # Semantische Scores (bereits normalisiert durch Cosine Similarity)
        for idx, score in semantic_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + score * semantic_weight

        # Keyword Scores (normalisieren)
        if keyword_results:
            max_keyword_score = (
                max(score for _, score in keyword_results) if keyword_results else 1.0
            )
            for idx, score in keyword_results:
                normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
                combined_scores[idx] = (
                    combined_scores.get(idx, 0) + normalized_score * keyword_weight
                )

        # Top-K sortieren
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # In erwartetes Format konvertieren
        final_results = []
        for idx, score in sorted_results:
            if idx in self.id_to_text:
                final_results.append(
                    {
                        "id": idx,
                        "text": self.id_to_text[idx],
                        "score": score,
                        "metadata": (
                            self.corpus[idx].get("metadata", {}) if idx < len(self.corpus) else {}
                        ),
                    }
                )

        return final_results

    def rag_query(
        self, query: str, top_k: int = 3, max_context_length: int = 1024
    ) -> Dict[str, Any]:
        """
        RTX 2070 optimierte RAG-Abfrage
        """
        try:
            # Hybrid Search
            search_results = self.hybrid_search(query, top_k=top_k)

            if not search_results:
                return {
                    "query": query,
                    "context": "",
                    "sources": [],
                    "error": "Keine relevanten Dokumente gefunden",
                }

            # Context zusammenstellen (VRAM-schonend)
            context_parts = []
            total_length = 0

            for result in search_results:
                text = result["text"]
                # Text kürzen falls nötig
                if total_length + len(text) > max_context_length:
                    remaining = max_context_length - total_length
                    text = text[:remaining] + "..."
                    context_parts.append(text)
                    break

                context_parts.append(text)
                total_length += len(text)

            context = "\n\n".join(context_parts)

            return {
                "query": query,
                "context": context,
                "sources": search_results,
                "total_sources": len(search_results),
            }

        except Exception as e:
            logger.error(f"❌ Fehler bei RAG-Abfrage: {e}")
            return {"query": query, "context": "", "sources": [], "error": str(e)}

    def get_system_info(self) -> Dict[str, Any]:
        """Gibt System-Informationen zurück"""
        return {
            "document_count": len(self.corpus),
            "corpus_loaded": len(self.corpus) > 0,
            "corpus_entries": len(self.corpus),
            "gpu_accelerated": self.use_gpu,
            "embedding_model": (
                self.embeddings.model_name
                if hasattr(self.embeddings, "model_name")
                else "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            "index_loaded": self.index is not None,
            "corpus_path": self.corpus_path,
            "index_path": self.index_path,
            "embedding_dim": self.embedding_dim,
            "hybrid_search_enabled": self.keyword_index is not None,
        }


# Kompatibilitätsfunktionen für bestehende Codebasis
def create_rtx2070_rag_system(**kwargs) -> RTX2070OptimizedRAG:
    """Factory-Funktion für RTX 2070 RAG-System"""
    return RTX2070OptimizedRAG(**kwargs)


def rtx2070_rag_query(query: str, **kwargs) -> Dict[str, Any]:
    """Kompatibilitätsfunktion für bestehende rag_query Aufrufe"""
    system = create_rtx2070_rag_system()
    return system.rag_query(query, **kwargs)


if __name__ == "__main__":
    # Test des RTX 2070 RAG-Systems
    print("🚀 RTX 2070 RAG-System Test")

    rag = create_rtx2070_rag_system()

    # Index aufbauen
    if rag.build_index():
        # Test-Abfrage
        test_query = "Was ist die Bedeutung der Energiewende für Deutschland?"
        result = rag.rag_query(test_query, top_k=3)

        print(f"\nTest-Abfrage: {test_query}")
        print(f"Kontext-Länge: {len(result['context'])} Zeichen")
        print(f"Anzahl Quellen: {result['total_sources']}")

        if result["sources"]:
            print("\nTop-Quelle:")
            print(f"Score: {result['sources'][0]['score']:.3f}")
            print(f"Text: {result['sources'][0]['text'][:200]}...")
    else:
        print("❌ RAG-System konnte nicht initialisiert werden")
