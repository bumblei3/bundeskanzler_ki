"""
Advanced RAG System 2.0 - Next Generation Retrieval
Hybrid Search mit BM25 + Semantic + German Language Models
"""

import json
import logging
import os
import pickle
import re
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class AdvancedRAGSystem:
    """
    Advanced Retrieval-Augmented Generation System 2.0

    Features:
    - Hybrid Search (BM25 + Semantic)
    - German-optimized Language Models
    - Query Expansion & Preprocessing
    - Intelligent Reranking
    - Performance Optimization
    """

    def __init__(
        self,
        corpus_path: str = None,
        german_model: str = "deepset/gbert-large",
        fallback_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        use_hybrid_search: bool = True,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ):
        """
        Initialisiert das Advanced RAG-System 2.0

        Args:
            corpus_path: Pfad zur Corpus-Datei
            german_model: Primäres deutsches Sprachmodell
            fallback_model: Fallback Modell falls German Model nicht verfügbar
            use_hybrid_search: Aktiviert Hybrid Search (BM25 + Semantic)
            bm25_weight: Gewichtung für BM25 Score (0.0-1.0)
            semantic_weight: Gewichtung für Semantic Score (0.0-1.0)
        """
        self.logger = logging.getLogger(__name__)

        # Pfad Setup
        if corpus_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            corpus_path = os.path.join(project_root, "data", "corpus.json")

        self.corpus_path = corpus_path

        # Model Configuration
        self.german_model = german_model
        self.fallback_model = fallback_model

        # Hybrid Search Configuration
        self.use_hybrid_search = use_hybrid_search
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Models and Indexes
        self.embedding_model = None
        self.semantic_index = None
        self.bm25_index = None

        # Data
        self.corpus_entries = []
        self.embeddings = None
        self.tokenized_corpus = []

        # Config
        self.config = {
            "top_k": 5,
            "similarity_threshold": 0.3,
            "max_context_length": 2048,
            "enable_query_expansion": True,
            "enable_reranking": True,
            "cache_embeddings": True,
        }

        # Performance Metrics
        self.stats = {
            "queries_processed": 0,
            "hybrid_searches": 0,
            "semantic_searches": 0,
            "bm25_searches": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
        }

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialisiert alle Systemkomponenten"""
        try:
            self.logger.info("🚀 Initialisiere Advanced RAG System 2.0...")

            # 1. Load Embedding Model
            self._load_embedding_model()

            # 2. Load and Process Corpus
            self._load_corpus()

            # 3. Build Indexes
            self._build_semantic_index()
            if self.use_hybrid_search:
                self._build_bm25_index()

            self.logger.info("✅ Advanced RAG System 2.0 erfolgreich initialisiert!")

        except Exception as e:
            self.logger.error(f"❌ Fehler bei der RAG-Initialisierung: {str(e)}")
            raise

    def _load_embedding_model(self):
        """Lädt das optimale Embedding-Modell mit CPU Fallback"""
        try:
            # Versuche zuerst deutsches Modell auf CPU
            self.logger.info(f"📡 Lade German Model auf CPU: {self.german_model}")
            self.embedding_model = SentenceTransformer(
                self.german_model, device="cpu"  # Force CPU usage
            )
            self.current_model = self.german_model
            self.logger.info("✅ German Language Model (CPU) erfolgreich geladen!")

        except Exception as e:
            self.logger.warning(f"⚠️ German Model nicht verfügbar: {str(e)}")

            try:
                # Fallback zu multilingual model auf CPU
                self.logger.info(f"🔄 Fallback zu CPU: {self.fallback_model}")
                self.embedding_model = SentenceTransformer(
                    self.fallback_model, device="cpu"  # Force CPU usage
                )
                self.current_model = self.fallback_model
                self.logger.info("✅ Fallback Model (CPU) erfolgreich geladen!")

            except Exception as e:
                self.logger.error(f"❌ Auch Fallback Model fehlgeschlagen: {str(e)}")
                raise

    def _load_corpus(self):
        """Lädt und preprocesst den Corpus"""
        if not os.path.exists(self.corpus_path):
            self.logger.warning(f"⚠️ Corpus nicht gefunden: {self.corpus_path}")
            self._create_default_corpus()
            return

        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Flexibles Format unterstützen
            if isinstance(data, list):
                self.corpus_entries = data
            elif isinstance(data, dict) and "documents" in data:
                self.corpus_entries = data["documents"]
            elif isinstance(data, dict) and "entries" in data:
                self.corpus_entries = data["entries"]
            else:
                raise ValueError("Unbekanntes Corpus-Format")

            self.logger.info(f"📚 Corpus geladen: {len(self.corpus_entries)} Dokumente")

        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden des Corpus: {str(e)}")
            self._create_default_corpus()

    def _create_default_corpus(self):
        """Erstellt einen Standard-Corpus für Tests"""
        self.corpus_entries = [
            {
                "text": "Deutschland hat sich ehrgeizige Klimaziele bis 2030 gesetzt. Bis dahin soll der CO2-Ausstoß um 65% reduziert werden.",
                "source": "Klimaschutzgesetz",
                "category": "Klimapolitik",
            },
            {
                "text": "Die Energiewende ist ein zentraler Baustein der deutschen Klimapolitik. Erneuerbare Energien werden massiv ausgebaut.",
                "source": "Energiewende-Programm",
                "category": "Energie",
            },
            {
                "text": "Der Kohleausstieg ist bis 2038 geplant. Strukturwandel in den Kohleregionen wird mit Milliarden gefördert.",
                "source": "Kohleausstiegsgesetz",
                "category": "Energiepolitik",
            },
            {
                "text": "Die Bundesregierung fördert Elektromobilität mit Kaufprämien und dem Ausbau der Ladeinfrastruktur.",
                "source": "Verkehrswende-Plan",
                "category": "Verkehr",
            },
            {
                "text": "Deutschland will bis 2045 klimaneutral werden. Alle Sektoren müssen ihren Beitrag leisten.",
                "source": "Klimaneutralitätsziel",
                "category": "Klimapolitik",
            },
        ]
        self.logger.info("📝 Standard-Corpus erstellt für Demo-Zwecke")

    def _build_semantic_index(self):
        """Erstellt den semantischen FAISS-Index"""
        try:
            self.logger.info("🔍 Erstelle semantischen Index...")

            # Extrahiere Texte
            texts = [entry["text"] for entry in self.corpus_entries]

            # Erstelle Embeddings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.embeddings = self.embedding_model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=True
                )

            # Normalisiere für Cosine Similarity
            faiss.normalize_L2(self.embeddings)

            # Erstelle FAISS Index
            dimension = self.embeddings.shape[1]
            self.semantic_index = faiss.IndexFlatIP(dimension)
            self.semantic_index.add(self.embeddings.astype("float32"))

            self.logger.info(
                f"✅ Semantischer Index erstellt: {len(texts)} Embeddings, Dimension: {dimension}"
            )

        except Exception as e:
            self.logger.error(f"❌ Fehler beim Erstellen des semantischen Index: {str(e)}")
            raise

    def _build_bm25_index(self):
        """Erstellt den BM25-Index für Keyword-Suche"""
        try:
            self.logger.info("📝 Erstelle BM25-Index...")

            # Tokenisiere Corpus für BM25
            self.tokenized_corpus = []
            for entry in self.corpus_entries:
                tokens = self._tokenize_text(entry["text"])
                self.tokenized_corpus.append(tokens)

            # Erstelle BM25 Index
            self.bm25_index = BM25Okapi(self.tokenized_corpus)

            self.logger.info(f"✅ BM25-Index erstellt: {len(self.tokenized_corpus)} Dokumente")

        except Exception as e:
            self.logger.error(f"❌ Fehler beim Erstellen des BM25-Index: {str(e)}")
            # BM25 ist optional, System kann ohne weiterlaufen
            self.use_hybrid_search = False

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenisiert Text für BM25 (deutsche Sprache optimiert)

        Args:
            text: Zu tokenisierender Text

        Returns:
            Liste von Tokens
        """
        # Grundlegende deutsche Stoppwörter
        german_stopwords = {
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "eines",
            "einer",
            "und",
            "oder",
            "aber",
            "ist",
            "sind",
            "war",
            "waren",
            "wird",
            "werden",
            "haben",
            "hat",
            "hatte",
            "hatten",
            "sein",
            "seine",
            "ihrer",
            "ihren",
            "mit",
            "von",
            "zu",
            "auf",
            "für",
            "durch",
            "über",
            "unter",
            "zwischen",
            "nicht",
            "keine",
            "kein",
            "nur",
            "auch",
            "noch",
            "schon",
            "wieder",
        }

        # Text normalisieren
        text = text.lower()
        # Nur Buchstaben und Zahlen behalten
        text = re.sub(r"[^\w\s]", " ", text)
        # Tokenisieren
        tokens = text.split()
        # Stoppwörter entfernen und kurze Tokens filtern
        tokens = [token for token in tokens if token not in german_stopwords and len(token) > 2]

        return tokens

    def _expand_query(self, query: str) -> str:
        """
        Erweitert Query um Synonyme und verwandte Begriffe

        Args:
            query: Original Query

        Returns:
            Erweiterte Query
        """
        if not self.config.get("enable_query_expansion", True):
            return query

        # Deutsche Synonym-Mappings für Politik/Klima
        synonyms = {
            "klima": ["klimaschutz", "klimawandel", "umwelt", "co2", "emission"],
            "energie": ["energiewende", "strom", "erneuerbar", "solar", "wind"],
            "kohle": ["braunkohle", "steinkohle", "kohlekraft", "kraftwerk"],
            "politik": ["regierung", "bundesregierung", "bundestag", "minister"],
            "deutschland": ["bundesrepublik", "brd", "deutsch"],
            "ziel": ["ziele", "target", "vorgabe", "plan"],
        }

        expanded_terms = []
        query_words = self._tokenize_text(query)

        for word in query_words:
            expanded_terms.append(word)
            if word in synonyms:
                # Füge 1-2 relevante Synonyme hinzu
                expanded_terms.extend(synonyms[word][:2])

        # Entferne Duplikate und behalte Reihenfolge
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        expanded_query = " ".join(unique_terms)

        if expanded_query != query:
            self.logger.debug(f"Query erweitert: '{query}' → '{expanded_query}'")

        return expanded_query

    def retrieve_relevant_documents(
        self, query: str, top_k: Optional[int] = None, use_hybrid: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Hauptfunktion für Dokumenten-Retrieval mit Hybrid Search

        Args:
            query: Suchanfrage
            top_k: Anzahl der Ergebnisse
            use_hybrid: Überschreibt Hybrid Search Setting

        Returns:
            Liste relevanter Dokumente mit Scores
        """
        start_time = datetime.now()

        try:
            # Parameter Setup
            if top_k is None:
                top_k = self.config["top_k"]
            if use_hybrid is None:
                use_hybrid = self.use_hybrid_search

            # Query Preprocessing
            expanded_query = self._expand_query(query)

            # Hybrid Search oder Pure Semantic
            if use_hybrid and self.bm25_index is not None:
                results = self._hybrid_search(expanded_query, top_k)
                self.stats["hybrid_searches"] += 1
            else:
                results = self._semantic_search(expanded_query, top_k)
                self.stats["semantic_searches"] += 1

            # Optional: Reranking
            if self.config.get("enable_reranking", True):
                results = self._rerank_results(query, results)

            # Performance Tracking
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(response_time)

            self.logger.debug(
                f"Query '{query}' → {len(results)} Ergebnisse in {response_time:.3f}s"
            )

            return results

        except Exception as e:
            self.logger.error(f"❌ Fehler bei Dokumenten-Retrieval: {str(e)}")
            return []

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Pure semantische Suche mit Embeddings"""
        try:
            # Query Embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)

            # FAISS Suche
            scores, indices = self.semantic_index.search(query_embedding, top_k)

            # Ergebnisse formatieren
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.corpus_entries):
                    result = self.corpus_entries[idx].copy()
                    result["score"] = float(score)
                    result["rank"] = i + 1
                    result["search_type"] = "semantic"
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"❌ Fehler bei semantischer Suche: {str(e)}")
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25 Keyword-Suche"""
        try:
            # Query tokenisieren
            query_tokens = self._tokenize_text(query)

            # BM25 Scores berechnen
            scores = self.bm25_index.get_scores(query_tokens)

            # Top-K Indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Ergebnisse formatieren
            results = []
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Nur relevante Ergebnisse
                    result = self.corpus_entries[idx].copy()
                    result["score"] = float(scores[idx])
                    result["rank"] = i + 1
                    result["search_type"] = "bm25"
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"❌ Fehler bei BM25-Suche: {str(e)}")
            return []

    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Hybrid Search kombiniert BM25 + Semantic mit RRF (Reciprocal Rank Fusion)
        """
        try:
            # Beide Suchmethoden ausführen
            semantic_results = self._semantic_search(
                query, top_k * 2
            )  # Mehr Ergebnisse für bessere Fusion
            bm25_results = self._bm25_search(query, top_k * 2)

            # RRF (Reciprocal Rank Fusion)
            fused_scores = defaultdict(float)
            doc_data = {}

            # Semantic Scores verarbeiten
            for i, result in enumerate(semantic_results):
                doc_key = result["text"]  # Eindeutiger Identifier
                doc_data[doc_key] = result
                # RRF Formel: 1 / (rank + k), k=60 ist Standard
                rrf_score = 1.0 / (i + 1 + 60)
                fused_scores[doc_key] += self.semantic_weight * rrf_score

            # BM25 Scores verarbeiten
            for i, result in enumerate(bm25_results):
                doc_key = result["text"]
                if doc_key not in doc_data:
                    doc_data[doc_key] = result
                rrf_score = 1.0 / (i + 1 + 60)
                fused_scores[doc_key] += self.bm25_weight * rrf_score

            # Nach kombiniertem Score sortieren
            sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

            # Top-K Ergebnisse formatieren
            results = []
            for i, (doc_key, score) in enumerate(sorted_docs[:top_k]):
                result = doc_data[doc_key].copy()
                result["score"] = float(score)
                result["rank"] = i + 1
                result["search_type"] = "hybrid"
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"❌ Fehler bei Hybrid Search: {str(e)}")
            # Fallback zu semantic search
            return self._semantic_search(query, top_k)

    def _rerank_results(
        self, original_query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reranking basierend auf Originalquery und zusätzlichen Faktoren
        """
        if not results:
            return results

        # Einfaches Reranking: Bevorzuge Ergebnisse mit Query-Begriffen im Text
        query_words = set(self._tokenize_text(original_query))

        for result in results:
            text_words = set(self._tokenize_text(result["text"]))
            overlap = len(query_words.intersection(text_words))
            # Boost Score basierend auf Wort-Überlappung
            boost = 1.0 + (overlap * 0.1)
            result["score"] *= boost
            result["word_overlap"] = overlap

        # Neu sortieren nach geboosteten Scores
        results.sort(key=lambda x: x["score"], reverse=True)

        # Ranks updaten
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return results

    def _update_stats(self, response_time: float):
        """Performance-Statistiken aktualisieren"""
        self.stats["queries_processed"] += 1

        # Rolling average für response time
        alpha = 0.1  # Smoothing factor
        if self.stats["avg_response_time"] == 0:
            self.stats["avg_response_time"] = response_time
        else:
            self.stats["avg_response_time"] = (
                alpha * response_time + (1 - alpha) * self.stats["avg_response_time"]
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Systeminformationen und Statistiken"""
        return {
            "version": "2.0",
            "model": self.current_model,
            "corpus_size": len(self.corpus_entries),
            "corpus_loaded": len(self.corpus_entries) > 0,
            "corpus_entries": len(self.corpus_entries),
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dimension": 384,
            "hybrid_search": self.use_hybrid_search,
            "indexes": {
                "semantic": self.semantic_index is not None,
                "bm25": self.bm25_index is not None,
            },
            "config": self.config,
            "stats": self.stats,
            "weights": {"bm25": self.bm25_weight, "semantic": self.semantic_weight},
        }

    def benchmark_search(self, test_queries: List[str], iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark verschiedener Suchmethoden

        Args:
            test_queries: Liste von Test-Queries
            iterations: Anzahl der Wiederholungen

        Returns:
            Benchmark-Ergebnisse
        """
        results = {
            "semantic": {"times": [], "results": []},
            "bm25": {"times": [], "results": []},
            "hybrid": {"times": [], "results": []},
        }

        self.logger.info(
            f"🔬 Starte Benchmark mit {len(test_queries)} Queries, {iterations} Iterationen"
        )

        for query in test_queries:
            for method in ["semantic", "bm25", "hybrid"]:
                if method == "bm25" and self.bm25_index is None:
                    continue

                method_times = []
                for _ in range(iterations):
                    start = datetime.now()

                    if method == "semantic":
                        docs = self._semantic_search(query, 5)
                    elif method == "bm25":
                        docs = self._bm25_search(query, 5)
                    else:  # hybrid
                        docs = self._hybrid_search(query, 5)

                    elapsed = (datetime.now() - start).total_seconds()
                    method_times.append(elapsed)

                avg_time = sum(method_times) / len(method_times)
                results[method]["times"].append(avg_time)
                results[method]["results"].append(len(docs))

        # Statistiken berechnen
        summary = {}
        for method, data in results.items():
            if data["times"]:
                summary[method] = {
                    "avg_time": sum(data["times"]) / len(data["times"]),
                    "avg_results": (
                        sum(data["results"]) / len(data["results"]) if data["results"] else 0
                    ),
                    "queries": len(data["times"]),
                }

        self.logger.info("✅ Benchmark abgeschlossen")
        return {
            "queries": test_queries,
            "iterations": iterations,
            "detailed": results,
            "summary": summary,
        }


# Backward compatibility - Alias für bestehenden Code
RAGSystemV2 = AdvancedRAGSystem
