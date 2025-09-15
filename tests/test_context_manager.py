import pytest
import numpy as np
from context_manager import ContextManager

def test_add_and_retrieve_interaction():
    cm = ContextManager(window_size=3)
    emb = np.array([1.0, 0.0, 0.0])
    cm.add_interaction("Frage1", "Antwort1", emb, topic="Politik")
    cm.add_interaction("Frage2", "Antwort2", emb, topic="Wirtschaft")
    cm.add_interaction("Frage3", "Antwort3", emb, topic="Politik")
    assert len(cm.interaction_history) == 3
    assert cm.get_current_topic() == "Politik"
    assert "Politik" in cm.get_topic_context()

def test_get_relevant_context():
    cm = ContextManager(window_size=3, min_context_similarity=0.1)
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    cm.add_interaction("Q1", "A1", emb1, topic="A")
    cm.add_interaction("Q2", "A2", emb2, topic="B")
    query_emb = np.array([1.0, 0.0, 0.0])
    relevant = cm.get_relevant_context(query_emb, top_k=2)
    assert len(relevant) >= 1
    assert relevant[0]["query"] == "Q1"

def test_summarize_context():
    cm = ContextManager(window_size=2)
    emb = np.array([0.5, 0.5, 0.5])
    cm.add_interaction("F1", "A1", emb, topic="T1")
    summary = cm.summarize_context()
    assert summary["current_topic"] == "T1"
    assert summary["interaction_count"] == 1
    assert summary["last_interaction"]["query"] == "F1"

def test_empty_context():
    cm = ContextManager()
    assert cm.get_relevant_context(np.array([1,2,3])) == []
    assert cm.get_current_topic() is None
    assert cm.summarize_context()["interaction_count"] == 0
