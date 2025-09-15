"""
Tests für das hierarchische Memory-System
"""
import pytest
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from hierarchical_memory import (
    HierarchicalMemory,
    MemoryItem,
    AdaptiveMemoryManager,
    EnhancedContextProcessor
)


@pytest.fixture
def temp_memory_path():
    """Temporärer Pfad für Memory-Tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_embedding():
    """Sample embedding für Tests"""
    return np.random.rand(512).astype(np.float32)


@pytest.fixture
def memory_system(temp_memory_path):
    """Hierarchisches Memory-System für Tests"""
    return HierarchicalMemory(
        base_path=temp_memory_path,
        levels=3
    )


def test_memory_item_creation():
    """Test MemoryItem-Erstellung"""
    embedding = np.random.rand(512).astype(np.float32)

    item = MemoryItem(
        content="Test content",
        embedding=embedding,
        timestamp=datetime.now(),
        importance=0.8
    )

    assert item.content == "Test content"
    assert item.importance == 0.8
    assert item.access_count == 0
    assert item.last_accessed is not None


def test_hierarchical_memory_add_memory(memory_system, sample_embedding):
    """Test Hinzufügen von Erinnerungen"""
    # Füge Erinnerung hinzu
    memory_system.add_memory(
        content="Test memory",
        embedding=sample_embedding,
        importance=0.6,
        tags=['test'],
        metadata={'source': 'unittest'}
    )
    
    # Prüfe, dass Erinnerung im Kurzzeitgedächtnis ist
    assert len(memory_system.short_term_memory) == 1
    assert memory_system.short_term_memory[0].content == "Test memory"
    assert memory_system.short_term_memory[0].importance == 0.6


def test_memory_retrieval(memory_system):
    """Test Abrufen relevanter Erinnerungen"""
    # Füge mehrere Erinnerungen hinzu
    embeddings = [np.random.rand(512).astype(np.float32) for _ in range(3)]
    
    memory_system.add_memory("Memory 1", embeddings[0], 0.8)
    memory_system.add_memory("Memory 2", embeddings[1], 0.6)
    memory_system.add_memory("Memory 3", embeddings[2], 0.4)
    
    # Suche mit ähnlichem Embedding
    query_embedding = embeddings[0] + np.random.normal(0, 0.1, 512)
    retrieved = memory_system.retrieve_memories(query_embedding, top_k=2)
    
    assert len(retrieved) <= 2
    assert all(isinstance(item, MemoryItem) for item in retrieved)


def test_memory_capacity_management(memory_system, sample_embedding):
    """Test Kapazitätsverwaltung"""
    # Füge mehr Erinnerungen hinzu als Kapazität
    for i in range(15):  # Mehr als short_term_capacity (10)
        memory_system.add_memory(
            content=f"Memory {i}",
            embedding=np.random.rand(512).astype(np.float32),
            importance=0.5 + (i * 0.01)  # Steigende Wichtigkeit
        )
    
    # Prüfe, dass Kapazität eingehalten wird
    assert len(memory_system.short_term_memory) <= memory_system.short_term_capacity


def test_memory_consolidation(memory_system):
    """Test Konsolidierung ins Langzeitgedächtnis"""
    # Füge wichtige Erinnerung hinzu
    memory_system.add_memory(
        content="Important memory",
        embedding=np.random.rand(512).astype(np.float32),
        importance=0.9  # Hohe Wichtigkeit
    )
    
    # Prüfe initiale Anzahl
    initial_total = len(memory_system.short_term_memory) + len(memory_system.long_term_memory)
    
    # Simuliere Zugriffe für Konsolidierung
    if len(memory_system.short_term_memory) > 0:
        memory = memory_system.short_term_memory[0]
        memory.access_count = 5
        
        # Trigger Konsolidierung
        memory_system._consolidate_memories()
    
    # Prüfe dass Gedächtnisse noch vorhanden sind
    final_total = len(memory_system.short_term_memory) + len(memory_system.long_term_memory)
    assert final_total >= initial_total


def test_search_by_tags(memory_system):
    """Test Tag-basierte Suche"""
    memory_system.add_memory(
        "Content with politics tag",
        np.random.rand(512).astype(np.float32),
        0.7,
        tags=['politics', 'government']
    )
    
    memory_system.add_memory(
        "Content with economics tag",
        np.random.rand(512).astype(np.float32),
        0.6,
        tags=['economics', 'policy']
    )
    
    # Suche nach Tag
    results = memory_system.search_by_tags(['politics'])
    assert len(results) == 1
    assert 'politics' in results[0].tags


def test_search_by_content(memory_system):
    """Test inhaltsbasierte Suche"""
    memory_system.add_memory(
        "This is about climate change policy",
        np.random.rand(512).astype(np.float32),
        0.8
    )
    
    memory_system.add_memory(
        "Economic growth strategies",
        np.random.rand(512).astype(np.float32),
        0.7
    )
    
    # Fuzzy search
    results = memory_system.search_by_content("climate", fuzzy=True)
    assert len(results) == 1
    assert "climate" in results[0].content.lower()


def test_forget_old_memories(memory_system):
    """Test Vergessen alter Erinnerungen"""
    # Füge alte Erinnerung hinzu
    old_memory = MemoryItem(
        content="Old memory",
        embedding=np.random.rand(512).astype(np.float32),
        timestamp=datetime.now() - timedelta(days=35),  # 35 Tage alt
        importance=0.3  # Niedrige Wichtigkeit
    )
    
    memory_system.add_item(old_memory, level=0)  # Füge zu level 0 hinzu
    
    # Vergesse alte Erinnerungen
    forgotten_count = memory_system.forget_old_memories(max_age_days=30)
    
    assert forgotten_count >= 1


def test_memory_stats(memory_system):
    """Test Gedächtnisstatistiken"""
    # Füge Erinnerungen hinzu
    for i in range(3):
        memory_system.add_memory(
            f"Memory {i}",
            np.random.rand(512).astype(np.float32),
            0.5 + i * 0.1
        )
    
    stats = memory_system.get_memory_stats()
    
    assert 'short_term_count' in stats
    assert 'long_term_count' in stats
    assert 'total_memories' in stats
    # Die Anzahl kann variieren wegen Konsolidierung
    assert stats['total_memories'] >= 2


def test_memory_persistence(temp_memory_path):
    """Test persistente Speicherung"""
    # Erstelle Memory-System
    memory = HierarchicalMemory(
        base_path=temp_memory_path,
        levels=3
    )

    # Füge Erinnerung hinzu
    embedding = np.random.rand(512).astype(np.float32)
    item = MemoryItem(
        content="Persistent memory",
        embedding=embedding,
        importance=0.8
    )
    memory.add_item(item, level=0)

    # Erstelle neue Instanz und prüfe ob Daten geladen wurden
    memory2 = HierarchicalMemory(base_path=temp_memory_path, levels=3)
    items = memory2.get_items(level=0)
    assert len(items) == 1
    assert items[0].content == "Persistent memory"
    
    # Prüfe dass Gedächtnis geladen wurde
    total_memories = sum(len(level_data['items']) for level_data in memory2.memory_levels.values())
    assert total_memories == 1


def test_adaptive_memory_manager(memory_system):
    """Test Adaptive Memory Manager"""
    manager = AdaptiveMemoryManager(memory_system)
    
    # Test importance calculation
    importance = manager.adaptive_importance_calculation(
        content="Important government policy with detailed explanations",  # Längerer Text
        context={'source': 'official', 'urgency': 0.8},
        user_feedback=0.9
    )
    
    assert importance > 0.5  # Sollte hohe Wichtigkeit haben
    
    # Füge Erinnerungen für Cleanup-Test hinzu
    memory_system.add_memory(
        "Low importance memory",
        np.random.rand(512).astype(np.float32),
        0.1  # Sehr niedrige Wichtigkeit
    )
    
    suggestions = manager.suggest_memory_cleanup()
    assert 'low_importance' in suggestions


def test_enhanced_context_processor(temp_memory_path):
    """Test Enhanced Context Processor"""
    processor = EnhancedContextProcessor(
        memory_path=temp_memory_path,
        embedding_dim=512
    )

    # Füge Kontext hinzu
    embedding = np.random.rand(512).astype(np.float32)
    metadata = {
        'source': 'government',
        'tags': ['policy', 'climate'],
        'urgency': 0.7
    }

    processor.add_context(
        "New comprehensive climate policy announcement with detailed implementation plans",
        embedding,
        metadata
    )

    # Test Suche nach relevantem Kontext
    relevant_contexts = processor.get_relevant_context("climate policy", max_results=5)

    assert len(relevant_contexts) > 0
    assert relevant_contexts[0]['text'] == "New comprehensive climate policy announcement with detailed implementation plans"
    assert relevant_contexts[0]['metadata']['source'] == 'government'

    # Test Memory-Statistiken
    stats = processor.get_memory_stats()
    assert stats['total_contexts'] == 1
    assert stats['embedding_dim'] == 512


def test_cosine_similarity(memory_system):
    """Test Kosinus-Ähnlichkeit Berechnung"""
    vec1 = np.array([1, 0, 0, 0], dtype=np.float32)
    vec2 = np.array([1, 0, 0, 0], dtype=np.float32)
    vec3 = np.array([0, 1, 0, 0], dtype=np.float32)
    
    # Identische Vektoren
    sim1 = memory_system._cosine_similarity(vec1, vec2)
    assert abs(sim1 - 1.0) < 1e-6
    
    # Orthogonale Vektoren
    sim2 = memory_system._cosine_similarity(vec1, vec3)
    assert abs(sim2 - 0.0) < 1e-6


def test_time_weight_calculation(memory_system):
    """Test zeitbasierte Gewichtung"""
    # Neue Erinnerung
    new_memory = MemoryItem(
        content="New memory",
        embedding=np.random.rand(512).astype(np.float32),
        timestamp=datetime.now(),
        importance=0.5
    )
    
    # Alte Erinnerung
    old_memory = MemoryItem(
        content="Old memory",
        embedding=np.random.rand(512).astype(np.float32),
        timestamp=datetime.now() - timedelta(days=5),
        importance=0.5
    )
    
    new_weight = memory_system._calculate_time_weight(new_memory)
    old_weight = memory_system._calculate_time_weight(old_memory)
    
    assert new_weight > old_weight  # Neuere Erinnerungen haben höhere Gewichtung