"""
Tests für memory_optimizer.py - Memory-Optimierung und Dateiverarbeitung
"""

import os
import tempfile
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from memory_optimizer import (
    ChunkedProcessor,
    LazyFileReader,
    MemoryMappedCorpus,
    MemoryOptimizer,
    optimize_tensorflow_memory,
    setup_memory_optimization,
)


class TestMemoryOptimizer:
    """Tests für MemoryOptimizer Klasse"""

    @patch("memory_optimizer.psutil")
    def test_get_memory_usage(self, mock_psutil):
        """Test get_memory_usage Methode"""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100 MB
        mock_psutil.Process.return_value = mock_process

        optimizer = MemoryOptimizer()
        usage = optimizer.get_memory_usage()

        assert usage == 100.0  # Should return MB

    def test_force_garbage_collection(self):
        """Test force_garbage_collection Methode"""
        optimizer = MemoryOptimizer()
        optimizer.force_garbage_collection()  # Should not raise any exceptions

    def test_optimize_numpy_arrays(self):
        """Test optimize_numpy_arrays Methode"""
        try:
            import numpy as np

            # Create some numpy arrays
            arrays = [np.ones(1000) for _ in range(5)]
            array_objects = arrays.copy()  # Keep references
        except AttributeError:
            # If numpy is stubbed, skip this test
            pytest.skip("NumPy not available for this test")

        optimizer = MemoryOptimizer()
        result = optimizer.optimize_numpy_arrays()

        assert isinstance(result, float)
        assert result >= 0

    def test_memory_context(self):
        """Test memory_context context manager"""
        optimizer = MemoryOptimizer()

        with optimizer.memory_context("test_operation"):
            # Should not raise any exceptions
            pass


class TestLazyFileReader:
    """Tests für LazyFileReader Klasse"""

    def test_init(self):
        """Test LazyFileReader Initialisierung"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            temp_file = f.name

        try:
            reader = LazyFileReader(temp_file, chunk_size=100)
            assert reader.file_path == temp_file
            assert reader.chunk_size == 100
            assert reader.file_size > 0
        finally:
            os.unlink(temp_file)

    def test_read_lines_lazy(self):
        """Test read_lines_lazy Methode"""
        content = "line1\nline2\nline3\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            reader = LazyFileReader(temp_file)
            lines = list(reader.read_lines_lazy())

            assert lines == ["line1", "line2", "line3"]
        finally:
            os.unlink(temp_file)

    def test_read_lines_lazy_file_not_found(self):
        """Test read_lines_lazy mit nicht existierender Datei"""
        reader = LazyFileReader("nonexistent_file.txt")
        lines = list(reader.read_lines_lazy())

        assert lines == []  # Should return empty iterator

    def test_read_chunks(self):
        """Test read_chunks Methode"""
        content = "line1\nline2\nline3\nline4\nline5\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            reader = LazyFileReader(temp_file)
            chunks = list(reader.read_chunks(chunk_size=2))

            assert len(chunks) == 3  # 5 lines / 2 chunk_size = 3 chunks (last partial)
            assert chunks[0] == ["line1", "line2"]
            assert chunks[1] == ["line3", "line4"]
            assert chunks[2] == ["line5"]
        finally:
            os.unlink(temp_file)

    def test_read_chunks_default_size(self):
        """Test read_chunks mit Standard-Chunk-Größe"""
        content = "line1\n" * 100  # 100 lines
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            reader = LazyFileReader(temp_file, chunk_size=10)
            chunks = list(reader.read_chunks())  # Should use default chunk_size

            assert len(chunks) == 10  # 100 lines / 10 chunk_size = 10 chunks
            assert len(chunks[0]) == 10
        finally:
            os.unlink(temp_file)


class TestMemoryMappedCorpus:
    """Tests für MemoryMappedCorpus Klasse"""

    def test_init_with_existing_file(self):
        """Test MemoryMappedCorpus Initialisierung mit existierender Datei"""
        content = "line1\nline2\nline3\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            corpus = MemoryMappedCorpus(temp_file)
            assert corpus.corpus_file == temp_file
            assert len(corpus.line_positions) == 3
            assert corpus.lines == []
        finally:
            corpus.close()
            os.unlink(temp_file)

    def test_init_with_nonexistent_file(self):
        """Test MemoryMappedCorpus Initialisierung mit nicht existierender Datei"""
        corpus = MemoryMappedCorpus("nonexistent_file.txt")
        assert corpus.line_positions == []
        assert corpus.lines == []

    def test_get_line(self):
        """Test get_line Methode"""
        content = "line1\nline2\nline3\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            corpus = MemoryMappedCorpus(temp_file)

            assert corpus.get_line(0) == "line1"
            assert corpus.get_line(1) == "line2"
            assert corpus.get_line(2) == "line3"
            assert corpus.get_line(3) is None  # Out of bounds
        finally:
            corpus.close()
            os.unlink(temp_file)

    def test_len(self):
        """Test __len__ Methode"""
        content = "line1\nline2\nline3\nline4\nline5\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            corpus = MemoryMappedCorpus(temp_file)
            assert len(corpus) == 5
        finally:
            corpus.close()
            os.unlink(temp_file)


class TestChunkedProcessor:
    """Tests für ChunkedProcessor Klasse"""

    def test_init(self):
        """Test ChunkedProcessor Initialisierung"""
        processor = ChunkedProcessor(chunk_size=500)
        assert processor.chunk_size == 500
        assert isinstance(processor.memory_optimizer, MemoryOptimizer)

    def test_init_with_custom_optimizer(self):
        """Test ChunkedProcessor mit custom MemoryOptimizer"""
        custom_optimizer = MemoryOptimizer()
        processor = ChunkedProcessor(chunk_size=100, memory_optimizer=custom_optimizer)
        assert processor.memory_optimizer == custom_optimizer

    def test_process_in_chunks(self):
        """Test process_in_chunks Methode"""
        processor = ChunkedProcessor(chunk_size=2)

        data = [1, 2, 3, 4, 5]
        results = []

        def test_processor(chunk):
            results.extend([x * 2 for x in chunk])
            return [x * 2 for x in chunk]

        result = processor.process_in_chunks(data, test_processor)

        assert result == [2, 4, 6, 8, 10]
        assert results == [2, 4, 6, 8, 10]

    def test_process_in_chunks_with_args_kwargs(self):
        """Test process_in_chunks mit zusätzlichen Argumenten"""
        processor = ChunkedProcessor(chunk_size=2)

        data = [1, 2, 3, 4]
        results = []

        def test_processor(chunk, multiplier=1, offset=0):
            processed = [(x * multiplier) + offset for x in chunk]
            results.extend(processed)
            return processed

        result = processor.process_in_chunks(data, test_processor, multiplier=3, offset=10)

        assert result == [13, 16, 19, 22]  # (1*3)+10, (2*3)+10, (3*3)+10, (4*3)+10
        assert results == [13, 16, 19, 22]


class TestGlobalFunctions:
    """Tests für globale Funktionen"""

    def test_setup_memory_optimization(self):
        """Test setup_memory_optimization"""
        # Should not raise any exceptions
        setup_memory_optimization()
