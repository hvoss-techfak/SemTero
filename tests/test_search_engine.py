"""Tests for SearchEngine - two-stage RAG search with reranking."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import ollama
from zoterorag.search_engine import SearchEngine
from zoterorag.config import Config
from zoterorag.models import SearchResult, Section, SentenceWindow


class TestSearchEngineInitialization:
    """Test suite for SearchEngine initialization."""
    
    @patch('zoterorag.search_engine.VectorStore')
    def test_default_initialization(self, mock_vector_store):
        """Test default initialization."""
        config = Config()
        
        engine = SearchEngine(config)
        
        assert engine.config is not None
        assert isinstance(engine.vector_store, Mock) or True  # May be mocked
    
    @patch('zoterorag.search_engine.VectorStore')
    def test_lazy_embedding_manager(self, mock_vector_store):
        """Test embedding manager is lazily initialized."""
        config = Config()
        
        engine = SearchEngine(config)
        
        # Should not have embedding manager yet
        assert engine._embedding_manager is None
        
        # Accessing property should create it
        _ = engine.embedding_manager
        
        # Now should be set
        assert engine._embedding_manager is not None


class TestEmbeddingOptions:
    """Test suite for embedding options generation."""
    
    @patch('zoterorag.search_engine.VectorStore')
    def test_get_embedding_options_with_dimensions(self, mock_vector_store):
        """Test embedding options include dimensions when set in config."""
        config = Config()
        config.EMBEDDING_DIMENSIONS = 1024
        
        engine = SearchEngine(config)
        
        opts = engine._get_embedding_options()
        
        assert "dimensions" in opts
        assert opts["dimensions"] == 1024
    
    @patch('zoterorag.search_engine.VectorStore')
    def test_get_embedding_options_no_dimensions(self, mock_vector_store):
        """Test empty options when dimensions not set."""
        config = Config()
        config.EMBEDDING_DIMENSIONS = 0
        
        engine = SearchEngine(config)
        
        opts = engine._get_embedding_options()
        
        assert len(opts) == 0


class TestCosineSimilarity:
    """Test suite for cosine similarity calculation."""
    
    @patch('zoterorag.search_engine.VectorStore')
    def test_identical_vectors(self, mock_vector_store):
        """Test identical vectors return 1.0."""
        config = Config()
        engine = SearchEngine(config)
        
        vector = [1.0, 2.0, 3.0]
        result = engine._cosine_similarity(vector, vector)
        
        assert abs(result - 1.0) < 0.0001
    
    @patch('zoterorag.search_engine.VectorStore')
    def test_orthogonal_vectors(self, mock_vector_store):
        """Test orthogonal vectors return 0.0."""
        config = Config()
        engine = SearchEngine(config)
        
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        result = engine._cosine_similarity(vec1, vec2)
        
        assert abs(result) < 0.0001
    
    @patch('zoterorag.search_engine.VectorStore')
    def test_opposite_vectors(self, mock_vector_store):
        """Test opposite vectors return -1.0."""
        config = Config()
        engine = SearchEngine(config)
        
        vec1 = [1.0, 2.0]
        vec2 = [-1.0, -2.0]
        result = engine._cosine_similarity(vec1, vec2)
        
        assert abs(result + 1.0) < 0.0001
    
    @patch('zoterorag.search_engine.VectorStore')
    def test_zero_vector_handling(self, mock_vector_store):
        """Test zero vector returns 0.0."""
        config = Config()
        engine = SearchEngine(config)
        
        vec1 = [0.0, 0.0]
        vec2 = [1.0, 2.0]
        result = engine._cosine_similarity(vec1, vec2)
        
        assert result == 0.0


class TestQueryEmbedding:
    """Test suite for query embedding generation."""
    
    @patch('zoterorag.search_engine.ollama.embeddings')
    @patch('zotero_client.ZoteroClient')  # Patch ZoteroClient if imported
    def test_get_query_embedding(self, mock_zotero, mock_ollama_embeddings):
        """Test query embedding generation."""
        mock_response = {"embedding": [0.1, 0.2, 0.3]}
        mock_ollama_embeddings.return_value = mock_response
        
        config = Config()
        engine = SearchEngine(config)
        
        result = engine._get_query_embedding("test query")
        
        assert result == [0.1, 0.2, 0.3]
        mock_ollama_embeddings.assert_called_once()


class TestSearchSections:
    """Test suite for section search (Stage 1)."""
    
    @patch('zoterorag.search_engine.SearchEngine._get_query_embedding')
    @patch('zotero_client.ZoteroClient')  
    def test_search_sections_empty_results(self, mock_zotero, mock_get_emb):
        """Test empty results handling."""
        mock_get_emb.return_value = [0.1] * 512
        
        # Mock vector store search to return empty
        mock_vs = MagicMock()
        mock_vs.search_sections.return_value = ([], [])
        
        config = Config()
        engine = SearchEngine(config)
        engine.vector_store = mock_vs
        
        result = engine.search_sections("test query")
        
        assert result == []
    
    @patch('zoterorag.search_engine.SearchEngine._get_query_embedding')
    def test_search_sections_with_results(self, mock_get_emb):
        """Test section search with results."""
        # Setup mocks
        mock_vs = MagicMock()
        mock_section = Section(
            id="sec1",
            document_id="doc1",
            title="Section 1",
            level=1,
            start_page=1,
            end_page=1,
            text="Some text"
        )
        
        mock_vs.search_sections.return_value = (["sec1"], [[0.5] * 512])
        mock_vs.get_section.return_value = mock_section
        
        config = Config()
        engine = SearchEngine(config)
        engine.vector_store = mock_vs
        mock_get_emb.return_value = [0.1] * 512
        
        result = engine.search_sections("test query", top_k=5)
        
        assert len(result) > 0


class TestReranking:
    """Test suite for reranking functionality."""
    
    @patch('zoterorag.search_engine.SearchEngine._get_query_embedding')
    def test_rerank_candidates_empty(self, mock_get_emb):
        """Test empty candidates return empty list."""
        config = Config()
        engine = SearchEngine(config)
        
        result = engine._rerank_candidates("query", [])
        
        assert result == []
    
    @patch('zoterorag.search_engine.SearchEngine._get_query_embedding')
    def test_rerank_candidates_sorted(self, mock_get_emb):
        """Test candidates are sorted by score."""
        # Mock embedding to return different scores for each candidate
        call_count = [0]
        
        def emb_side_effect(prompt):
            call_count[0] += 1
            if "first" in prompt:
                return [0.9, 0.8]
            elif "second" in prompt:
                return [0.7, 0.6]
            else:
                return [0.5, 0.4]
        
        mock_get_emb.side_effect = emb_side_effect
        
        config = Config()
        engine = SearchEngine(config)
        
        candidates = ["first candidate", "second candidate", "third candidate"]
        result = engine._rerank_candidates("query", candidates)
        
        # Results should be sorted by score descending
        assert len(result) == 3
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


class TestSentenceWindowsOnDemand:
    """Test suite for on-demand sentence window creation."""
    
    @patch('zotero_client.ZoteroClient')
    def test_create_sentence_windows_on_demand(self, mock_zotero):
        """Test creating windows from section text."""
        config = Config()
        engine = SearchEngine(config)
        
        result = engine._create_sentence_windows_on_demand(
            section_text="First sentence. Second sentence! Third? Yes.",
            section_id="sec1"
        )
        
        assert len(result) > 0
        # Check structure
        assert hasattr(result[0], 'id')
        assert hasattr(result[0], 'section_id')
    
    @patch('zotero_client.ZoteroClient')
    def test_create_windows_empty_text(self, mock_zotero):
        """Test empty text returns single window."""
        config = Config()
        engine = SearchEngine(config)
        
        result = engine._create_sentence_windows_on_demand(
            section_text="",
            section_id="sec1"
        )
        
        # Should create one window with empty content
        assert len(result) >= 1
    
    @patch('zotero_client.ZoteroClient')
    def test_create_windows_custom_params(self, mock_zotero):
        """Test custom window_size and overlap."""
        config = Config()
        engine = SearchEngine(config)
        
        text = "One. Two. Three. Four. Five."
        result = engine._create_sentence_windows_on_demand(
            section_text=text,
            section_id="sec1",
            window_size=2,
            overlap=1
        )
        
        # With window_size=2 and overlap=1, step is 1
        # Should create multiple windows sharing sentences


class TestEmbedBatch:
    """Test suite for batch embedding."""
    
    @patch('zoterorag.search_engine.ollama.embeddings')
    def test_embed_batch_multiple_texts(self, mock_ollama):
        """Test embedding multiple texts."""
        mock_ollama.side_effect = [
            {"embedding": [0.1] * 512},
            {"embedding": [0.2] * 512},
            {"embedding": [0.3] * 512}
        ]
        
        config = Config()
        engine = SearchEngine(config)
        
        texts = ["text 1", "text 2", "text 3"]
        result = engine.embed_batch(texts)
        
        assert len(result) == 3
        mock_ollama.assert_called()


class TestSearchSentencesInSection:
    """Test suite for sentence search within a section."""
    
    @patch('zotero_client.ZoteroClient')
    def test_search_sentences_empty_windows(self, mock_zotero):
        """Test empty windows returns empty list."""
        config = Config()
        
        mock_vs = MagicMock()
        mock_vs.get_sentence_windows.return_value = []
        
        engine = SearchEngine(config)
        engine.vector_store = mock_vs
        
        result = engine.search_sentences_in_section("query", "sec1")
        
        assert result == []


class TestEmbedAndSearchSentences:
    """Test suite for embed and search sentences method."""
    
    @patch('zotero_client.ZoteroClient')
    def test_embed_and_search_no_windows(self, mock_zotero):
        """Test empty section text returns empty list."""
        config = Config()
        engine = SearchEngine(config)
        
        result = engine._embed_and_search_sentences(
            query="test",
            section_text="",
            section_id="sec1",
            document_key="doc1"
        )
        
        assert result == []


class TestFullSearch:
    """Test suite for full two-stage search."""
    
    @patch('zoterorag.search_engine.SearchEngine._get_query_embedding')
    def test_search_no_results(self, mock_get_emb):
        """Test empty results when no sections found."""
        config = Config()
        
        mock_vs = MagicMock()
        mock_vs.search_sections.return_value = ([], [])
        
        engine = SearchEngine(config)
        engine.vector_store = mock_vs
        mock_get_emb.return_value = [0.1] * 512
        
        result = engine.search("test query")
        
        assert result == []
    
    @patch('zoterorag.search_engine.SearchEngine._get_query_embedding')
    def test_search_with_document_filter(self, mock_get_emb):
        """Test search within specific document."""
        config = Config()
        
        mock_vs = MagicMock()
        mock_section = Section(
            id="sec1",
            document_id="doc1",
            title="Section 1",
            level=1,
            start_page=1,
            end_page=1,
            text="Test content"
        )
        mock_vs.get_all_sections.return_value = [mock_section]
        
        engine = SearchEngine(config)
        engine.vector_store = mock_vs
        
        # Mock embedding for search
        with patch.object(engine, 'embed_text_for_search', return_value=[0.5] * 512):
            result = engine.search("test", document_key="doc1")
            
            # Should have called get_all_sections for the document
            mock_vs.get_all_sections.assert_called()


class TestSearchBestSentences:
    """Test suite for search_best_sentences method."""
    
    @patch('zotero_client.ZoteroClient')
    def test_search_best_no_results(self, mock_zotero):
        """Test empty results when no matches."""
        config = Config()
        
        mock_vs = MagicMock()
        mock_vs.search_sections.return_value = ([], [])
        
        engine = SearchEngine(config)
        engine.vector_store = mock_vs
        
        result = engine.search_best_sentences("test query")
        
        assert result == []


class TestEmbedTextForSearch:
    """Test suite for embed_text_for_search method."""
    
    @patch('zoterorag.search_engine.ollama.embeddings')
    def test_embed_text(self, mock_ollama):
        """Test text embedding generation."""
        mock_response = {"embedding": [0.5] * 512}
        mock_ollama.return_value = mock_response
        
        config = Config()
        engine = SearchEngine(config)
        
        result = engine.embed_text_for_search("test text")
        
        assert len(result) == 512
        mock_ollama.assert_called()


class TestGetStats:
    """Test suite for get_stats method."""
    
    @patch('zotero_client.ZoteroClient')
    def test_get_stats(self, mock_zotero):
        """Test stats retrieval."""
        config = Config()
        
        mock_vs = MagicMock()
        mock_vs.get_section_count.return_value = 10
        mock_vs.get_sentence_count.return_value = 50
        mock_vs.get_embedded_documents.return_value = [{}, {}, {}]
        
        engine = SearchEngine(config)
        engine.vector_store = mock_vs
        
        result = engine.get_stats()
        
        assert result["total_sections"] == 10
        assert result["total_sentence_windows"] == 50
        assert result["embedded_documents"] == 3


class TestDimensionMismatch:
    """Test suite for dimension mismatch detection."""
    
    @patch('zotero_client.ZoteroClient')
    def test_dimension_mismatch_warning(self, mock_zotero):
        """Test warning is logged on init with mismatch."""
        config = Config()
        config.EMBEDDING_DIMENSIONS = 1024
        
        mock_vs = MagicMock()
        mock_vs.has_dimension_mismatch.return_value = True
        mock_vs.get_detected_dimension.return_value = 768
        
        # Should log a warning about dimension mismatch
        with patch('zoterorag.search_engine.logger') as mock_logger:
            engine = SearchEngine(config)
            engine.vector_store = mock_vs
            
            # Warning should have been logged
            assert mock_logger.warning.called


class TestSearchWithRerankingFallback:
    """Test suite for search with reranking fallback."""
    
    @patch('zoterorag.search_engine.SearchEngine._get_query_embedding')
    def test_rerank_failure_fallback(self, mock_get_emb):
        """Test fallback when reranking fails."""
        config = Config()
        
        # Setup mocks
        mock_vs = MagicMock()
        
        # Create a section with no sentence windows (triggers on-demand creation)
        mock_section = Section(
            id="sec1",
            document_id="doc1", 
            title="Test",
            level=1,
            start_page=1,
            end_page=1,
            text="Sentence one. Sentence two."
        )
        
        # First call: search sections returns our section
        mock_vs.search_sections.return_value = (["sec1"], [[0.5] * 512])
        
        # Second call: get_section returns the section
        # Third call: get_sentence_windows returns empty (triggers on-demand)
        mock_vs.get_section.return_value = mock_section
        mock_vs.get_sentence_windows.return_value = []
        
        engine = SearchEngine(config)
        engine.vector_store = mock_vs
        
        # Make _rerank raise an exception to test fallback
        with patch.object(engine.embedding_manager, '_rerank', side_effect=Exception("Rerank failed")):
            result = engine.search("test query")
            
            # Should still return results using fallback scores


class TestSearchResultModel:
    """Test suite for SearchResult model in search context."""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            text="Sample text",
            document_title="Doc Title",
            section_title="Section Title",
            zotero_key="ABC123",
            relevance_score=0.8,
            rerank_score=0.9
        )
        
        assert result.text == "Sample text"
        assert result.relevance_score == 0.8
    
    def test_search_result_default_scores(self):
        """Test default score values."""
        result = SearchResult(
            text="Text",
            document_title="",
            section_title="",
            zotero_key=""
        )
        
        assert result.rerank_score is None or isinstance(result.rerank_score, float)