"""Tests for VectorStore class."""

import sys
import os

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from zoterorag.models import Section, SentenceWindow


class TestVectorStore:
    """Test suite for VectorStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            
            # Mock collections
            sections_collection = MagicMock()
            sentences_collection = MagicMock()
            
            mock_client.get_collection.return_value = sections_collection
            mock_client.create_collection.side_effect = lambda name: (
                sections_collection if name == "sections" else sentences_collection
            )
            
            # Setup count and get methods
            type(sections_collection).count = PropertyMock(return_value=0)
            type(sentences_collection).count = PropertyMock(return_value=0)
            
            yield mock_client, sections_collection, sentences_collection

    @pytest.fixture
    def vector_store(self, temp_dir, mock_chroma_client):
        """Create a VectorStore instance with mocked ChromaDB."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            yield store

    # --- Test _detect_dimensions ---

    def test_detect_dimensions_with_existing_embeddings(self, temp_dir):
        """Test dimension detection when embeddings exist."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            
            # Create mock collections
            sections_collection = MagicMock()
            sentences_collection = MagicMock()
            
            def get_collection_side_effect(name):
                return sections_collection if name == "sections" else sentences_collection
            
            mock_client.get_collection.side_effect = get_collection_side_effect
            
            # Mock embeddings with dimension 384
            import numpy as np
            mock_embedding = np.array([0.1] * 384)
            
            sections_collection.get.return_value = {
                "ids": ["sec_1"],
                "embeddings": [mock_embedding]
            }
            sentences_collection.get.return_value = {
                "ids": ["win_1"],
                "embeddings": [mock_embedding]
            }
            
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            assert store._detected_section_dim == 384
            assert store._detected_sentence_dim == 384

    def test_detect_dimensions_with_empty_collections(self, temp_dir):
        """Test dimension detection with empty collections."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            
            sections_collection = MagicMock()
            sentences_collection = MagicMock()
            
            # Empty embeddings
            sections_collection.get.return_value = {"ids": [], "embeddings": []}
            sentences_collection.get.return_value = {"ids": [], "embeddings": None}
            
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            assert store._detected_section_dim is None

    # --- Test get_detected_dimension ---

    def test_get_detected_dimension_returns_stored_value(self, vector_store):
        """Test that get_detected_dimension returns the detected value."""
        vector_store._detected_section_dim = 768
        assert vector_store.get_detected_dimension() == 768
        
        vector_store._detected_section_dim = None
        assert vector_store.get_detected_dimension() is None

    # --- Test has_dimension_mismatch ---

    def test_has_dimension_mismatch_returns_true_when_different(self, vector_store):
        """Test dimension mismatch detection when dimensions differ."""
        vector_store._detected_section_dim = 384
        assert vector_store.has_dimension_mismatch(768) is True

    def test_has_dimension_mismatch_returns_false_when_same(self, vector_store):
        """Test dimension mismatch returns false when dimensions match."""
        vector_store._detected_section_dim = 768
        assert vector_store.has_dimension_mismatch(768) is False

    def test_has_dimension_mismatch_returns_false_when_no_existing_data(self, vector_store):
        """Test dimension mismatch returns false when no existing data."""
        vector_store._detected_section_dim = None
        assert vector_store.has_dimension_mismatch(768) is False

    # --- Test get_embedded_documents ---

    def test_get_embedded_documents_returns_dict(self, temp_dir):
        """Test that get_embedded_documents returns parsed JSON dict."""
        embedded_docs = {"doc1": 5, "doc2": 10}
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))

            # Create embedded_docs.json file
            meta_path = temp_dir / "embedded_docs.json"
            with open(meta_path, "w") as f:
                json.dump(embedded_docs, f)
            
            result = store.get_embedded_documents()
            assert result == embedded_docs

    def test_get_embedded_documents_returns_empty_when_file_missing(self, temp_dir):
        """Test that get_embedded_documents returns empty dict when file doesn't exist."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))

            result = store.get_embedded_documents()
            assert result == {}

    def test_get_embedded_documents_returns_empty_on_corrupt_file(self, temp_dir):
        """Test that get_embedded_documents handles corrupt JSON gracefully."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))

            # Create corrupted embedded_docs.json file
            meta_path = temp_dir / "embedded_docs.json"
            with open(meta_path, "w") as f:
                f.write("{ invalid json }")
            
            result = store.get_embedded_documents()
            assert result == {}

    def test_get_embedded_documents_handles_empty_file(self, temp_dir):
        """Test that get_embedded_documents handles empty file."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))

            # Create empty embedded_docs.json file
            meta_path = temp_dir / "embedded_docs.json"
            with open(meta_path, "w") as f:
                f.write("")
            
            result = store.get_embedded_documents()
            assert result == {}

    # --- Test save_embedded_documents ---

    def test_save_embedded_documents_writes_to_file(self, temp_dir):
        """Test that save_embedded_documents writes the dict to JSON file."""
        docs = {"doc1": 5, "doc2": 10}

        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            store.save_embedded_documents(docs)
            
            meta_path = temp_dir / "embedded_docs.json"
            with open(meta_path, "r") as f:
                saved_data = json.load(f)
            
            assert saved_data == docs

    def test_save_embedded_documents_skips_empty_when_flag_false(self, temp_dir):
        """Test that save_embedded_documents skips saving empty dict when allow_empty=False."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            # Should not raise, but should also not create file
            store.save_embedded_documents({}, allow_empty=False)
            
            meta_path = temp_dir / "embedded_docs.json"
            assert not meta_path.exists()

    def test_save_embedded_documents_overwrites_existing(self, temp_dir):
        """Test that save_embedded_documents overwrites existing data."""
        docs = {"doc1": 5}

        # Create initial file
        meta_path = temp_dir / "embedded_docs.json"
        with open(meta_path, "w") as f:
            json.dump({"old_doc": 3}, f)

        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            store.save_embedded_documents(docs)
            
            with open(meta_path, "r") as f:
                saved_data = json.load(f)
            
            assert saved_data == docs

    # --- Test update_embedded_document ---

    def test_update_embedded_document_adds_new_entry(self, temp_dir):
        """Test that update_embedded_document adds a new document."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            store.update_embedded_document("doc1", 5)
            
            result = store.get_embedded_documents()
            assert result == {"doc1": 5}

    def test_update_embedded_document_updates_existing(self, temp_dir):
        """Test that update_embedded_document updates existing entry."""
        # Create initial file
        meta_path = temp_dir / "embedded_docs.json"
        with open(meta_path, "w") as f:
            json.dump({"doc1": 3}, f)

        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            store.update_embedded_document("doc1", 10)
            
            result = store.get_embedded_documents()
            assert result == {"doc1": 10}

    # --- Test add_sections ---

    def test_add_sections_with_valid_data(self, temp_dir):
        """Test adding sections with embeddings."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sections_collection = MagicMock()

        def get_or_create(name):
            return sections_collection

        mock_client.get_or_create_collection.side_effect = get_or_create
        type(sections_collection).count = PropertyMock(return_value=0)

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            sections = [
                Section(
                    id="sec_1",
                    document_id="doc1",
                    title="Introduction",
                    level=1,
                    start_page=1,
                    end_page=2,
                    text="This is the introduction."
                )
            ]
            embeddings = [[0.1] * 384]
            
            store.add_sections(sections, embeddings, "doc1", "zotero_key_1", "parent_item")

            sections_collection.upsert.assert_called_once()

    def test_add_sections_with_empty_data(self, temp_dir):
        """Test that add_sections handles empty lists gracefully."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            # Should not raise
            store.add_sections([], [], "doc1", "zotero_key_1", "parent_item")
            store.add_sections(None, None, "doc1", "zotero_key_1", "parent_item")

    # --- Test get_section ---

    def test_get_section_returns_section_when_found(self, temp_dir):
        """Test that get_section returns Section when found."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sections_collection = MagicMock()

        sections_collection.get.return_value = {
            "ids": ["sec_1"],
            "documents": ["This is section text."],
            "metadatas": [{
                "document_key": "doc1",
                "title": "Section 1",
                "level": "1",
                "start_page": "1",
                "end_page": "2"
            }]
        }

        mock_client.get_collection.return_value = sections_collection

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_section("sec_1")
            
            assert result is not None
            assert result.id == "sec_1"
            assert result.title == "Section 1"

    def test_get_section_returns_none_when_not_found(self, temp_dir):
        """Test that get_section returns None when section doesn't exist."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sections_collection = MagicMock()

        # Empty result
        sections_collection.get.return_value = {"ids": []}

        mock_client.get_collection.return_value = sections_collection

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_section("nonexistent")
            
            assert result is None

    # --- Test get_all_sections ---

    def test_get_all_sections_for_document(self, temp_dir):
        """Test retrieving all sections for a document."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sections_collection = MagicMock()

        sections_collection.get.return_value = {
            "ids": ["sec_1", "sec_2"],
            "documents": ["Text 1", "Text 2"],
            "metadatas": [
                {"document_key": "doc1", "title": "Section 1", "level": "1", "start_page": "1", "end_page": "1"},
                {"document_key": "doc1", "title": "Section 2", "level": "1", "start_page": "2", "end_page": "2"}
            ]
        }

        mock_client.get_collection.return_value = sections_collection

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_all_sections("doc1")
            
            assert len(result) == 2
            assert result[0].id == "sec_1"
            assert result[1].id == "sec_2"

    def test_get_all_sections_returns_empty_when_none(self, temp_dir):
        """Test that get_all_sections returns empty list when no sections exist."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sections_collection = MagicMock()

        # Empty result
        sections_collection.get.return_value = {"ids": []}

        mock_client.get_collection.return_value = sections_collection

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_all_sections("doc1")
            
            assert result == []

    # --- Test search_sections ---

    def test_search_sections_returns_results(self, temp_dir):
        """Test that search_sections returns matching sections."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sections_collection = MagicMock()

        import numpy as np
        mock_embedding = np.array([0.1] * 384)

        sections_collection.query.return_value = {
            "ids": [["sec_1", "sec_2"]],
            "embeddings": [[mock_embedding, mock_embedding]]
        }

        mock_client.get_collection.return_value = sections_collection

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            query_emb = [0.1] * 384
            ids, embeddings = store.search_sections(query_emb, top_k=10)
            
            assert len(ids) == 2

    def test_search_sections_returns_empty_on_no_results(self, temp_dir):
        """Test that search_sections handles empty results."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sections_collection = MagicMock()

        # Empty result
        sections_collection.query.return_value = {"ids": [[]]}

        mock_client.get_collection.return_value = sections_collection

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            query_emb = [0.1] * 384
            ids, embeddings = store.search_sections(query_emb, top_k=10)
            
            assert ids == []
            assert embeddings == []

    # --- Test add_sentence_windows ---

    def test_add_sentence_windows_with_valid_data(self, temp_dir):
        """Test adding sentence windows with embeddings."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sentences_collection = MagicMock()

        def get_or_create(name):
            return sentences_collection if name == "sentences" else MagicMock()

        mock_client.get_or_create_collection.side_effect = get_or_create
        type(sentences_collection).count = PropertyMock(return_value=0)

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            windows = [
                SentenceWindow(
                    id="win_1",
                    section_id="sec_1",
                    window_index=0,
                    text="This is a sentence.",
                    sentences=["This is a sentence."],
                    is_embedded=False
                )
            ]
            embeddings = [[0.1] * 384]
            
            store.add_sentence_windows(windows, embeddings, "doc1")
            
            sentences_collection.upsert.assert_called_once()

    def test_add_sentence_windows_with_empty_data(self, temp_dir):
        """Test that add_sentence_windows handles empty lists gracefully."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            # Should not raise
            store.add_sentence_windows([], [], "doc1")
            store.add_sentence_windows(None, None, "doc1")

    # --- Test get_sentence_windows ---

    def test_get_sentence_windows_for_section(self, temp_dir):
        """Test retrieving sentence windows for a section."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sentences_collection = MagicMock()
        
        sentences_collection.get.return_value = {
            "ids": ["win_1", "win_2"],
            "documents": ["Sentence 1.", "Sentence 2."],
            "metadatas": [
                {"document_key": "doc1", "section_id": "sec_1", "window_index": "0"},
                {"document_key": "doc1", "section_id": "sec_1", "window_index": "1"}
            ]
        }
        
        mock_client.get_collection.return_value = sentences_collection
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_sentence_windows("sec_1")
            
            assert len(result) == 2
            assert result[0].section_id == "sec_1"

    def test_get_sentence_windows_returns_empty_when_none(self, temp_dir):
        """Test that get_sentence_windows returns empty list when no windows exist."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sentences_collection = MagicMock()
        
        # Empty result
        sentences_collection.get.return_value = {"ids": []}
        
        mock_client.get_collection.return_value = sentences_collection
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_sentence_windows("sec_1")
            
            assert result == []

    # --- Test search_sentences ---

    def test_search_sentences_within_document(self, temp_dir):
        """Test searching sentence windows within a document."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sentences_collection = MagicMock()
        
        import numpy as np
        mock_embedding = np.array([0.1] * 384)
        
        sentences_collection.query.return_value = {
            "ids": [["win_1"]],
            "embeddings": [[mock_embedding]]
        }
        
        mock_client.get_collection.return_value = sentences_collection
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            query_emb = [0.1] * 384
            ids, embeddings = store.search_sentences(query_emb, "doc1", top_k=5)
            
            # Verify the filter was applied
            call_args = sentences_collection.query.call_args
            assert call_args[1]["where"] == {"document_key": "doc1"}

    def test_search_sentences_returns_empty_on_no_results(self, temp_dir):
        """Test that search_sentences handles empty results."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sentences_collection = MagicMock()
        
        # Empty result
        sentences_collection.query.return_value = {"ids": [[]]}
        
        mock_client.get_collection.return_value = sentences_collection
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            query_emb = [0.1] * 384
            ids, embeddings = store.search_sentences(query_emb, "doc1", top_k=5)
            
            assert ids == []
            assert embeddings == []

    # --- Test delete_document ---

    def test_delete_document_calls_delete_on_collections(self, temp_dir):
        """Test that delete_document removes all vectors for a document."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sections_collection = MagicMock()
        sentences_collection = MagicMock()
        
        def get_or_create(name):
            return sections_collection if name == "sections" else sentences_collection
        
        mock_client.get_or_create_collection.side_effect = get_or_create
        type(sections_collection).count = PropertyMock(return_value=0)
        type(sentences_collection).count = PropertyMock(return_value=0)
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            store.delete_document("doc1")
            
            sections_collection.delete.assert_called_once_with(where={"document_key": "doc1"})
            sentences_collection.delete.assert_called_once_with(where={"document_key": "doc1"})

    # --- Test clear_all ---

    def test_clear_all_resets_collections(self, temp_dir):
        """Test that clear_all deletes and recreates collections."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sections_collection = MagicMock()
        
        def get_or_create(name):
            return sections_collection
        
        mock_client.get_or_create_collection.side_effect = get_or_create
        type(sections_collection).count = PropertyMock(return_value=0)
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            store.clear_all()
            
            mock_client.delete_collection.assert_called()
            # Should recreate collections
            assert mock_client.get_or_create_collection.call_count >= 2

    # --- Test get_document_title ---

    def test_get_document_title_returns_title_when_found(self, temp_dir):
        """Test that get_document_title returns title from section metadata."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sections_collection = MagicMock()
        
        sections_collection.get.return_value = {
            "metadatas": [{"title": "My Document Title"}]
        }
        
        mock_client.get_collection.return_value = sections_collection
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_document_title("doc1")
            
            assert result == "My Document Title"

    def test_get_document_title_returns_none_when_not_found(self, temp_dir):
        """Test that get_document_title returns None when document not found."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sections_collection = MagicMock()
        
        # Empty result
        sections_collection.get.return_value = {"metadatas": []}
        
        mock_client.get_collection.return_value = sections_collection
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_document_title("nonexistent")
            
            assert result is None

    # --- Test get_section_count and get_sentence_count ---

    def test_get_section_count_returns_count(self, temp_dir):
        """Test that get_section_count returns the number of sections."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sections_collection = MagicMock()
        
        type(sections_collection).count = PropertyMock(return_value=42)
        
        def get_or_create(name):
            return sections_collection if name == "sections" else MagicMock()
        
        mock_client.get_or_create_collection.side_effect = get_or_create
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_section_count()
            
            assert result == 42

    def test_get_sentence_count_returns_count(self, temp_dir):
        """Test that get_sentence_count returns the number of sentence windows."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sentences_collection = MagicMock()
        
        type(sentences_collection).count = PropertyMock(return_value=100)
        
        def get_or_create(name):
            return MagicMock() if name == "sections" else sentences_collection
        
        mock_client.get_or_create_collection.side_effect = get_or_create
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_sentence_count()
            
            assert result == 100

    # --- Test persistence directory creation ---

    def test_persist_directory_created_on_init(self, temp_dir):
        """Test that persist directory is created on initialization."""
        new_dir = temp_dir / "new_vectordb"
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(new_dir))
            
            assert new_dir.exists()
            assert new_dir.is_dir()

    # --- Test thread safety ---

    def test_embedded_docs_lock_exists(self, temp_dir):
        """Test that the lock is created for thread safety."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            assert hasattr(store, "_embedded_docs_lock")