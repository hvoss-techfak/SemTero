"""Extended tests for PDFProcessor - extraction and sectioning methods."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from zoterorag.pdf_processor import PDFProcessor
from zoterorag.models import Section, SentenceWindow


class TestExtractMarkdown:
    """Test suite for extract_markdown method."""
    
    @pytest.fixture
    def processor(self):
        return PDFProcessor()
    
    def test_extract_markdown_nonexistent_file(self, processor):
        """Should return empty string for nonexistent file."""
        result = processor.extract_markdown("/nonexistent/file.pdf")
        assert result == ""
    
    @patch('zoterorag.pdf_processor.pymupdf4llm.to_markdown')
    def test_extract_markdown_with_layout(self, mock_to_markdown, processor):
        """Test extraction with layout preservation."""
        mock_to_markdown.return_value = "Mocked markdown content"
        
        with patch('pathlib.Path.exists', return_value=True):
            result = processor.extract_markdown("/test.pdf")
        
        assert result == "Mocked markdown content"
        mock_to_markdown.assert_called_once()
    
    @patch('zoterorag.pdf_processor.pymupdf4llm.to_markdown')
    def test_extract_markdown_without_layout(self, mock_to_markdown, processor):
        """Test extraction without layout preservation."""
        processor_no_layout = PDFProcessor(use_layout=False)
        
        with patch('pathlib.Path.exists', return_value=True):
            result = processor_no_layout.extract_markdown("/test.pdf")
        
        # Called without page_chunks
        call_args = mock_to_markdown.call_args[0][0]
        assert "/test.pdf" in call_args
    
    @patch('zoterorag.pdf_processor.pymupdf4llm.to_markdown')
    def test_extract_markdown_returns_list(self, mock_to_markdown, processor):
        """Test handling of list return from pymupdf4llm."""
        mock_to_markdown.return_value = [
            {"text": "Page 1 content"},
            {"text": "Page 2 content"}
        ]
        
        with patch('pathlib.Path.exists', return_value=True):
            result = processor.extract_markdown("/test.pdf")
        
        assert result == "Page 1 content\nPage 2 content"
    
    @patch('zoterorag.pdf_processor.pymupdf4llm.to_markdown')
    def test_extract_markdown_exception_handling(self, mock_to_markdown, processor):
        """Test exception handling during extraction."""
        mock_to_markdown.side_effect = Exception("PDF error")
        
        with patch('pathlib.Path.exists', return_value=True):
            result = processor.extract_markdown("/test.pdf")
        
        assert result == ""


class TestExtractQuarterSections:
    """Test suite for extract_quarter_sections method."""
    
    @pytest.fixture
    def processor(self):
        return PDFProcessor(page_splits=4)
    
    def test_extract_quarter_sections_nonexistent_file(self, processor):
        """Should return empty list for nonexistent file."""
        result = processor.extract_quarter_sections("/nonexistent/file.pdf")
        assert result == []
    
    @patch('zoterorag.pdf_processor.pymupdf4llm.to_markdown')
    def test_extract_quarter_sections_single_page(self, mock_to_markdown, processor):
        """Test single page PDF handling."""
        mock_to_markdown.return_value = "Single page text content"
        
        with patch('pathlib.Path.exists', return_value=True):
            result = processor.extract_quarter_sections("/test.pdf")
        
        assert len(result) == 1
        assert result[0].document_id == "test"
    
    @patch('zoterorag.pdf_processor.pymupdf4llm.to_markdown')
    def test_extract_quarter_sections_multiple_pages(self, mock_to_markdown, processor):
        """Test multiple page PDF handling."""
        mock_to_markdown.return_value = [
            {"page": 1, "text": "Page 1 line 1\nPage 1 line 2\nPage 1 line 3"},
            {"page": 2, "text": "Page 2 content here"}
        ]
        
        with patch('pathlib.Path.exists', return_value=True):
            result = processor.extract_quarter_sections("/test.pdf")
        
        # With page_splits=4 and only ~3 lines on first page + some on second,
        # should create sections
        assert len(result) > 0
    
    @patch('zoterorag.pdf_processor.pymupdf4llm.to_markdown')
    def test_extract_quarter_sections_empty_pages(self, mock_to_markdown, processor):
        """Test handling of empty page content."""
        mock_to_markdown.return_value = [
            {"page": 1, "text": ""},
            {"page": 2, "text": ""}
        ]
        
        with patch('pathlib.Path.exists', return_value=True):
            result = processor.extract_quarter_sections("/test.pdf")
        
        # Empty pages should produce no sections
        assert len(result) == 0
    
    @patch('zoterorag.pdf_processor.pymupdf4llm.to_markdown')
    def test_extract_quarter_sections_exception(self, mock_to_markdown, processor):
        """Test exception handling."""
        mock_to_markdown.side_effect = Exception("PDF error")
        
        with patch('pathlib.Path.exists', return_value=True):
            result = processor.extract_quarter_sections("/test.pdf")
        
        assert result == []


class TestCreateSentenceWindows:
    """Test suite for create_sentence_windows method."""
    
    @pytest.fixture
    def processor(self):
        return PDFProcessor()
    
    @pytest.fixture
    def sample_section(self):
        return Section(
            id="doc1_sec_0",
            document_id="doc1",
            title="Introduction",
            level=1,
            start_page=1,
            end_page=1,
            text="This is the first sentence. This is the second sentence! Is this the third? Yes, it is.",
        )
    
    def test_create_sentence_windows_basic(self, processor, sample_section):
        """Test basic window creation."""
        result = processor.create_sentence_windows(sample_section, window_size=3, overlap=2)
        
        assert len(result) > 0
        # Check first window has expected structure
        assert hasattr(result[0], 'id')
        assert hasattr(result[0], 'section_id')
        assert hasattr(result[0], 'sentences')
    
    def test_create_sentence_windows_no_overlap(self, processor, sample_section):
        """Test windows with no overlap."""
        result = processor.create_sentence_windows(sample_section, window_size=2, overlap=0)
        
        # Windows should not share sentences
        if len(result) >= 2:
            first_window_texts = set(result[0].sentences)
            second_window_texts = set(result[1].sentences)
            assert not first_window_texts.intersection(second_window_texts)
    
    def test_create_sentence_windows_max_overlap(self, processor, sample_section):
        """Test windows with maximum overlap (slide by 1)."""
        result = processor.create_sentence_windows(sample_section, window_size=3, overlap=2)
        
        # With overlap of 2 and window size of 3, step is 1
        if len(result) >= 2:
            first_window_sents = set(result[0].sentences)
            second_window_sents = set(result[1].sentences)
            # Should share 2 sentences (maximum overlap)
            assert len(first_window_sents.intersection(second_window_sents)) == 2
    
    def test_create_sentence_windows_few_sentences(self, processor):
        """Test handling of section with fewer sentences than window_size."""
        short_section = Section(
            id="doc1_sec_0",
            document_id="doc1",
            title="Short",
            level=1,
            start_page=1,
            end_page=1,
            text="Just one sentence.",
        )
        
        result = processor.create_sentence_windows(short_section, window_size=3, overlap=2)
        
        # Should still create at least one window
        assert len(result) >= 1
    
    def test_create_sentence_windows_empty_text(self, processor):
        """Test handling of empty section text."""
        empty_section = Section(
            id="doc1_sec_0",
            document_id="doc1",
            title="Empty",
            level=1,
            start_page=1,
            end_page=1,
            text="",
        )
        
        result = processor.create_sentence_windows(empty_section, window_size=3, overlap=2)
        
        # Should still create a single window with the empty content
        assert len(result) >= 1
    
    def test_create_sentence_windows_ids_unique(self, processor, sample_section):
        """Test that generated IDs are unique."""
        result = processor.create_sentence_windows(sample_section, window_size=3, overlap=2)
        
        ids = [w.id for w in result]
        assert len(ids) == len(set(ids))
    
    def test_create_sentence_windows_preserves_order(self, processor, sample_section):
        """Test that windows are created in order."""
        result = processor.create_sentence_windows(sample_section, window_size=2, overlap=1)
        
        # Check window_index increases sequentially
        for i, window in enumerate(result):
            assert window.window_index == i


class TestHeadingMethods:
    """Test suite for heading-related methods."""
    
    @pytest.fixture
    def processor(self):
        return PDFProcessor()
    
    def test_remove_headings_basic(self, processor):
        """Test basic heading removal."""
        text = """# Title

Some content

## Section 1

More content"""
        
        result = processor._remove_headings(text)
        
        assert "#" not in result.split("\n")[0]  # First line should be just "Title"
        assert "Some content" in result
    
    def test_remove_headings_multiple_levels(self, processor):
        """Test removal of all heading levels."""
        text = """# H1
## H2
### H3
#### H4
##### H5
###### H6
Content"""
        
        result = processor._remove_headings(text)
        
        assert "#" not in result[:20]
    
    def test_find_headings_in_text_basic(self, processor):
        """Test heading finding and section creation."""
        text = """# Introduction

First paragraph here.

## Background

Background information.
"""
        
        sections = processor._find_headings_in_text(
            text=text,
            prev_text="",
            prev_title="",
            prev_level=0,
            start_page=1,
            current_page=2,
            section_index=0,
            doc_id="doc1"
        )
        
        assert len(sections) > 0
        # Check that sections have correct structure
        for sec in sections:
            assert hasattr(sec, 'id')
            assert hasattr(sec, 'document_id')


class TestHeadingMarkerStripping:
    """Test suite for _strip_heading_markers method."""
    
    @pytest.fixture
    def processor(self):
        return PDFProcessor()
    
    def test_strip_all_levels(self, processor):
        """Test stripping all heading levels."""
        text = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        
        result = processor._strip_heading_markers(text)
        
        lines = result.split("\n")
        assert "H1" in lines[0]
        assert "H6" in lines[5]


class TestBlockquoteCleanup:
    """Test suite for _cleanup_blockquotes method."""
    
    @pytest.fixture
    def processor(self):
        return PDFProcessor()
    
    def test_cleanup_simple_blockquote(self, processor):
        text = "> Simple quote"
        result = processor._cleanup_blockquotes(text)
        assert ">" not in result
    
    def test_cleanup_nested_blockquote(self, processor):
        text = """> Level 1
>> Level 2
>>> Level 3"""
        
        result = processor._cleanup_blockquotes(text)
        
        # No > markers should remain
        for line in result.split("\n"):
            assert not line.startswith(">")
    
    def test_cleanup_multiple_quotes(self, processor):
        text = """> First quote
>
> Second quote"""
        
        result = processor._cleanup_blockquotes(text)
        
        assert "First quote" in result
        assert "Second quote" in result


class TestHorizontalRules:
    """Test suite for _remove_horizontal_rules method."""
    
    @pytest.fixture
    def processor(self):
        return PDFProcessor()
    
    def test_remove_dashes(self, processor):
        text = "Text\n---\nMore"
        result = processor._remove_horizontal_rules(text)
        
        assert "---" not in result
    
    def test_remove_stars(self, processor):
        text = "Text\n***\nMore"
        result = processor._remove_horizontal_rules(text)
        
        assert "***" not in result
    
    def test_remove_underscores(self, processor):
        text = "Text\n___\nMore"
        result = processor._remove_horizontal_rules(text)
        
        assert "___" not in result


class TestWhitespaceNormalization:
    """Test suite for _normalize_whitespace method."""
    
    @pytest.fixture
    def processor(self):
        return PDFProcessor()
    
    def test_normalize_multiple_newlines(self, processor):
        text = "Line 1\n\n\n\nLine 2"
        result = processor._normalize_whitespace(text)
        
        assert "\n\n\n" not in result
    
    def test_normalize_trailing_spaces(self, processor):
        text = "Text with trailing   \nMore text"
        result = processor._normalize_whitespace(text)
        
        for line in result.split("\n"):
            assert not line.endswith(" ")
    
    def test_preserve_single_newlines(self, processor):
        text = "Line 1\nLine 2"
        result = processor._normalize_whitespace(text)
        
        # Single newlines should be preserved
        assert "\n" in result


class TestProcessDocumentFunction:
    """Test suite for process_document function."""
    
    @patch('zoterorag.pdf_processor.PDFProcessor')
    def test_process_document_uses_default_processor(self, mock_processor_class):
        """Test that process_document creates PDFProcessor with defaults."""
        from zoterorag.pdf_processor import process_document
        from zoterorag.models import Document
        
        # Setup mock
        mock_processor = Mock()
        mock_processor.extract_quarter_sections.return_value = []
        mock_processor_class.return_value = mock_processor
        
        doc = Document(
            zotero_key="test123",
            title="Test Doc",
            authors=["Author"],
            pdf_path=Path("/test/doc.pdf")
        )
        
        result = process_document(doc)
        
        # Should call extract_quarter_sections with string path
        mock_processor.extract_quarter_sections.assert_called_once()