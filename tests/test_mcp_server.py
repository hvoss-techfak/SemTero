import sys
import os
import asyncio

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zoterorag.config import Config
from zoterorag.mcp_server import MCPZoteroServer
from zoterorag.models import SearchResult


def test_mcp_search_documents_passes_citation_return_mode_and_filters_require_cited_bibtex(monkeypatch):
    config = Config()
    server = MCPZoteroServer(config)

    # Avoid any real Zotero API calls
    monkeypatch.setattr(server, "_get_metadata_for_key", lambda key: {"title": "", "bibtex": ""})

    # Stub vector_store title lookup
    server.search_engine.vector_store.get_document_title = lambda key: None

    # Two results: one with cited bibtex, one without
    r_with = SearchResult(
        text="hello",
        document_title="",
        section_title="",
        zotero_key="docA",
        relevance_score=0.9,
        rerank_score=0.9,
        cited_bibtex=["@article{a, title={A}}"],
    )
    r_without = SearchResult(
        text="world",
        document_title="",
        section_title="",
        zotero_key="docB",
        relevance_score=0.9,
        rerank_score=0.9,
        cited_bibtex=[],
    )

    calls = {}

    def fake_search_best_sentences(**kwargs):
        calls.update(kwargs)
        return [r_with, r_without]

    server.search_engine.search_best_sentences = fake_search_best_sentences

    out = asyncio.run(
        server.search_documents(
            query="q",
            citation_return_mode="bibtex",
            require_cited_bibtex=True,
            min_relevance=0.0,
        )
    )

    # Ensure we passed the mode through
    assert calls.get("citation_return_mode") == "bibtex"

    # Ensure we filtered out the one without cited bibtex
    assert len(out) == 1
    assert out[0]["zotero_key"] == "docA"
