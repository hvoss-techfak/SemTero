"""MCP server for ZoteroRAG using FastMCP.

This module implements an MCP (Model Context Protocol) server that exposes
Zotero RAG functionality as tools that can be called by LLM clients.
Uses the FastMCP framework for simpler implementation.
"""

import logging
from typing import Any, Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Context

from .config import Config
from .zotero_client import ZoteroClient
from .embedding_manager import EmbeddingManager
from .search_engine import SearchEngine


logger = logging.getLogger(__name__)

# Create FastMCP instance with proper capabilities
mcp = FastMCP(
    "zoterorag",
    instructions="MCP server for RAG search in Zotero library using Ollama",
    json_response=True,
)


class MCPZoteroServer:
    """MCP server exposing Zotero RAG tools.
    
    This class wraps the FastMCP-decorated functions and provides
    business logic integration.
    """

    def __init__(self, config: Config):
        self.config = config
        self.zotero_client = ZoteroClient(api_url=config.ZOTERO_API_URL)
        # Pass zotero_client to EmbeddingManager for on-demand PDF fetching
        self.embedding_manager = EmbeddingManager(config, zotero_client=self.zotero_client)
        self.search_engine = SearchEngine(config)

        # Cache for document metadata to avoid repeated API calls
        # NOTE: An embedding/document key can refer to a PDF attachment item.
        # We keep:
        #   - _metadata_cache: resolved bibliographic metadata (typically from parent item)
        #   - _parent_key_cache: attachment key -> parent key resolution
        self._metadata_cache: dict[str, dict] = {}
        self._parent_key_cache: dict[str, str] = {}
        self._pdf_stem_to_item_key_cache: dict[str, str] = {}

        # Force initialization of the ThreadPoolExecutor by accessing it once
        # This ensures it's ready when we need it for background embedding
        _ = self.embedding_manager.executor
        logger.info("[MCPZoteroServer] EmbeddingManager executor initialized")

    def _resolve_parent_key(self, item_key: str) -> str:
        """Resolve an item key to its bibliographic (parent) key when applicable.

        Many embeddings are keyed by attachment item keys (PDFs). Bibliographic
        data (title/authors/date/BibTeX) usually lives on the parent item.

        Returns the parent key if known, otherwise the original key.
        """
        if not item_key:
            return item_key

        cached = self._parent_key_cache.get(item_key)
        if cached:
            return cached

        try:
            item = self.zotero_client.get_item(item_key)
            parent = (item.get("data") or {}).get("parentItem")
            resolved = parent or item_key
            self._parent_key_cache[item_key] = resolved
            return resolved
        except Exception:
            # If we can't resolve, fall back to original key.
            self._parent_key_cache[item_key] = item_key
            return item_key

    def _resolve_zotero_item_key(self, maybe_doc_key: str) -> str:
        """Resolve a vector-store document key to an actual Zotero item key.

        In this project, embeddings are often stored under the PDF filename stem
        (Path.stem) as the vector-store document_key. Zotero metadata, however,
        is accessible only via the real Zotero item key.

        Strategy:
        - If the key exists in Zotero as an item, keep it.
        - Otherwise, try to find an attachment item whose filename stem matches.
        """
        if not maybe_doc_key:
            return maybe_doc_key

        cached = self._pdf_stem_to_item_key_cache.get(maybe_doc_key)
        if cached:
            return cached

        # Fast path: does this key exist as a Zotero item key?
        try:
            item = self.zotero_client.get_item(maybe_doc_key)
            if item and item.get("key"):
                self._pdf_stem_to_item_key_cache[maybe_doc_key] = maybe_doc_key
                return maybe_doc_key
        except Exception:
            pass

        # Slow path: scan for a matching attachment by filename stem.
        try:
            url = f"{self.config.ZOTERO_API_URL.rstrip('/')}/api/users/0/items"
            params = {"itemType": "attachment", "limit": 100}

            while True:
                resp = self.zotero_client.session.get(url, params=params)
                if resp.status_code != 200:
                    break
                items = resp.json() or []
                if not items:
                    break

                for it in items:
                    data = (it or {}).get("data") or {}
                    if data.get("itemType") != "attachment":
                        continue
                    fname = data.get("filename") or ""
                    if not fname:
                        continue
                    stem = Path(fname).stem
                    if stem == maybe_doc_key:
                        key = it.get("key") or data.get("key")
                        if key:
                            self._pdf_stem_to_item_key_cache[maybe_doc_key] = key
                            return key

                # Pagination
                params["start"] = int(params.get("start", 0)) + int(params.get("limit", 100))

        except Exception:
            pass

        # Give up
        self._pdf_stem_to_item_key_cache[maybe_doc_key] = maybe_doc_key
        return maybe_doc_key

    def _get_metadata_for_key(self, item_key: str) -> dict:
        """Get cached or fresh metadata for an item.

        If the key refers to a PDF attachment, ZoteroClient.get_item_metadata()
        traverses parents and returns bibliographic fields from the parent.
        We additionally normalize caching so that multiple attachment keys of the
        same parent reuse the same cached metadata.
        """
        if not item_key:
            return {
                "bibtex": "",
                "file_path": "",
                "title": "",
                "authors": [],
                "date": "",
                "item_type": "",
            }

        # Resolve vector-store doc keys (PDF stems) to real Zotero item keys.
        item_key = self._resolve_zotero_item_key(item_key)

        # Prefer caching by resolved bibliographic key.
        resolved_key = self._resolve_parent_key(item_key)

        if resolved_key in self._metadata_cache:
            meta = dict(self._metadata_cache[resolved_key])
            # Ensure file_path points to the requested (likely attachment) key.
            if not meta.get("file_path"):
                meta["file_path"] = f"{self.config.ZOTERO_API_URL.rstrip('/')}/api/users/0/items/{item_key}/file"
            return meta

        # Try to get from Zotero (handles attachment->parent traversal internally)
        metadata = self.zotero_client.get_item_metadata(item_key)

        if metadata:
            # If metadata came back without file_path, prefer attachment file URL.
            if not metadata.get("file_path"):
                metadata["file_path"] = f"{self.config.ZOTERO_API_URL.rstrip('/')}/api/users/0/items/{item_key}/file"

            self._metadata_cache[resolved_key] = metadata
            # Also allow direct lookup by original key.
            self._metadata_cache[item_key] = metadata
            return metadata

        return {
            "bibtex": "",
            "file_path": "",
            "title": "",
            "authors": [],
            "date": "",
            "item_type": ""
        }

    def shutdown(self):
        """Gracefully shutdown background resources."""
        try:
            if getattr(self, "embedding_manager", None):
                self.embedding_manager.shutdown()
        except Exception as e:
            logger.debug(f"[MCPZoteroServer] shutdown: embedding_manager error: {e}")


# Global server instance for business logic
_server_instance: Optional[MCPZoteroServer] = None


def set_server_instance(server: MCPZoteroServer):
    """Set the global server instance for use in tool functions."""
    global _server_instance
    _server_instance = server


def get_server() -> MCPZoteroServer:
    """Get the current server instance."""
    return _server_instance


# =============================================================================
# FastMCP Tool Implementations
# =============================================================================

@mcp.tool()
async def search_documents(
    query: str,
    document_key: Optional[str] = None,
    top_sections: int = 5,
    top_sentences_per_section: int = 3,
) -> list[dict]:
    """Search across embedded documents using two-stage RAG.
    
    Returns enriched results including BibTeX and file metadata.
    
    Args:
        query: The search query string
        document_key: Optional filter to a specific Zotero document key
        top_sections: Number of top sections to return (default: 5)
        top_sentences_per_section: Number of sentences per section (default: 3)
    """
    server = get_server()
    if not server:
        return [{"error": "Server not initialized"}]
    
    results = server.search_engine.search(
        query=query,
        document_key=document_key,
        top_sections=top_sections,
        top_sentences_per_section=top_sentences_per_section
    )

    # First pass: collect all unique keys and fetch metadata for each once
    key_to_metadata: dict[str, dict] = {}

    for r in results:
        zotero_key = r.zotero_key

        if zotero_key and zotero_key not in key_to_metadata:
            # Fetch Zotero metadata (only once per unique document)
            meta = server._get_metadata_for_key(zotero_key)

            # If no Zotero title, fall back to vector store title
            if not meta.get("title"):
                vs_title = server.search_engine.vector_store.get_document_title(zotero_key)
                if vs_title:
                    meta["title"] = vs_title

            key_to_metadata[zotero_key] = meta

    # Second pass: apply metadata to all results
    enriched_results = []

    for r in results:
        result_dict = r.to_dict()

        zotero_key = r.zotero_key

        if zotero_key and zotero_key in key_to_metadata:
            meta = key_to_metadata[zotero_key]

            # Apply all enriched data to this result
            result_dict["bibtex"] = meta.get("bibtex", "")
            result_dict["file_path"] = meta.get("file_path", "")
            result_dict["authors"] = meta.get("authors", [])
            result_dict["date"] = meta.get("date", "")
            result_dict["item_type"] = meta.get("item_type", "")

            # Set document_title - prefer Zotero title, fallback to vector store
            doc_title = meta.get("title", "")
            if not doc_title:
                vs_title = server.search_engine.vector_store.get_document_title(zotero_key)
                if vs_title:
                    doc_title = vs_title
            result_dict["document_title"] = doc_title

        # If we still don't have a document title, use section info as fallback
        if not result_dict.get("document_title") and result_dict.get("section_title"):
            result_dict["document_title"] = f"Document containing: {result_dict['section_title'][:50]}"

        enriched_results.append(result_dict)

    return enriched_results


@mcp.tool()
async def get_library_items(limit: int = 25) -> list[dict]:
    """Get items from Zotero library with their metadata.
    
    Args:
        limit: Maximum number of items to return (default: 25)
    """
    server = get_server()
    if not server:
        return [{"error": "Server not initialized"}]
        
    items = server.zotero_client.get_items(limit=limit)
    return [
        {
            "key": item.get("key", ""),
            "title": item.get("data", {}).get("title", ""),
            "authors": [a.get("firstName", "") + " " + a.get("lastName", "")
                       for a in item.get("data", {}).get("creators", [])],
            "date": item.get("data", {}).get("date", ""),
        }
        for item in items
    ]


@mcp.tool()
async def get_documents_with_pdfs() -> list[dict]:
    """Get all documents that have PDFs attached in Zotero."""
    server = get_server()
    if not server:
        return [{"error": "Server not initialized"}]
        
    docs = server.zotero_client.get_documents_with_pdfs()
    return [
        {
            "key": doc.zotero_key,
            "title": doc.title,
            "authors": doc.authors,
            "has_pdf": bool(doc.pdf_path),
        }
        for doc in docs
    ]


@mcp.tool()
async def sync_and_embed(embed_sentences: bool = False, document_key: Optional[str] = None) -> dict:
    """Sync from Zotero and embed new documents.
    
    Uses on-demand PDF fetching - PDFs are fetched temporarily for embedding
    and not saved to disk permanently.
    
    Args:
        embed_sentences: Whether to embed sentences (default: false)
        document_key: Optional specific document key to sync
    """
    server = get_server()
    if not server:
        return {"status": "error", "message": "Server not initialized"}
        
    # Get documents with PDFs
    all_docs_with_pdfs = list(server.zotero_client.get_documents_with_pdfs())

    if not all_docs_with_pdfs:
        return {
            "status": "no_documents",
            "message": "No documents with PDFs found in Zotero"
        }

    # Filter by document key if specified
    docs_to_process = [
        doc for doc in all_docs_with_pdfs
        if document_key is None or doc.zotero_key == document_key
    ]

    if not docs_to_process:
        return {
            "status": "no_match",
            "message": f"No documents matching {document_key}"
        }

    # Get currently embedded docs
    embedded = server.embedding_manager.vector_store.get_embedded_documents()

    # Filter to only new/updated documents
    pending = [
        doc for doc in docs_to_process
        if doc.zotero_key not in embedded or embedded[doc.zotero_key] == 0
    ]

    if not pending:
        return {
            "status": "up_to_date",
            "message": f"All {len(docs_to_process)} documents already embedded"
        }

    # Submit for background embedding using on-demand PDF fetching
    futures = []
    for doc in pending:
        def make_callback(doc_key: str):
            def cb(status):
                logger.info(f"Document {doc_key}: sections={status.embedded_sections}")
            return cb

        try:
            future = server.embedding_manager.embed_document_async_with_client(
                doc,
                server.zotero_client,
                callback=make_callback(doc.zotero_key)
            )
            futures.append(future)
        except Exception as e:
            logger.error(f"Failed to submit {doc.zotero_key} for embedding: {e}")

    return {
        "status": "started",
        "pending_count": len(pending),
        "message": f"Started embedding {len(pending)} documents (temp file mode)"
    }


@mcp.tool()
async def get_embedding_status() -> dict:
    """Get current embedding statistics and status."""
    server = get_server()
    if not server:
        return {"error": "Server not initialized"}
        
    stats = server.search_engine.get_stats()
    embedded_docs = server.embedding_manager.vector_store.get_embedded_documents()
    return {
        **stats,
        "documents": list(embedded_docs.keys())
    }


@mcp.tool()
async def delete_document(document_key: str) -> dict:
    """Delete all embeddings for a specific document.
    
    Args:
        document_key: The Zotero document key to delete
    """
    server = get_server()
    if not server:
        return {"status": "error", "message": "Server not initialized"}
        
    server.embedding_manager.vector_store.delete_document(document_key)

    # Update metadata
    embedded = server.embedding_manager.vector_store.get_embedded_documents()
    if document_key in embedded:
        del embedded[document_key]
        server.embedding_manager.vector_store.save_embedded_documents(embedded)

    return {"status": "deleted", "document_key": document_key}


@mcp.tool()
async def reembed_document(document_key: str) -> dict:
    """Re-embed a specific document.
    
    Deletes existing embeddings and creates new ones using on-demand PDF fetching.
    
    Args:
        document_key: The Zotero document key to re-embed
    """
    server = get_server()
    if not server:
        return {"status": "error", "message": "Server not initialized"}
        
    docs = list(server.zotero_client.get_documents_with_pdfs())
    target_doc = None
    for doc in docs:
        if doc.zotero_key == document_key:
            target_doc = doc
            break

    if not target_doc:
        return {"status": "error", "message": f"Document {document_key} not found"}

    # Delete existing embeddings first
    server.embedding_manager.vector_store.delete_document(document_key)

    embedded = server.embedding_manager.vector_store.get_embedded_documents()
    if document_key in embedded:
        del embedded[document_key]
        server.embedding_manager.vector_store.save_embedded_documents(embedded)

    # Re-embed using on-demand PDF fetching (temp file mode)
    server.embedding_manager.embed_document_async_with_client(
        target_doc,
        server.zotero_client
    )

    return {"status": "reembedding", "document_key": document_key}


@mcp.tool()
async def search_sentences(
    query: str,
    document_key: Optional[str] = None,
    top_sections: int = 5,
    top_sentences: int = 10,
    ensure_sentence_embeddings: bool = True,
) -> list[dict]:
    """Search and return best matching sentence windows.

    This is a sentence-first variant of `search_documents`.

    Pipeline:
    1) Embed the query
    2) Retrieve best matching sections (top_sections)
    3) Ensure sentence windows for these sections are persistently embedded
    4) Search sentence embeddings and return top_sentences

    Returns enriched results including BibTeX, title, authors, and date.
    """
    server = get_server()
    if not server:
        return [{"error": "Server not initialized"}]

    results = server.search_engine.search_best_sentences(
        query=query,
        document_key=document_key,
        top_sections=top_sections,
        top_sentences=top_sentences,
        ensure_sentence_embeddings=ensure_sentence_embeddings,
    )

    # Enrich with Zotero metadata (parent item)
    key_to_metadata: dict[str, dict] = {}
    for r in results:
        zotero_key = r.zotero_key
        if zotero_key and zotero_key not in key_to_metadata:
            meta = server._get_metadata_for_key(zotero_key)
            if not meta.get("title"):
                vs_title = server.search_engine.vector_store.get_document_title(zotero_key)
                if vs_title:
                    meta["title"] = vs_title
            key_to_metadata[zotero_key] = meta

    enriched: list[dict] = []
    for r in results:
        result_dict = r.to_dict()
        zotero_key = r.zotero_key

        if zotero_key and zotero_key in key_to_metadata:
            meta = key_to_metadata[zotero_key]
            result_dict["bibtex"] = meta.get("bibtex", "")
            result_dict["file_path"] = meta.get("file_path", "")
            result_dict["authors"] = meta.get("authors", [])
            result_dict["date"] = meta.get("date", "")
            result_dict["item_type"] = meta.get("item_type", "")

            doc_title = meta.get("title", "")
            if not doc_title:
                vs_title = server.search_engine.vector_store.get_document_title(zotero_key)
                if vs_title:
                    doc_title = vs_title
            result_dict["document_title"] = doc_title

        enriched.append(result_dict)

    return enriched


# =============================================================================
# Main entry points
# =============================================================================

async def run_mcp_server(transport: str = "stdio"):
    """Run the MCP server.
    
    Args:
        transport: Transport to use - "stdio" or "streamable-http"
    """
    # Initialize config and create server instance
    config = Config()
    server_instance = MCPZoteroServer(config)
    set_server_instance(server_instance)
    
    logger.info(f"Starting ZoteroRAG MCP server with {transport} transport...")
    
    if transport == "stdio":
        await mcp.run()
    elif transport == "streamable-http":
        # Run with streamable HTTP transport
        await mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}")


def main():
    """Run the MCP server using stdio transport."""
    import asyncio
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting ZoteroRAG MCP server...")
    
    try:
        asyncio.run(run_mcp_server(transport="stdio"))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()

