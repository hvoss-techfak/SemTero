#!/usr/bin/env python3
"""MCP Client for ZoteroRAG.

This client connects to the running ZoteroRAG MCP server and calls its tools.

Examples:
  - Query best matching sentences:
      uv run mcp_client.py --sentence "my query" --top-sections 5 --top-sentences 10

Notes:
- This client expects the MCP server to be runnable as a subprocess (stdio transport).
- It uses the MCP tool `search_sentences`, which returns sentence windows enriched with
  Title/Authors/BibTeX/Published-at (date) and a file URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of MCP result objects into JSON-serializable data."""
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]

    # Pydantic models / MCP result types often implement model_dump()
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return _to_jsonable(dump())
        except Exception:
            pass

    # Fallback: try __dict__
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return _to_jsonable(d)

    return str(obj)


def _format_json(data: Any) -> str:
    return json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")


def _shorten(text: str, max_len: int = 260) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


@dataclass
class SentenceHit:
    text: str
    document_title: str
    section_title: str
    zotero_key: str
    score: float
    authors: list[str]
    date: str
    bibtex: str
    file_path: str

    @classmethod
    def from_dict(cls, d: dict) -> "SentenceHit":
        return cls(
            text=d.get("text", ""),
            document_title=d.get("document_title", ""),
            section_title=d.get("section_title", ""),
            zotero_key=d.get("zotero_key", ""),
            score=float(d.get("rerank_score", d.get("relevance_score", 0.0)) or 0.0),
            authors=d.get("authors") or [],
            date=d.get("date", ""),
            bibtex=d.get("bibtex", ""),
            file_path=d.get("file_path", ""),
        )


def _get_mapping(maybe: Any) -> dict:
    if isinstance(maybe, dict):
        return maybe
    return {}


async def _call_search_sentences(args) -> list[dict]:
    """Spawn the server as a stdio subprocess and call the MCP tool."""

    # Local import so this script can still show help even if deps aren't installed.
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    repo_root = Path(__file__).parent

    # Start the server via stdio transport.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(repo_root / "main.py")],
        cwd=str(repo_root),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_args = {
                "query": args.sentence,
                "document_key": args.document_key,
                "top_sections": args.top_sections,
                "top_sentences": args.top_sentences,
                "ensure_sentence_embeddings": not args.no_persist,
            }

            result = await session.call_tool("search_sentences", tool_args)

            # Prefer FastMCP structuredContent (json_response=True)
            structured = getattr(result, "structuredContent", None)
            if isinstance(structured, dict):
                maybe = structured.get("result")
                if isinstance(maybe, list):
                    return [r for r in maybe if isinstance(r, dict)]

            # Fall back: if tool returned raw JSON text content, parse it
            content = getattr(result, "content", None)
            if isinstance(content, list) and content:
                first = content[0]
                text = getattr(first, "text", None)
                if isinstance(text, str) and text.strip():
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            return [r for r in parsed if isinstance(r, dict)]
                        if isinstance(parsed, dict) and "result" in parsed and isinstance(parsed["result"], list):
                            return [r for r in parsed["result"] if isinstance(r, dict)]
                        if isinstance(parsed, dict):
                            return [parsed]
                    except Exception:
                        pass

            # Last resort: normalize whatever we got into plain Python data.
            normalized = _to_jsonable(result)
            if isinstance(normalized, dict):
                sc = normalized.get("structuredContent")
                if isinstance(sc, dict):
                    maybe = sc.get("result")
                    if isinstance(maybe, list):
                        return [r for r in maybe if isinstance(r, dict)]

            return [{"error": "Unexpected MCP response shape", "raw": str(result)}]


def _print_sentence_hits(raw: list[dict], show_bibtex: bool = False) -> None:
    hits = [SentenceHit.from_dict(r) for r in (raw or []) if isinstance(r, dict)]

    if not hits:
        print("No sentence matches returned.")
        return

    print(f"Returned {len(hits)} sentence match(es).\n")

    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h.score:.4f}  key={h.zotero_key}")
        if h.document_title:
            print(f"    Title: {h.document_title}")
        if h.authors:
            print(f"    Authors: {', '.join(h.authors)}")
        if h.date:
            print(f"    Published at: {h.date}")
        if h.section_title:
            print(f"    Section: {h.section_title}")
        if h.file_path:
            print(f"    PDF: {h.file_path}")

        print(f"    Sentence: {_shorten(h.text)}")

        if show_bibtex and h.bibtex:
            print("    BibTeX:")
            for line in h.bibtex.strip().splitlines():
                print(f"      {line}")
            print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="ZoteroRAG MCP client")

    parser.add_argument(
        "--sentence",
        "-s",
        required=True,
        help="Query sentence/text to match against your library",
    )
    parser.add_argument(
        "--document-key",
        default=None,
        help="Optional Zotero document key to restrict search",
    )
    parser.add_argument(
        "--top-sections",
        type=int,
        default=5,
        help="How many top sections to consider (default: 5)",
    )
    parser.add_argument(
        "--top-sentences",
        type=int,
        default=10,
        help="How many top sentences to return (default: 10)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't persistently embed sentences (only use existing sentence embeddings)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON response",
    )
    parser.add_argument(
        "--bibtex",
        action="store_true",
        help="Print BibTeX entries",
    )

    args = parser.parse_args()

    _print_section("ZoteroRAG: sentence search")
    print(f"Query: {args.sentence}")
    print(f"Top sections: {args.top_sections} | Top sentences: {args.top_sentences}")
    if args.document_key:
        print(f"Document key filter: {args.document_key}")
    if args.no_persist:
        print("Sentence persistence: OFF")

    raw = await _call_search_sentences(args)

    if args.json:
        print(_format_json(raw))
        return

    _print_sentence_hits(raw, show_bibtex=args.bibtex)


if __name__ == "__main__":
    asyncio.run(main())