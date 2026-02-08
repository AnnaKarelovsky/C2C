"""Browser-like tool for searching and opening documents.

Provides ``search``, ``open``, and ``find`` functions with the same interface
as the GPT-OSS SimpleBrowserTool MCP server, backed by the
BrowseCompPlusSearcher FAISS index.

Usage::

    from rosetta.workflow.browse_searcher import configure_search
    from rosetta.workflow.gpt_tool import search, open, find

    configure_search(index_path="...", sglang_url="...")
    tools = [FunctionTool(search), FunctionTool(open), FunctionTool(find)]
"""

import textwrap
from typing import Any, Dict, List, Union
from urllib.parse import quote

from rosetta.workflow.browse_searcher import _get_searcher


# ---------------------------------------------------------------------------
# Page / state management (mirrors GPT-OSS SimpleBrowserState)
# ---------------------------------------------------------------------------

FIND_PAGE_LINK_FORMAT = "# 【{idx}†{title}】"
DEFAULT_VIEW_TOKENS = 1024
DEFAULT_LINE_WIDTH = 80


class _PageContents:
    """A single browsable page."""

    __slots__ = ("url", "title", "text", "urls", "snippets")

    def __init__(
        self,
        url: str,
        title: str,
        text: str,
        urls: Dict[str, str] | None = None,
        snippets: Dict[str, Any] | None = None,
    ):
        self.url = url
        self.title = title
        self.text = text
        self.urls: Dict[str, str] = urls or {}
        self.snippets: Dict[str, Any] | None = snippets


class _BrowserState:
    """Page-stack state identical to GPT-OSS ``SimpleBrowserState``."""

    def __init__(self) -> None:
        self.pages: Dict[str, _PageContents] = {}
        self.page_stack: List[str] = []

    @property
    def current_cursor(self) -> int:
        return len(self.page_stack) - 1

    def add_page(self, page: _PageContents) -> None:
        self.pages[page.url] = page
        self.page_stack.append(page.url)

    def get_page(self, cursor: int = -1) -> _PageContents:
        if self.current_cursor < 0:
            raise ValueError("No pages to access!")
        if cursor == -1 or cursor == self.current_cursor:
            return self.pages[self.page_stack[-1]]
        try:
            page_url = self.page_stack[cursor]
        except IndexError:
            raise ValueError(
                f"Cursor `{cursor}` is out of range. "
                f"Available cursor indices: [0 - {self.current_cursor}]."
            )
        return self.pages[page_url]

    def reset(self) -> None:
        self.pages.clear()
        self.page_stack.clear()


# Module-level state (one per process, reset between questions)
_state = _BrowserState()


def reset_browser() -> None:
    """Reset the browser state, clearing all pages and the page stack."""
    _state.reset()


# ---------------------------------------------------------------------------
# Display helpers (mirrors GPT-OSS rendering)
# ---------------------------------------------------------------------------


def _wrap_lines(text: str, width: int = DEFAULT_LINE_WIDTH) -> List[str]:
    lines = text.split("\n")
    wrapped: List[str] = []
    for line in lines:
        if line:
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
            )
        else:
            wrapped.append("")
    return wrapped


def _strip_links(text: str) -> str:
    """Remove 【idx†...】 link formatting from text for find matching."""
    import re

    partial_initial = re.compile(r"^[^【】]*】")
    partial_final = re.compile(r"【\d*(?:†(?P<content>[^†】]*)(?:†[^†】]*)?)?$")
    link_pattern = re.compile(r"【\d+†(?P<content>[^†】]+)(?:†[^†】]+)?】")

    text = re.sub(partial_initial, "", text)
    text = re.sub(partial_final, lambda mo: mo.group("content") or "", text)
    text = re.sub(link_pattern, lambda mo: mo.group("content"), text)
    return text


def _show_page(
    page: _PageContents,
    cursor: int,
    loc: int = 0,
    num_lines: int = -1,
) -> str:
    """Render a page with line numbers and scrollbar, same as GPT-OSS."""
    lines = _wrap_lines(page.text)
    total_lines = len(lines)

    if total_lines == 0:
        return f"[{cursor}] {page.title}\n(empty page)"

    if loc >= total_lines:
        raise ValueError(
            f"Invalid location parameter: `{loc}`. "
            f"Cannot exceed page maximum of {total_lines - 1}."
        )

    # Compute end location
    if num_lines <= 0:
        # Approximate token budget → character budget (≈4 chars/token)
        max_chars = DEFAULT_VIEW_TOKENS * 4
        char_count = 0
        end_loc = loc
        for i in range(loc, total_lines):
            char_count += len(lines[i]) + 1
            end_loc = i + 1
            if char_count >= max_chars:
                break
    else:
        end_loc = min(loc + num_lines, total_lines)

    body = "\n".join(
        f"L{i + loc}: {line}" for i, line in enumerate(lines[loc:end_loc])
    )

    header = page.title
    if page.url:
        header += f" ({page.url})"
    scrollbar = f"viewing lines [{loc} - {end_loc - 1}] of {total_lines - 1}"
    header += f"\n**{scrollbar}**\n\n"

    return f"[{cursor}] {header}{body}"


# ---------------------------------------------------------------------------
# Public tool functions  (GPT-OSS MCP interface — exact descriptions)
# ---------------------------------------------------------------------------


def search(
    query: str,
    topn: int = 10,
) -> str:
    """Searches for information related to `query` and displays `topn` results.

    Args:
        query (str): Search query string.
        topn (int): Number of results to display (default: 10).

    Returns:
        str: Formatted search results page with numbered links.
    """
    searcher = _get_searcher()
    results = searcher.search(query, k=topn)

    # Build a browsable results page (same layout as GPT-OSS search)
    result_chunks: List[str] = []
    urls: Dict[str, str] = {}
    snippets_map: Dict[str, Any] = {}

    for i, r in enumerate(results):
        snippet = r.get("snippet", "")
        # Truncate for the results listing
        display = snippet[:300] + "..." if len(snippet) > 300 else snippet
        link_title = FIND_PAGE_LINK_FORMAT.format(idx=i, title=r["docid"])
        result_chunks.append(f"{link_title}\nScore: {r['score']:.4f}\n{display}")
        urls[str(i)] = r["docid"]
        snippets_map[str(i)] = {"docid": r["docid"], "snippet": snippet}

    text = (
        "\n\n".join(result_chunks)
        if result_chunks
        else f"No results found for: `{query}`"
    )

    page = _PageContents(
        url=f"search?q={quote(query)}",
        title=f"Search results for: `{query}`",
        text=text,
        urls=urls,
        snippets=snippets_map,
    )
    _state.add_page(page)
    return _show_page(page, _state.current_cursor, loc=0)


def open(
    id: Union[int, str] = -1,
    cursor: int = -1,
    loc: int = -1,
    num_lines: int = -1,
    view_source: bool = False,
) -> str:
    """Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.
    Valid link ids are displayed with the formatting: `【{id}†.*】`.
    If `cursor` is not provided, the most recent page is implied.
    If `id` is a string, it is treated as a fully qualified URL associated with `source`.
    If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
    Use this function without `id` to scroll to a new location of an opened page.

    Args:
        id (int | str): Link ID (int >= 0) to click, or document ID (str) to
            open directly. Use -1 (default) to scroll the current page.
        cursor (int): Which page in the page stack to reference (-1 for
            current page).
        loc (int): Line number to start viewing from (-1 for automatic).
        num_lines (int): Number of lines to display (-1 for automatic).
        view_source (bool): If True, show raw source text (default: False).

    Returns:
        str: Formatted page content with line numbers and scrollbar.
    """
    searcher = _get_searcher()
    stay_on_current_page = False

    if isinstance(id, str):
        # Direct document ID open
        docid = id
        doc = searcher.get_document(docid)
        if doc is None:
            return f"Error: Document `{docid}` not found."
        page = _PageContents(url=docid, title=docid, text=doc["text"])

    elif id >= 0:
        # Click a link on current/specified page
        curr_page = _state.get_page(cursor)
        try:
            docid = curr_page.urls[str(id)]
        except KeyError:
            raise ValueError(f"Invalid link id `{id}`.")

        doc = searcher.get_document(docid)
        if doc is None:
            return f"Error: Document `{docid}` not found."
        page = _PageContents(url=docid, title=docid, text=doc["text"])

    else:
        # Scroll current page
        stay_on_current_page = True
        page = _state.get_page(cursor)

    if not stay_on_current_page:
        _state.add_page(page)

    if loc < 0:
        loc = 0

    return _show_page(page, _state.current_cursor, loc=loc, num_lines=num_lines)


def find(
    pattern: str,
    cursor: int = -1,
) -> str:
    """Finds exact matches of `pattern` in the current page, or the page given by `cursor`.

    Args:
        pattern (str): Text pattern to search for (case-insensitive).
        cursor (int): Which page to search (-1 for current page).

    Returns:
        str: Page showing match locations with surrounding context.
    """
    page = _state.get_page(cursor)
    if page.snippets is not None:
        raise ValueError(
            "Cannot run `find` on search results page or find results page"
        )

    lines = _wrap_lines(page.text)
    txt = "\n".join(lines)
    without_links = _strip_links(txt)
    lines = without_links.split("\n")

    max_results = 50
    num_show_lines = 4
    result_chunks: List[str] = []
    urls: Dict[str, str] = {}
    snippets_map: Dict[str, Any] = {}

    line_idx = 0
    match_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if pattern.lower() not in line.lower():
            line_idx += 1
            continue

        snippet = "\n".join(lines[line_idx : line_idx + num_show_lines])
        link_title = FIND_PAGE_LINK_FORMAT.format(
            idx=match_idx, title=f"match at L{line_idx}"
        )
        result_chunks.append(f"{link_title}\n{snippet}")
        urls[str(match_idx)] = page.url
        snippets_map[str(match_idx)] = {
            "url": page.url,
            "text": snippet,
            "title": f"#{match_idx}",
            "line_idx": line_idx,
        }

        if len(result_chunks) == max_results:
            break
        match_idx += 1
        line_idx += num_show_lines

    if result_chunks:
        display_text = "\n\n".join(result_chunks)
    else:
        display_text = f"No `find` results for pattern: `{pattern}`"

    result_page = _PageContents(
        url=f"{page.url}/find?pattern={quote(pattern)}",
        title=f"Find results for text: `{pattern}` in `{page.title}`",
        text=display_text,
        urls=urls,
        snippets=snippets_map,
    )
    _state.add_page(result_page)
    return _show_page(result_page, _state.current_cursor, loc=0)
