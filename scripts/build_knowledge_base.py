#!/usr/bin/env python3
"""Build an auditable Markdown/JSONL knowledge base from Zotero and OCR output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class BibEntry:
    entry_type: str
    citekey: str
    fields: dict[str, str]
    raw: str
    bib_path: Path
    line: int
    collection: str
    attachments: list[Path]


@dataclass
class OcrDocument:
    directory: Path
    metadata: dict[str, Any]
    text: str

    @property
    def source_pdf(self) -> str:
        value = self.metadata.get("source_pdf")
        return str(value) if value else ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link Zotero BibTeX metadata, PDF attachments, and OCR Markdown."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["."],
        help="Zotero export directories or .bib files (default: current directory).",
    )
    parser.add_argument(
        "--bib",
        action="append",
        default=[],
        help="Additional .bib file; may be repeated.",
    )
    parser.add_argument(
        "--ocr-dir",
        action="append",
        default=None,
        help="OCR directory from ocr_zotero.py; may be repeated (default: ocr_results).",
    )
    parser.add_argument(
        "--output-dir", default="knowledge_base", help="Output directory (default: knowledge_base)."
    )
    parser.add_argument(
        "--chunk-size", type=positive_int, default=4000, help="Maximum chunk size in characters."
    )
    parser.add_argument(
        "--chunk-overlap", type=nonnegative_int, default=300, help="Chunk overlap in characters."
    )
    parser.add_argument(
        "--include-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create metadata-only documents when OCR is missing (default: true).",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def discover_bib_files(inputs: Iterable[str], extras: Iterable[str]) -> list[Path]:
    found: dict[str, Path] = {}
    missing: list[str] = []
    for raw in [*inputs, *extras]:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() == ".bib":
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("*.bib")
        else:
            missing.append(raw)
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            found[resolved.as_posix()] = resolved
    if missing:
        raise SystemExit("Input does not exist or is not a directory/.bib: " + ", ".join(missing))
    return [found[key] for key in sorted(found)]


def find_matching_delimiter(text: str, opening: int, open_char: str, close_char: str) -> int:
    depth = 0
    quoted = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced BibTeX delimiters")


def raw_bib_entries(text: str, source: Path) -> Iterator[tuple[str, int]]:
    pattern = re.compile(r"@([A-Za-z]+)\s*([({])", re.M)
    position = 0
    while match := pattern.search(text, position):
        opening = match.start(2)
        open_char = match.group(2)
        close_char = "}" if open_char == "{" else ")"
        try:
            ending = find_matching_delimiter(text, opening, open_char, close_char)
        except ValueError as exc:
            line = text.count("\n", 0, match.start()) + 1
            raise SystemExit(f"{source}:{line}: {exc}") from exc
        yield text[match.start() : ending + 1], text.count("\n", 0, match.start()) + 1
        position = ending + 1


def split_top_level(value: str, delimiter: str = ",") -> list[str]:
    result: list[str] = []
    start = 0
    braces = 0
    quoted = False
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            braces += 1
        elif not quoted and char == "}":
            braces -= 1
        elif not quoted and braces == 0 and char == delimiter:
            result.append(value[start:index])
            start = index + 1
    result.append(value[start:])
    return result


def unwrap_bib_value(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    while len(value) >= 2:
        if value[0] == '"' and value[-1] == '"':
            value = value[1:-1].strip()
            continue
        if value[0] == "{" and value[-1] == "}" and outer_braces_wrap(value):
            value = value[1:-1].strip()
            continue
        break
    return re.sub(r"\s+", " ", value).strip()


def outer_braces_wrap(value: str) -> bool:
    """Return true only when the first brace closes at the final character."""
    depth = 0
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index == len(value) - 1
    return False


def attachment_paths(file_field: str, bib_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for item in file_field.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        candidate = ""
        if len(parts) >= 3:
            candidate = ":".join(parts[1:-1])
        elif len(parts) == 2:
            candidate = parts[1]
        if not candidate:
            continue
        path = Path(candidate.replace("\\", "/")).expanduser()
        if not path.is_absolute():
            path = bib_dir / path
        if path.suffix.lower() == ".pdf":
            paths.append(path.resolve())
    return paths


def collection_name(bib_path: Path) -> str:
    return bib_path.parent.name or bib_path.stem


def parse_bib_file(path: Path) -> list[BibEntry]:
    text = path.read_text(encoding="utf-8-sig")
    entries: list[BibEntry] = []
    for raw, line in raw_bib_entries(text, path):
        header = re.match(r"@([A-Za-z]+)\s*[({]\s*([^,\s]+)\s*,", raw, re.S)
        if not header or header.group(1).lower() in {"comment", "preamble", "string"}:
            continue
        fields: dict[str, str] = {}
        body = raw[header.end() : -1]
        for part in split_top_level(body):
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            fields[name.strip().lower()] = unwrap_bib_value(value)
        entries.append(
            BibEntry(
                entry_type=header.group(1),
                citekey=header.group(2).strip(),
                fields=fields,
                raw=raw.strip(),
                bib_path=path,
                line=line,
                collection=collection_name(path),
                attachments=attachment_paths(fields.get("file", ""), path.parent),
            )
        )
    return entries


def load_ocr_documents(roots: Iterable[str]) -> list[OcrDocument]:
    documents: list[OcrDocument] = []
    for raw in roots:
        root = Path(raw).expanduser()
        if not root.exists():
            continue
        meta_files = [root] if root.is_file() and root.name == "meta.json" else root.rglob("meta.json")
        for meta_path in sorted(meta_files, key=lambda p: p.as_posix()):
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Cannot read {meta_path}: {exc}") from exc
            if not isinstance(metadata, dict):
                continue
            result_path = meta_path.parent / "result.md"
            text = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
            documents.append(OcrDocument(meta_path.parent, metadata, text.strip()))
    return documents


def normalized_path(value: str | Path) -> str:
    return Path(value).expanduser().resolve().as_posix()


def build_ocr_lookup(documents: Iterable[OcrDocument]) -> dict[str, list[OcrDocument]]:
    lookup: dict[str, list[OcrDocument]] = {}
    for document in documents:
        source = document.source_pdf
        if not source:
            continue
        keys = {normalized_path(source), Path(source).name.casefold(), Path(source).stem.casefold()}
        for key in keys:
            lookup.setdefault(key, []).append(document)
    return lookup


def match_ocr(entry: BibEntry, lookup: dict[str, list[OcrDocument]]) -> tuple[OcrDocument | None, str]:
    for attachment in entry.attachments:
        exact = lookup.get(normalized_path(attachment), [])
        if len(exact) == 1:
            return exact[0], "path"
        by_name = lookup.get(attachment.name.casefold(), [])
        if len(by_name) == 1:
            return by_name[0], "filename"
        by_stem = lookup.get(attachment.stem.casefold(), [])
        if len(by_stem) == 1:
            return by_stem[0], "stem"
    return None, ""


def slug(value: str, fallback: str = "document") -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("._-") or fallback


def markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def json_scalar(value: Any) -> str:
    return json.dumps(value or "", ensure_ascii=False)


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def unique_doc_path(entry: BibEntry, output: Path, used: set[str]) -> Path:
    collection = slug(entry.collection, "collection")
    stem = slug(entry.citekey)
    relative = f"docs/{collection}/{stem}.md"
    if relative in used:
        suffix = slug(entry.bib_path.stem)
        relative = f"docs/{collection}/{stem}-{suffix}.md"
    counter = 2
    candidate = relative
    while candidate in used:
        candidate = relative[:-3] + f"-{counter}.md"
        counter += 1
    used.add(candidate)
    return output / candidate


def render_document(entry: BibEntry, ocr: OcrDocument | None, match_method: str) -> str:
    fields = entry.fields
    title = fields.get("title") or entry.citekey
    status = "available" if ocr and ocr.text else "missing"
    backend = str(ocr.metadata.get("backend", "")) if ocr else ""
    attachment_lines = "\n".join(
        f"- `{relative_or_absolute(path, Path.cwd())}`" for path in entry.attachments
    ) or "- 无"
    ocr_text = ocr.text if ocr and ocr.text else "_尚无 OCR 正文；本文件当前仅含 Zotero/BibTeX 元数据。_"
    return f"""---
citekey: {json_scalar(entry.citekey)}
title: {json_scalar(title)}
authors: {json_scalar(fields.get('author', ''))}
year: {json_scalar(fields.get('year', ''))}
doi: {json_scalar(fields.get('doi', ''))}
url: {json_scalar(fields.get('url', ''))}
collection: {json_scalar(entry.collection)}
ocr_status: {json_scalar(status)}
ocr_backend: {json_scalar(backend)}
---

# {title}

## 引用信息

- Citekey: `{entry.citekey}`
- Authors: {fields.get('author', '')}
- Year: {fields.get('year', '')}
- DOI: {fields.get('doi', '')}
- URL: {fields.get('url', '')}
- BibTeX: `{entry.bib_path.as_posix()}:{entry.line}`
- OCR: `{status}`{f' / `{backend}` / match `{match_method}`' if ocr else ''}

## 摘要

{fields.get('abstract', '') or '_Zotero 中没有摘要。_'}

## PDF 附件

{attachment_lines}

## OCR 正文

{ocr_text}

## 原始 BibTeX

```bibtex
{entry.raw}
```
"""


def split_long_text(text: str, maximum: int, overlap: int) -> Iterator[str]:
    if len(text) <= maximum:
        if text.strip():
            yield text.strip()
        return
    start = 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            candidates = [text.rfind("\n\n", start, end), text.rfind("\n", start, end), text.rfind(" ", start, end)]
            boundary = max(candidates)
            if boundary > start + maximum // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)


def chunks_for(text: str, maximum: int, overlap: int) -> list[str]:
    sections = re.split(r"(?=^#{1,3}\s+)", text, flags=re.M)
    chunks: list[str] = []
    buffer = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        candidate = f"{buffer}\n\n{section}".strip() if buffer else section
        if len(candidate) <= maximum:
            buffer = candidate
            continue
        if buffer:
            chunks.extend(split_long_text(buffer, maximum, overlap))
        buffer = section
    if buffer:
        chunks.extend(split_long_text(buffer, maximum, overlap))
    return chunks


def write_index(manifest: list[dict[str, Any]], output: Path) -> None:
    rows = [
        "| Collection | Citekey | Year | Title | OCR | Backend | Document |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in manifest:
        relative = str(item["doc_path"])
        path = Path(relative)
        rows.append(
            "| {collection} | `{citekey}` | {year} | {title} | {status} | {backend} | [{name}]({url}) |".format(
                collection=markdown_cell(item["collection"]),
                citekey=markdown_cell(item["citekey"]),
                year=markdown_cell(item["year"]),
                title=markdown_cell(item["title"]),
                status=markdown_cell(item["ocr_status"]),
                backend=markdown_cell(item["ocr_backend"]),
                name=path.name,
                url=urllib_quote_path(relative),
            )
        )
    available = sum(item["ocr_status"] == "available" for item in manifest)
    content = f"""# Zotero OCR Knowledge Base

这是由 `build_knowledge_base.py` 生成的索引。引用前请核对具体文档中的 Citekey、DOI、原始 BibTeX 和附件路径；`ocr_status=missing` 表示没有读取全文。

- 文献数：{len(manifest)}
- 已有 OCR：{available}
- 缺少 OCR：{len(manifest) - available}
- 机器清单：`manifest.jsonl`
- RAG 分块：`chunks.jsonl`

{chr(10).join(rows)}
"""
    (output / "index.md").write_text(content, encoding="utf-8")


def urllib_quote_path(path: str) -> str:
    from urllib.parse import quote

    return quote(path, safe="/")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.chunk_overlap >= args.chunk_size:
        raise SystemExit("--chunk-overlap must be smaller than --chunk-size.")
    bib_files = discover_bib_files(args.inputs, args.bib)
    if not bib_files:
        raise SystemExit("No .bib files found.")
    entries = [entry for path in bib_files for entry in parse_bib_file(path)]
    ocr_roots = args.ocr_dir if args.ocr_dir is not None else ["ocr_results"]
    ocr_documents = load_ocr_documents(ocr_roots)
    lookup = build_ocr_lookup(ocr_documents)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    used_paths: set[str] = set()
    matched_directories: set[str] = set()
    manifest: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []

    for entry in entries:
        ocr, method = match_ocr(entry, lookup)
        if not ocr and not args.include_missing:
            continue
        if ocr:
            matched_directories.add(ocr.directory.resolve().as_posix())
        doc_path = unique_doc_path(entry, output, used_paths)
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_document(entry, ocr, method)
        doc_path.write_text(rendered, encoding="utf-8")
        item: dict[str, Any] = {
            "collection": entry.collection,
            "citekey": entry.citekey,
            "entry_type": entry.entry_type,
            "title": entry.fields.get("title", ""),
            "author": entry.fields.get("author", ""),
            "year": entry.fields.get("year", ""),
            "doi": entry.fields.get("doi", ""),
            "url": entry.fields.get("url", ""),
            "abstract": entry.fields.get("abstract", ""),
            "bib_source": f"{relative_or_absolute(entry.bib_path, Path.cwd())}:{entry.line}",
            "attachments": [relative_or_absolute(path, Path.cwd()) for path in entry.attachments],
            "doc_path": doc_path.relative_to(output).as_posix(),
            "ocr_status": "available" if ocr and ocr.text else "missing",
            "ocr_backend": str(ocr.metadata.get("backend", "")) if ocr else "",
            "ocr_source": ocr.directory.resolve().as_posix() if ocr else "",
            "match_method": method,
        }
        manifest.append(item)
        searchable = "\n\n".join(
            value
            for value in (
                f"Citekey: {entry.citekey}",
                item["title"],
                item["author"],
                item["abstract"],
                ocr.text if ocr else "",
            )
            if value
        )
        for index, chunk in enumerate(chunks_for(searchable, args.chunk_size, args.chunk_overlap)):
            chunk_records.append(
                {
                    "id": f"{slug(entry.collection)}:{entry.citekey}:{index}",
                    "collection": entry.collection,
                    "citekey": entry.citekey,
                    "title": item["title"],
                    "year": item["year"],
                    "doi": item["doi"],
                    "doc_path": item["doc_path"],
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    manifest.sort(key=lambda item: (item["collection"].casefold(), item["citekey"].casefold()))
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for item in chunk_records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    unmatched = [
        {
            "ocr_dir": document.directory.resolve().as_posix(),
            "source_pdf": document.source_pdf,
            "status": document.metadata.get("status", ""),
        }
        for document in ocr_documents
        if document.directory.resolve().as_posix() not in matched_directories
    ]
    (output / "unmatched_ocr.json").write_text(
        json.dumps(unmatched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_index(manifest, output)
    available = sum(item["ocr_status"] == "available" for item in manifest)
    print(
        f"Built {len(manifest)} documents ({available} with OCR), "
        f"{len(chunk_records)} chunks, {len(unmatched)} unmatched OCR outputs in {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
