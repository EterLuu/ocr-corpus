#!/usr/bin/env python3
"""OCR PDFs exported by Zotero with Baidu API or local Unlimited-OCR.

Both backends write the same portable layout::

    ocr_results/<document-id>/meta.json
    ocr_results/<document-id>/result.md

The knowledge-base builder only consumes this layout and therefore does not
need to know which OCR backend produced a document.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
TASK_URL = "https://aip.baidubce.com/rest/2.0/brain/online/v2/unlimited-ocr-parser/task"
QUERY_URL = f"{TASK_URL}/query"
DEFAULT_MODEL = "baidu/Unlimited-OCR"
MAX_FILE_DATA_BYTES = 50 * 1024 * 1024


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF file(s) or Zotero export directory/directories (recursive).",
    )
    parser.add_argument(
        "--output-dir", default="ocr_results", help="Output directory (default: ocr_results)."
    )
    parser.add_argument("--limit", type=positive_int, help="Only process the first N PDFs.")
    parser.add_argument(
        "--force", action="store_true", help="Reprocess documents that already have result.md."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List documents without loading a model or calling an API."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR Zotero-exported PDFs with Baidu API or local Unlimited-OCR."
    )
    subparsers = parser.add_subparsers(dest="backend", required=True)

    baidu = subparsers.add_parser("baidu", help="Use Baidu Unlimited-OCR API.")
    add_common_arguments(baidu)
    baidu.add_argument("--api-key-env", default="BAIDU_OCR_API_KEY")
    baidu.add_argument("--secret-key-env", default="BAIDU_OCR_SECRET_KEY")
    baidu.add_argument(
        "--file-mode",
        choices=("data", "url", "auto"),
        default="data",
        help="Upload base64 data, use a public URL, or switch to URL above 50 MiB.",
    )
    baidu.add_argument(
        "--url-base",
        help="Public base URL mirroring paths relative to the current directory.",
    )
    baidu.add_argument(
        "--url-map",
        type=Path,
        help="JSON object mapping local PDF paths to public URLs.",
    )
    baidu.add_argument("--poll-interval", type=positive_float, default=8.0)
    baidu.add_argument("--timeout", type=positive_float, default=1800.0)
    baidu.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit tasks and save task IDs; a later run resumes polling.",
    )
    baidu.add_argument(
        "--request-timeout", type=positive_float, default=120.0, help="HTTP timeout in seconds."
    )

    local = subparsers.add_parser("local", help="Use a local/Hugging Face Unlimited-OCR model.")
    add_common_arguments(local)
    local.add_argument(
        "--model",
        default=os.environ.get("UNLIMITED_OCR_MODEL", DEFAULT_MODEL),
        help=f"Local model directory or Hugging Face ID (default: {DEFAULT_MODEL}).",
    )
    local.add_argument("--offline", action="store_true", help="Forbid model downloads.")
    local.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    local.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    local.add_argument("--dpi", type=positive_int, default=200)
    local.add_argument("--image-size", type=positive_int, default=1024)
    local.add_argument("--max-length", type=positive_int, default=32768)
    local.add_argument("--no-repeat-ngram-size", type=positive_int, default=35)
    local.add_argument("--ngram-window", type=positive_int, default=1024)
    local.add_argument(
        "--keep-pages", action="store_true", help="Keep PDF pages rendered as PNG files."
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def discover_pdfs(inputs: Iterable[str]) -> list[Path]:
    found: dict[str, Path] = {}
    missing: list[str] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            candidates = path.rglob("*")
        elif path.is_file():
            candidates = [path]
        else:
            missing.append(raw)
            continue
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() == ".pdf":
                resolved = candidate.resolve()
                found[resolved.as_posix()] = resolved
    if missing:
        raise SystemExit("Input does not exist: " + ", ".join(missing))
    return [found[key] for key in sorted(found)]


def safe_stem(path: Path, max_length: int = 72) -> str:
    name = "".join(c if c.isalnum() or c in "._-" else "_" for c in path.stem)
    name = name.strip("._-") or "document"
    return name[:max_length].rstrip("._-")


def document_id(path: Path) -> str:
    digest = hashlib.sha256(path.resolve().as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem(path)}-{digest}"


def output_path(root: Path, pdf: Path) -> Path:
    return root / document_id(pdf)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def base_metadata(pdf: Path, backend: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "document_id": document_id(pdf),
        "source_pdf": pdf.resolve().as_posix(),
        "source_name": pdf.name,
        "source_size": pdf.stat().st_size,
        "backend": backend,
        "updated_at": utc_now(),
    }


def post_form(url: str, data: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(data).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Baidu returned invalid JSON: {body[:300]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected Baidu response: {value!r}")
    return value


def get_json(url: str, params: dict[str, str], timeout: float) -> dict[str, Any]:
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(request_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Baidu returned invalid JSON: {body[:300]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected Baidu response: {value!r}")
    return value


def check_baidu_response(response: dict[str, Any]) -> None:
    error_code = response.get("error_code")
    if error_code not in (None, 0, "0"):
        raise RuntimeError(
            f"Baidu error {error_code}: {response.get('error_msg', 'unknown error')}"
        )


def get_access_token(args: argparse.Namespace) -> str:
    api_key = os.environ.get(args.api_key_env)
    secret_key = os.environ.get(args.secret_key_env)
    if not api_key or not secret_key:
        raise SystemExit(f"Set {args.api_key_env} and {args.secret_key_env} first.")
    response = get_json(
        TOKEN_URL,
        {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key},
        args.request_timeout,
    )
    token = response.get("access_token")
    if not token:
        raise RuntimeError(f"Could not obtain a Baidu access token: {response}")
    return str(token)


def load_url_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise SystemExit("--url-map must be a JSON object of local_path -> URL strings.")
    mapping: dict[str, str] = {}
    for key, url in value.items():
        mapping[Path(key).expanduser().resolve().as_posix()] = url
        mapping[Path(key).as_posix()] = url
    return mapping


def public_url(pdf: Path, args: argparse.Namespace, mapping: dict[str, str]) -> str:
    for key in (pdf.resolve().as_posix(), pdf.as_posix()):
        if key in mapping:
            return mapping[key]
    if args.url_base:
        try:
            relative = pdf.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"{pdf} is outside the current directory; add it to --url-map.") from exc
        quoted = urllib.parse.quote(relative, safe="/")
        return urllib.parse.urljoin(args.url_base.rstrip("/") + "/", quoted)
    raise RuntimeError(f"No public URL for {pdf}; use --url-base or --url-map.")


def submit_baidu(pdf: Path, token: str, args: argparse.Namespace, mapping: dict[str, str]) -> str:
    mode = args.file_mode
    if mode == "auto":
        mode = "data" if pdf.stat().st_size <= MAX_FILE_DATA_BYTES else "url"
    form = {"file_name": pdf.name}
    if mode == "data":
        if pdf.stat().st_size > MAX_FILE_DATA_BYTES:
            raise RuntimeError(f"{pdf} exceeds 50 MiB; use --file-mode url or auto.")
        form["file_data"] = base64.b64encode(pdf.read_bytes()).decode("ascii")
    else:
        form["file_url"] = public_url(pdf, args, mapping)
    response = post_form(
        f"{TASK_URL}?access_token={urllib.parse.quote(token)}", form, args.request_timeout
    )
    check_baidu_response(response)
    result = response.get("result")
    task_id = result.get("task_id") if isinstance(result, dict) else None
    if not task_id:
        raise RuntimeError(f"Baidu response did not contain task_id: {response}")
    return str(task_id)


def query_baidu(task_id: str, token: str, args: argparse.Namespace) -> dict[str, Any]:
    response = post_form(
        f"{QUERY_URL}?access_token={urllib.parse.quote(token)}",
        {"task_id": task_id},
        args.request_timeout,
    )
    check_baidu_response(response)
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"Baidu query response did not contain a result object: {response}")
    return result


def download(url: str, destination: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "zotero-ultimate-ocr/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Cannot download OCR result from {url}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)


def run_baidu(pdfs: list[Path], args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir)
    mapping = load_url_map(args.url_map)
    token = get_access_token(args)
    failures = 0
    for index, pdf in enumerate(pdfs, 1):
        target = output_path(output_root, pdf)
        meta_path = target / "meta.json"
        result_path = target / "result.md"
        meta = read_json(meta_path)
        if result_path.exists() and not args.force:
            print(f"[{index}/{len(pdfs)}] skip {pdf.name} (complete)", file=sys.stderr)
            continue
        if args.force and result_path.exists():
            result_path.unlink()
        try:
            task_id = None if args.force else meta.get("task_id")
            submitted_now = False
            if not isinstance(task_id, str) or not task_id:
                print(f"[{index}/{len(pdfs)}] submit {pdf.name}", file=sys.stderr)
                task_id = submit_baidu(pdf, token, args, mapping)
                submitted_now = True
                meta = {**base_metadata(pdf, "baidu"), "task_id": task_id, "status": "submitted"}
                write_json(meta_path, meta)
            if args.submit_only:
                # The documented submit QPS is 2; keep batch submission below it.
                if submitted_now and index < len(pdfs):
                    time.sleep(0.55)
                continue

            started = time.monotonic()
            if submitted_now:
                time.sleep(args.poll_interval)
            while True:
                result = query_baidu(task_id, token, args)
                status = str(result.get("status", "unknown")).lower()
                meta.update({"status": status, "query_result": result, "updated_at": utc_now()})
                write_json(meta_path, meta)
                print(f"[{index}/{len(pdfs)}] {pdf.name}: {status}", file=sys.stderr)
                if status == "success":
                    markdown_url = result.get("markdown_url")
                    if not isinstance(markdown_url, str) or not markdown_url:
                        raise RuntimeError("Task succeeded but markdown_url is missing.")
                    download(markdown_url, result_path, args.request_timeout)
                    parse_url = result.get("parse_result_url")
                    if isinstance(parse_url, str) and parse_url:
                        try:
                            download(parse_url, target / "result.json", args.request_timeout)
                        except RuntimeError as exc:
                            print(f"warning: {exc}", file=sys.stderr)
                    meta.update({"status": "success", "completed_at": utc_now()})
                    write_json(meta_path, meta)
                    break
                if status == "failed":
                    raise RuntimeError(str(result.get("task_error") or "Baidu task failed"))
                if time.monotonic() - started >= args.timeout:
                    raise RuntimeError(f"Timed out after {args.timeout:g} seconds; rerun to resume.")
                time.sleep(args.poll_interval)
        except Exception as exc:
            failures += 1
            failed_meta = meta or base_metadata(pdf, "baidu")
            failed_meta.update({"status": "error", "error": str(exc), "updated_at": utc_now()})
            write_json(meta_path, failed_meta)
            print(f"error: {pdf}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def resolve_local_runtime(args: argparse.Namespace) -> tuple[Any, Any, str, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Local backend requires torch and transformers; see README.md installation instructions."
        ) from exc
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false.")
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device == "cuda" else "float32"
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    model_ref = Path(args.model).expanduser()
    model_name = model_ref.resolve().as_posix() if model_ref.exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=args.offline
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=args.offline,
        use_safetensors=True,
        torch_dtype=dtype,
    ).eval().to(device)
    return tokenizer, model, device, dtype


def render_pdf(pdf: Path, destination: Path, dpi: int) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit("Local backend requires PyMuPDF; install the local requirements.") from exc
    destination.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf)
    pages: list[Path] = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        for number, page in enumerate(document, 1):
            page_path = destination / f"page-{number:04d}.png"
            page.get_pixmap(matrix=matrix, alpha=False).save(page_path)
            pages.append(page_path)
    finally:
        document.close()
    return pages


def run_local(pdfs: list[Path], args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir)
    pending = [pdf for pdf in pdfs if args.force or not (output_path(output_root, pdf) / "result.md").exists()]
    if not pending:
        print("All documents are already complete.", file=sys.stderr)
        return 0
    tokenizer, model, device, dtype = resolve_local_runtime(args)
    failures = 0
    for index, pdf in enumerate(pending, 1):
        target = output_path(output_root, pdf)
        target.mkdir(parents=True, exist_ok=True)
        meta_path = target / "meta.json"
        if args.force and (target / "result.md").exists():
            (target / "result.md").unlink()
        temporary_dir: str | None = None
        try:
            print(f"[{index}/{len(pending)}] OCR {pdf.name}", file=sys.stderr)
            if args.keep_pages:
                page_dir = target / "pages"
            else:
                temporary_dir = tempfile.mkdtemp(prefix="zotero_ocr_pages_")
                page_dir = Path(temporary_dir)
            pages = render_pdf(pdf, page_dir, args.dpi)
            if not pages:
                raise RuntimeError("PDF contains no pages.")
            model.infer_multi(
                tokenizer,
                prompt="<image>Multi page parsing.",
                image_files=[page.as_posix() for page in pages],
                output_path=target.as_posix(),
                image_size=args.image_size,
                max_length=args.max_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                ngram_window=args.ngram_window,
                save_results=True,
            )
            result_path = target / "result.md"
            if not result_path.exists():
                raise RuntimeError("Model finished without creating result.md.")
            meta = {
                **base_metadata(pdf, "local"),
                "status": "success",
                "model": args.model,
                "device": device,
                "dtype": str(dtype),
                "page_count": len(pages),
                "completed_at": utc_now(),
            }
            write_json(meta_path, meta)
        except Exception as exc:
            failures += 1
            meta = {**base_metadata(pdf, "local"), "status": "error", "error": str(exc)}
            write_json(meta_path, meta)
            print(f"error: {pdf}: {exc}", file=sys.stderr)
        finally:
            if temporary_dir:
                shutil.rmtree(temporary_dir, ignore_errors=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pdfs = discover_pdfs(args.inputs)
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit("No PDF files found.")
    if args.dry_run:
        for pdf in pdfs:
            print(f"{document_id(pdf)}\t{pdf}")
        print(f"{len(pdfs)} PDF(s)", file=sys.stderr)
        return 0
    return run_baidu(pdfs, args) if args.backend == "baidu" else run_local(pdfs, args)


if __name__ == "__main__":
    raise SystemExit(main())
