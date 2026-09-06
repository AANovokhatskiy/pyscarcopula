"""Validate built documentation links, legacy anchors, math, and browser layout.

Run after ``mkdocs build --strict``. Browser checks require the docs extra and
``python -m playwright install chromium`` (add --with-deps on Linux CI).
"""

from __future__ import annotations

import argparse
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Thread
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
RAW_MATH = re.compile(r"\$|\\[\[\]()]")
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}


class Page(HTMLParser):
    """Collect article links/text while ignoring code and wrapped math."""

    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicates: set[str] = set()
        self.links: list[str] = []
        self.raw_math: list[str] = []
        self.stack: list[tuple[str, bool, bool]] = []
        self.has_article = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        identifier = attrs.get("id")
        if identifier:
            if identifier in self.ids:
                self.duplicates.add(identifier)
            self.ids.add(identifier)
        article = tag == "article" or bool(self.stack and self.stack[-1][1])
        ignored = (bool(self.stack and self.stack[-1][2])
                   or tag in {"pre", "code", "script", "style"}
                   or "arithmatex" in attrs.get("class", "").split())
        self.has_article |= tag == "article"
        if article and tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag not in VOID_TAGS:
            self.stack.append((tag, article, ignored))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self.stack and self.stack[-1][1] and not self.stack[-1][2]:
            if RAW_MATH.search(data):
                self.raw_math.append(data.strip()[:160])


def check_site(site: Path, anchors: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    site = site.resolve()
    pages = {p.resolve(): Page(p.read_text(encoding="utf-8"))
             for p in site.rglob("*.html")}
    errors = []
    content = []
    for path, page in sorted(pages.items()):
        relative = path.relative_to(site).as_posix()
        if page.has_article and relative != "404.html":
            content.append(relative)
        errors.extend(f"{relative}: raw math delimiter: {value!r}"
                      for value in page.raw_math)
        errors.extend(f"{relative}: duplicate id: {value}" for value in sorted(page.duplicates))
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc or url.path.startswith("/"):
                continue
            target = ((path.parent / unquote(url.path)).resolve()
                      if url.path else path)
            if target.is_dir():
                target /= "index.html"
            if not target.exists():
                errors.append(f"{relative}: missing target: {href}")
            elif url.fragment and target in pages:
                if unquote(url.fragment) not in pages[target].ids:
                    errors.append(f"{relative}: missing anchor: {href}")
    for relative, identifiers in anchors.items():
        page = pages.get((site / relative).resolve())
        for identifier in identifiers:
            if page is None or identifier not in page.ids:
                errors.append(f"{relative}: missing legacy anchor: {identifier}")
    return content, errors


# Read-only DOM inspection. The same checks cover MathJax containers introduced
# after initial HTML parsing, including literal dollars inside malformed wrappers.
BROWSER_INSPECTION = r"""() => {
  const article = document.querySelector('article');
  const raw = [];
  const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement.closest('pre,code,script,style,mjx-container')) continue;
    if (/\$|\\[\[\]()]/.test(node.textContent)) raw.push(node.textContent.trim().slice(0,160));
  }
  const math = [...article.querySelectorAll('.arithmatex')];
  return {
    title: document.title,
    math: math.length,
    rendered: article.querySelectorAll('mjx-container').length,
    unrendered: math.filter(x => !x.querySelector('mjx-container')).length,
    mathErrors: [...article.querySelectorAll('mjx-merror,[data-mjx-error]')].map(x => x.textContent),
    raw,
    overflow: document.documentElement.scrollWidth > innerWidth + 1,
    width: innerWidth,
    scrollWidth: document.documentElement.scrollWidth
  };
}"""


def browser_errors(result: dict) -> list[str]:
    return [name for name in ("unrendered", "mathErrors", "raw", "overflow") if result[name]]


def check_browser(site: Path, pages: list[str], output: Path) -> tuple[list[dict], list[str]]:
    from playwright.sync_api import sync_playwright

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site.resolve())))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results, errors = [], []
    output.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            for width in (1280, 390):
                page = browser.new_page(viewport={"width": width, "height": 900})
                for relative in pages:
                    label = f"{relative}@{width}"
                    try:
                        page.goto(f"http://127.0.0.1:{server.server_port}/{relative}")
                        page.wait_for_function("() => window.MathJax && MathJax.startup && MathJax.startup.promise")
                        page.evaluate("() => MathJax.startup.promise")
                        page.evaluate("() => document.fonts.ready")
                        result = page.evaluate(BROWSER_INSPECTION)
                        result["page"] = relative
                        results.append(result)
                        failed = browser_errors(result)
                        if failed:
                            errors.append(f"{label}: {', '.join(failed)}")
                    except Exception as exc:
                        errors.append(f"{label}: {type(exc).__name__}: {exc}")
                        failed = True
                    if failed:
                        page.screenshot(path=str(output / f"{relative.replace('/', '_')}-{width}.png"), full_page=True)
                page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=ROOT / "site")
    parser.add_argument("--anchors", type=Path, default=ROOT / "tests/fixtures/documentation_anchors.json")
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/docs-check")
    args = parser.parse_args()
    anchors = json.loads(args.anchors.read_text(encoding="utf-8"))
    pages, errors = check_site(args.site_dir, anchors)
    if not pages:
        errors.append("No documentation articles found; build the site first.")
    results = []
    if args.browser and pages:
        results, browser_failures = check_browser(args.site_dir, pages, args.output_dir)
        errors.extend(browser_failures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps({
        "pages": pages, "errors": errors, "browser": results,
    }, indent=2), encoding="utf-8")
    for error in errors:
        print(error)
    print(f"Checked {len(pages)} pages, {len(results)} browser views: {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
