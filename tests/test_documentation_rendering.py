"""Regression cases for defects that a successful MkDocs build can miss."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_docs", Path(__file__).resolve().parents[1] / "tools/check_docs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize("html", [
    '<table><tr><td>Abs value</td><td>$</td><td>x</td></tr></table>',
    '<p>$$x = 1$$ where x is the parameter.</p>',
    '<ul><li>GAS: $$g_t = 1$$</li></ul>',
    r'<p>Unwrapped \(x\) or \[y\]</p>',
])
def test_detects_unwrapped_math_after_markdown_rendering(html):
    page = checker.Page(f'<article>{html}</article>')
    assert page.raw_math


def test_allows_wrapped_math_and_literal_delimiters_in_code():
    page = checker.Page(r'''<article><span class="arithmatex">\(x\)</span>
        <div class="arithmatex">\[y\]</div><pre><code>$x</code></pre>
        <p>Run <code>echo $PATH</code>.</p></article>''')
    assert not page.raw_math


def test_links_and_legacy_anchors_are_checked_against_generated_html(tmp_path):
    (tmp_path / 'index.html').write_text('''<article>
        <a href="target/#kept">valid</a><a href="target/#removed">broken</a>
        <a href="missing/">missing page</a></article>''', encoding='utf-8')
    (tmp_path / 'target').mkdir()
    (tmp_path / 'target/index.html').write_text(
        '<article><a id="kept"></a></article>', encoding='utf-8')
    pages, errors = checker.check_site(tmp_path, {'target/index.html': ['kept', 'old']})
    assert len(pages) == 2
    assert any('missing anchor: target/#removed' in value for value in errors)
    assert any('missing target: missing/' in value for value in errors)
    assert any('missing legacy anchor: old' in value for value in errors)
    assert len(errors) == 3


def test_rejects_duplicate_anchor_ids():
    page = checker.Page('<article><h2 id="same">A</h2><a id="same"></a></article>')
    assert page.duplicates == {'same'}


@pytest.mark.parametrize('failure', ['unrendered', 'mathErrors', 'raw', 'overflow'])
def test_browser_findings_fail_validation(failure):
    result = dict(unrendered=0, mathErrors=[], raw=[], overflow=False)
    result[failure] = 1
    assert checker.browser_errors(result) == [failure]
