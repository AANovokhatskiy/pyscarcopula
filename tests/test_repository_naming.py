"""Semantic repository naming across paths and source text."""

from pathlib import Path

import pytest

from tools.check_repository_naming import SOURCE_DIRECTORIES, check_repository


def test_repository_uses_semantic_names():
    assert check_repository(Path(__file__).resolve().parents[1]) == []


@pytest.mark.parametrize("folder", SOURCE_DIRECTORIES)
@pytest.mark.parametrize("label", ["stage", "gate", "phase", "FV", "WP"])
@pytest.mark.parametrize("separator", ["", " ", "_", "-"])
def test_numbered_development_labels_are_detected_in_every_source_directory(
        tmp_path, folder, label, separator):
    path = tmp_path / folder / "contract.md"
    path.parent.mkdir(parents=True)
    text = label + separator + "12"
    path.write_text(text, encoding="utf-8")
    failure, = check_repository(tmp_path)
    assert failure.path == path.relative_to(tmp_path)
    assert failure.line == 1
    assert failure.match == text


def test_numbered_development_filenames_are_detected_even_when_empty(tmp_path):
    path = tmp_path / "benchmarks" / ("gate" + "12_manifest.json")
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")
    failure, = check_repository(tmp_path)
    assert failure.line == 0


def test_algorithm_terms_and_embedded_words_are_accepted(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(
        "two-stage estimation; gradient_gate; signal phase; "
        "propagate; delegate; stage selection; variance threshold", encoding="utf-8")
    assert check_repository(tmp_path) == []
