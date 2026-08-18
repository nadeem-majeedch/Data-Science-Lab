"""Tests for the dataset loading utilities."""

import io
from pathlib import Path

import pandas as pd
import pytest

from utils.data_loader import (
    DataLoadError,
    list_sample_datasets,
    load_dataset,
    load_sample_dataset,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "samples"


class FakeUpload:
    """Minimal stand-in for a Streamlit UploadedFile."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._buffer = io.BytesIO(content)

    def read(self, *args, **kwargs):
        return self._buffer.read(*args, **kwargs)

    def seek(self, offset):
        return self._buffer.seek(offset)


@pytest.fixture(scope="module")
def sample_csv() -> Path:
    return SAMPLE_DIR / "student_grades.csv"


@pytest.fixture(scope="module")
def sample_xlsx() -> Path:
    return SAMPLE_DIR / "sales_records.xlsx"


def test_load_csv_from_path(sample_csv):
    df = load_dataset(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] == 9
    assert "student_id" in df.columns


def test_load_xlsx_from_path(sample_xlsx):
    df = load_dataset(sample_xlsx)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "order_id" in df.columns


def test_load_csv_from_upload_like(sample_csv):
    content = sample_csv.read_bytes()
    df = load_dataset(FakeUpload("uploaded.csv", content))
    assert len(df) > 0
    assert "student_id" in df.columns


def test_load_xlsx_from_upload_like(sample_xlsx):
    content = sample_xlsx.read_bytes()
    df = load_dataset(FakeUpload("uploaded.xlsx", content))
    assert len(df) > 0
    assert "order_id" in df.columns


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "data.txt"
    bad.write_text("a,b\n1,2")
    with pytest.raises(DataLoadError):
        load_dataset(bad)


def test_missing_file_raises():
    with pytest.raises(DataLoadError):
        load_dataset(SAMPLE_DIR / "does_not_exist.csv")


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(DataLoadError):
        load_dataset(empty)


def test_malformed_csv_raises(tmp_path):
    malformed = tmp_path / "malformed.csv"
    malformed.write_text('a,b\n"unclosed quote\n')
    with pytest.raises(DataLoadError):
        load_dataset(malformed)


def test_latin1_csv_falls_back_successfully(tmp_path):
    # "caf\xe9" is valid latin-1 but invalid UTF-8; the fallback must handle it.
    latin1 = tmp_path / "latin1.csv"
    latin1.write_bytes("name,note\nAna,caf\xe9\n".encode("latin-1"))
    df = load_dataset(latin1)
    assert df.loc[0, "note"] == "caf\u00e9"


def test_invalid_xlsx_raises(tmp_path):
    bad = tmp_path / "fake.xlsx"
    bad.write_bytes(b"this is not a real xlsx workbook")
    with pytest.raises(DataLoadError):
        load_dataset(bad)


def test_upload_with_bad_extension_raises():
    with pytest.raises(DataLoadError):
        load_dataset(FakeUpload("data.zip", b"whatever"))


def test_list_sample_datasets_returns_supported_files():
    samples = list_sample_datasets()
    assert len(samples) >= 2
    for sample in samples:
        assert sample.suffix.lower() in {".csv", ".xlsx"}


def test_load_sample_dataset_by_name(sample_csv):
    df = load_sample_dataset(sample_csv.name)
    assert len(df) > 0
