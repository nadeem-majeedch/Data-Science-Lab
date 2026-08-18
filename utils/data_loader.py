"""Loading CSV and Excel datasets into pandas DataFrames.

This module centralizes every way a dataset enters the app:

* user-uploaded files (CSV / XLSX) passed as Streamlit ``UploadedFile`` objects
* sample datasets shipped with the repo under ``datasets/samples/``

All loaders raise :class:`DataLoadError` with a human-friendly message so the
UI can show clean errors instead of raw tracebacks.
"""

from io import BytesIO
from pathlib import Path

import pandas as pd

SAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "samples"

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv": "CSV",
    ".xlsx": "Excel",
    ".xls": "Excel (legacy)",
}

CSV_ENCODINGS = ("utf-8-sig", "latin-1")


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded, with a user-friendly message."""


def _has_name_and_read(source) -> bool:
    """Return True for file-like objects such as Streamlit UploadedFile."""
    return hasattr(source, "name") and hasattr(source, "read")


def _read_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or file-like object with encoding fallback."""
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            if _has_name_and_read(source):
                # File-like objects are consumed as they are read; rewind so
                # each encoding attempt starts from the beginning.
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError("The file appears to be empty.") from exc
    raise DataLoadError(
        "Could not read the CSV file. Make sure it is a valid comma-separated "
        "text file."
    ) from last_error


def _read_excel(source) -> pd.DataFrame:
    """Read an Excel workbook from a path or file-like object."""
    try:
        return pd.read_excel(source, engine="openpyxl")
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(
            "Could not read the Excel file. Make sure it is a valid .xlsx "
            "workbook."
        ) from exc


def _validate(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Run cheap sanity checks on a freshly loaded DataFrame."""
    if df is None or len(df.columns) == 0:
        raise DataLoadError(
            f"'{source_name}' does not contain any columns to work with."
        )
    if len(df) == 0:
        raise DataLoadError(f"'{source_name}' does not contain any rows.")
    return df


def load_dataset(source) -> pd.DataFrame:
    """Load a dataset from a path, a string path, or an uploaded file.

    Args:
        source: A ``Path``, ``str``, or a file-like object exposing ``.name``
            and ``.read()`` (such as a Streamlit uploaded file).

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        DataLoadError: If the file type is unsupported or the file cannot be
            parsed.
    """
    if _has_name_and_read(source):
        name = Path(source.name).name
        extension = Path(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise DataLoadError(
                f"Unsupported file type '{extension or 'none'}'. "
                f"Please upload one of: {supported}."
            )
        source.seek(0)
        raw = source.read()
        if not raw:
            raise DataLoadError(f"'{name}' appears to be empty.")
        # Normalize into a BytesIO so file-like readers (e.g. openpyxl) can
        # seek freely regardless of the uploader implementation.
        source = BytesIO(raw)
        if extension == ".csv":
            df = _read_csv(source)
        else:
            df = _read_excel(source)
        return _validate(df, name)

    path = Path(source)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DataLoadError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Supported files: {supported}."
        )
    if not path.exists():
        raise DataLoadError(f"File not found: {path.name}")
    try:
        if extension == ".csv":
            df = _read_csv(path)
        else:
            df = _read_excel(path)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Could not read '{path.name}': {exc}") from exc
    return _validate(df, path.name)


def list_sample_datasets() -> list[Path]:
    """Return the paths of all bundled sample datasets, sorted by name."""
    if not SAMPLE_DATA_DIR.exists():
        return []
    return sorted(
        path
        for path in SAMPLE_DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_sample_dataset(sample_name: str | Path) -> pd.DataFrame:
    """Load a bundled sample dataset by its file name or full path.

    Args:
        sample_name: Name of the file inside ``datasets/samples/`` or a full
            path to a sample file.

    Raises:
        DataLoadError: If the sample cannot be found or parsed.
    """
    path = Path(sample_name)
    if not path.is_absolute():
        path = SAMPLE_DATA_DIR / path
    return load_dataset(path)
