"""Tests for ValidationDB audio file linking (_link_audio_files).

These cover the failure modes that caused empty audio_file_path values and
the silent infinite 'Analyzing Audio' spinner in the Validation interface:
- predictions CSV filename columns containing full Windows/Unix paths
- case differences between CSV filenames and files on disk
- mismatched audio extensions
- missing/wrong audio directory
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.validation_db import ValidationDB


@pytest.fixture
def audio_dir(tmp_path):
    """Create a fake audio directory with nested files."""
    (tmp_path / "site1").mkdir()
    (tmp_path / "site1" / "rec_001.wav").write_bytes(b"fake")
    (tmp_path / "rec_002.WAV").write_bytes(b"fake")
    (tmp_path / "rec_003.flac").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_bytes(b"not audio")
    return tmp_path


@pytest.fixture
def db():
    return ValidationDB()


def test_exact_basename_match(db, audio_dir):
    paths = db._link_audio_files(["rec_001.wav"], str(audio_dir))
    assert paths == [str(audio_dir / "site1" / "rec_001.wav")]


def test_filename_with_unix_path_is_matched(db, audio_dir):
    paths = db._link_audio_files(["/data/deploy1/rec_001.wav"], str(audio_dir))
    assert paths == [str(audio_dir / "site1" / "rec_001.wav")]


def test_filename_with_windows_path_is_matched(db, audio_dir):
    paths = db._link_audio_files([r"C:\data\deploy1\rec_001.wav"], str(audio_dir))
    assert paths == [str(audio_dir / "site1" / "rec_001.wav")]


def test_case_insensitive_match(db, audio_dir):
    paths = db._link_audio_files(["REC_001.WAV", "rec_002.wav"], str(audio_dir))
    assert paths == [
        str(audio_dir / "site1" / "rec_001.wav"),
        str(audio_dir / "rec_002.WAV"),
    ]


def test_extension_mismatch_falls_back_to_stem(db, audio_dir):
    # CSV says .wav but the file on disk is .flac
    paths = db._link_audio_files(["rec_003.wav"], str(audio_dir))
    assert paths == [str(audio_dir / "rec_003.flac")]


def test_filename_without_extension_matches_stem(db, audio_dir):
    paths = db._link_audio_files(["rec_001"], str(audio_dir))
    assert paths == [str(audio_dir / "site1" / "rec_001.wav")]


def test_unmatched_filename_returns_empty_string(db, audio_dir):
    paths = db._link_audio_files(["does_not_exist.wav"], str(audio_dir))
    assert paths == [""]


def test_missing_audio_directory_returns_empty_strings(db, tmp_path):
    paths = db._link_audio_files(
        ["rec_001.wav", "rec_002.wav"], str(tmp_path / "nonexistent")
    )
    assert paths == ["", ""]


def test_non_audio_files_are_ignored(db, audio_dir):
    paths = db._link_audio_files(["notes.txt"], str(audio_dir))
    assert paths == [""]
