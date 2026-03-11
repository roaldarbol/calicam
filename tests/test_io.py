"""Tests for calicam.io — save/load round-trips."""

import numpy as np
import pytest

from calicam.io import load, load_json, load_yaml, save, save_json, save_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_results_equal(a, b):
    np.testing.assert_allclose(a.camera_matrix, b.camera_matrix, rtol=1e-6)
    np.testing.assert_allclose(a.dist_coeffs, b.dist_coeffs, rtol=1e-6)
    assert a.reprojection_error == pytest.approx(b.reprojection_error, rel=1e-6)
    assert a.image_size == b.image_size


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_save_load_json_roundtrip(tmp_path, sample_result):
    path = tmp_path / "cal.json"
    save_json(sample_result, path)
    loaded = load_json(path)
    _assert_results_equal(sample_result, loaded)


def test_json_creates_parent_dirs(tmp_path, sample_result):
    path = tmp_path / "a" / "b" / "cal.json"
    save_json(sample_result, path)
    assert path.exists()


def test_json_file_content(tmp_path, sample_result):
    import json

    path = tmp_path / "cal.json"
    save_json(sample_result, path)
    data = json.loads(path.read_text())

    assert data["image_width"] == 640
    assert data["image_height"] == 480
    assert "camera_matrix" in data
    assert "distortion_coefficients" in data
    assert "reprojection_error" in data


def test_load_json_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        load_json(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def test_save_load_yaml_roundtrip(tmp_path, sample_result):
    path = tmp_path / "cal.yaml"
    save_yaml(sample_result, path)
    loaded = load_yaml(path)
    _assert_results_equal(sample_result, loaded)


def test_yaml_creates_parent_dirs(tmp_path, sample_result):
    path = tmp_path / "x" / "y" / "cal.yaml"
    save_yaml(sample_result, path)
    assert path.exists()


def test_load_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


def test_load_yml_extension(tmp_path, sample_result):
    path = tmp_path / "cal.yml"
    save_yaml(sample_result, path)
    loaded = load_yaml(path)
    _assert_results_equal(sample_result, loaded)


# ---------------------------------------------------------------------------
# Auto-dispatch  save() / load()
# ---------------------------------------------------------------------------


def test_save_auto_json(tmp_path, sample_result):
    path = tmp_path / "cal.json"
    save(sample_result, path)
    assert path.exists()


def test_save_auto_yaml(tmp_path, sample_result):
    path = tmp_path / "cal.yaml"
    save(sample_result, path)
    assert path.exists()


def test_save_unsupported_extension_raises(tmp_path, sample_result):
    with pytest.raises(ValueError, match="Unsupported"):
        save(sample_result, tmp_path / "cal.txt")


def test_load_auto_json(tmp_path, sample_result):
    path = tmp_path / "cal.json"
    save_json(sample_result, path)
    loaded = load(path)
    _assert_results_equal(sample_result, loaded)


def test_load_auto_yaml(tmp_path, sample_result):
    path = tmp_path / "cal.yaml"
    save_yaml(sample_result, path)
    loaded = load(path)
    _assert_results_equal(sample_result, loaded)


def test_load_auto_yml(tmp_path, sample_result):
    path = tmp_path / "cal.yml"
    save_yaml(sample_result, path)
    loaded = load(path)
    _assert_results_equal(sample_result, loaded)


def test_load_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "cal.csv"
    bad.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported"):
        load(bad)


# ---------------------------------------------------------------------------
# Cross-format round-trip (write JSON, read YAML)
# ---------------------------------------------------------------------------


def test_json_to_yaml_roundtrip(tmp_path, sample_result):
    json_path = tmp_path / "cal.json"
    yaml_path = tmp_path / "cal.yaml"
    save_json(sample_result, json_path)
    intermediate = load_json(json_path)
    save_yaml(intermediate, yaml_path)
    final = load_yaml(yaml_path)
    _assert_results_equal(sample_result, final)
