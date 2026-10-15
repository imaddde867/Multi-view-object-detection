"""Tests for the config module: one merge semantic for every pipeline."""

from __future__ import annotations

import pytest

from multiview.config import merge


class TestMerge:
    def test_scalar_override_wins(self) -> None:
        merged = merge({"a": 1}, {"a": 2})
        assert merged["a"] == 2

    def test_none_override_means_keep_base(self) -> None:
        # The single semantic: CLI flags that were not given (None) must not
        # wipe values from the YAML config. This is what merge_dicts did;
        # the three inline copies in run/pipeline/train assigned None instead.
        merged = merge({"detector": {"conf": 0.25}}, {"detector": {"conf": None}})
        assert merged["detector"]["conf"] == 0.25

    def test_none_override_introduces_no_key(self) -> None:
        merged = merge({}, {"conf": None})
        assert merged == {}

    def test_recursive_dict_merge(self) -> None:
        merged = merge({"runtime": {"device": "cpu", "half": True}}, {"runtime": {"device": "cuda:0"}})
        assert merged["runtime"] == {"device": "cuda:0", "half": True}

    def test_dict_override_replaces_scalar_base(self) -> None:
        merged = merge({"sam3": None}, {"sam3": {"confidence": 0.4}})
        assert merged["sam3"] == {"confidence": 0.4}

    def test_does_not_mutate_base_or_nested_dicts(self) -> None:
        # run.py's inline merge did base[k].update(v) on the dict returned by
        # _default_cfg(); merge must rebuild instead of mutating.
        base = {"runtime": {"device": "cpu", "half": False}, "a": 1}
        defaults_copy = {"runtime": dict(base["runtime"]), "a": 1}
        merge(base, {"runtime": {"half": True}})
        assert base == defaults_copy

    def test_empty_override_returns_equal_copy(self) -> None:
        base = {"a": 1, "b": {"c": 2}}
        merged = merge(base, {})
        assert merged == base
        assert merged is not base

    def test_list_override_replaces_wholesale(self) -> None:
        merged = merge({"groups": ["a", "b"]}, {"groups": ["c"]})
        assert merged["groups"] == ["c"]
