"""CLI smoke tests: every subcommand must resolve its names and call
through to the pipeline entry point.

Regression origin: the config-module rename left `merge_dicts(...)` calls
in cli.py while the import became `merge` — import of the module still
succeeded, so only an end-to-end dispatch caught the NameError. These
tests stub the four heavy entry points and exercise main() per command.
"""

from __future__ import annotations

import pytest

import multiview.cli as cli


@pytest.fixture()
def stubbed(monkeypatch):
    calls = {}

    def _stub(name):
        def fn(*args, **kwargs):
            calls[name] = (args, kwargs)
        return fn

    monkeypatch.setattr(cli, "label_videos", _stub("label"))
    monkeypatch.setattr(cli, "train_yolo", _stub("train"))
    monkeypatch.setattr(cli, "run_multiview", _stub("run"))
    monkeypatch.setattr(cli, "verify_dataset", lambda *a, **k: _fake_report(calls, a, k))
    return calls


class _FakeReport:
    warnings: list = []
    errors: list = []
    root = "."
    names: list = []
    images = 0
    empty_labels = 0
    boxes_total = 0
    boxes_per_class = {}


def _fake_report(calls, args, kwargs):
    calls["verify"] = (args, kwargs)
    return _FakeReport()


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["multiview", *argv])
    cli.main()


class TestDispatch:
    def test_label(self, stubbed, monkeypatch) -> None:
        _run(monkeypatch, ["label", "--targets", "person,car"])
        assert "label" in stubbed

    def test_label_overrides_flow_into_config(self, stubbed, monkeypatch) -> None:
        _run(monkeypatch, ["label", "--targets", "person,car", "--conf", "0.4"])
        cfg = stubbed["label"][0][0]
        assert cfg["targets"] == "person,car"
        assert cfg["conf"] == pytest.approx(0.4)

    def test_train(self, stubbed, monkeypatch) -> None:
        _run(monkeypatch, ["train", "--model", "yolo11m.pt", "--epochs", "10"])
        cfg = stubbed["train"][0][0]
        assert cfg["model"] == "yolo11m.pt"
        assert cfg["epochs"] == 10

    def test_run_with_yaml_and_overrides(self, stubbed, monkeypatch, tmp_path) -> None:
        cfg_path = tmp_path / "sys.yaml"
        cfg_path.write_text(
            "cameras:\n  cam1: {source: a.mp4}\ngroups:\n  g1: [cam1]\n"
            "detector: {model: w.pt}\noutput: {dir: out}\nrun: {}\n"
        )
        _run(monkeypatch, ["run", "--config", str(cfg_path), "--max-frames", "5"])
        cfg = stubbed["run"][0][0]
        assert cfg["run"]["max_frames"] == 5

    def test_verify(self, stubbed, monkeypatch) -> None:
        _run(monkeypatch, ["verify", "--dataset", "data/processed/ds/dataset.yaml"])
        assert "verify" in stubbed
