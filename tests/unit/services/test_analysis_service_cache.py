import sys
import types

import pytest

from wjp_analyser.performance import cache_manager
from wjp_analyser.services import analysis_service


@pytest.fixture(autouse=True)
def reset_cache_instances():
    cache_manager._CACHE_INSTANCES.clear()
    yield
    cache_manager._CACHE_INSTANCES.clear()


@pytest.fixture
def stub_dxf_module(monkeypatch):
    call_counter = {"count": 0}

    class FakeArgs:
        def __init__(self, out: str = "out"):
            self.out = out
            self.material = "steel"
            self.thickness = 1.0
            self.kerf = 1.0
            self.streaming_mode = False
            self.early_simplify_tolerance = 0.0

    def fake_analyze(dxf_path, args):
        call_counter["count"] += 1
        return {"path": dxf_path, "out": args.out, "metrics": {}}

    fake_module = types.ModuleType("wjp_analyser.analysis.dxf_analyzer")
    fake_module.AnalyzeArgs = FakeArgs
    fake_module.analyze_dxf = fake_analyze

    monkeypatch.setitem(sys.modules, "wjp_analyser.analysis.dxf_analyzer", fake_module)
    return call_counter


def test_run_analysis_uses_cached_result(tmp_path, stub_dxf_module):
    dxf_file = tmp_path / "sample.dxf"
    dxf_file.write_text("dummy dxf content")
    out_dir = tmp_path / "analysis"

    first = analysis_service.run_analysis(str(dxf_file), out_dir=str(out_dir))
    second = analysis_service.run_analysis(str(dxf_file), out_dir=str(out_dir))

    assert stub_dxf_module["count"] == 1
    assert first == second

    # Simulate new process by resetting registry; disk-backed cache should persist.
    cache_manager._CACHE_INSTANCES.clear()
    third = analysis_service.run_analysis(str(dxf_file), out_dir=str(out_dir))
    assert stub_dxf_module["count"] == 1
    assert third == first
