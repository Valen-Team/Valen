"""Installed evaluators must work without the repository's scripts package."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def test_installed_evaluation_entrypoints_and_coordinated_branches(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source, distribution, installed = (tmp_path / name for name in ("source", "dist", "installed"))
    source.mkdir()
    distribution.mkdir()
    for name in ("valen", "evaluation"):
        shutil.copytree(root / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(root / name, source / name)
    build = subprocess.run([sys.executable, "-I", "-c",
        "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])",
        str(distribution)], cwd=source, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout[-4000:] + build.stderr[-4000:]
    wheel, = distribution.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert "valen/training/spatial_cluster.py" in archive.namelist()
        assert not any(name.startswith("scripts/") for name in archive.namelist())
        archive.extractall(installed)
    script = r'''
import importlib.abc, json, pathlib, runpy, sys
site, work = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(site))
class RejectScripts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts" or fullname.startswith("scripts."):
            raise ModuleNotFoundError("Repository scripts are unavailable in the wheel")
sys.meta_path.insert(0, RejectScripts())
import valen, evaluation
assert pathlib.Path(valen.__file__).is_relative_to(site)
assert pathlib.Path(evaluation.__file__).is_relative_to(site)
def launch(name, arguments):
    sys.argv = [name, *arguments]
    try:
        runpy.run_module(name, run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
for name in ["evaluation.det.evaluate", "evaluation.seg.evaluate", "evaluation.gui.evaluate",
             "evaluation.reasonseg", "evaluation.examples", "evaluation.gui.diagnose", "evaluation.det.groundingme"]:
    launch(name, ["--help"])
import valen.training.spatial_cluster as coordinator
class RecordedCluster:
    primary = False
    commands = []
    def __init__(self, directory, nodes, gpus_per_node):
        assert nodes * gpus_per_node == 8
    def execute(self, command, **options):
        self.commands.append(command)
coordinator.SpatialCluster = RecordedCluster
common = ["--data-root", str(work / "data"), "--model-path", str(work / "model"),
          "--output-dir", str(work / "examples"), "--nodes", "2", "--gpus-per-node", "4"]
launch("evaluation.examples", ["--task", "gui", *common, "--launch-id", "example-smoke", "--device", "cpu"])
assert len(RecordedCluster.commands) == 1
launch("evaluation.gui.diagnose", ["--data-root", str(work / "data"),
    "--model-root", str(work / "models"), "--output-dir", str(work / "diagnostics"),
    "--nodes", "2", "--gpus-per-node", "4", "--sizes", "0.8B"])
assert len(RecordedCluster.commands) == 3
data, output = work / "reasonseg", work / "reasonseg-results"
(data / "manifests").mkdir(parents=True)
(data / "processed").mkdir()
output.mkdir()
(data / "manifests/preflight.json").write_text(json.dumps({"validated": True}))
for name in ["reasonseg_val", "reasonseg_test"]:
    (data / "processed" / (name + ".jsonl")).touch()
(output / "metrics.json").write_text(json.dumps({"benchmarks": {
    "reasonseg_val": {}, "reasonseg_test": {}}}))
launch("evaluation.reasonseg", ["--task", "det", "--data-root", str(data),
    "--model-path", str(work / "model"), "--output-dir", str(output),
    "--nodes", "2", "--gpus-per-node", "4", "--launch-id", "reasonseg-smoke"])
assert len(RecordedCluster.commands) == 4
launch("evaluation.det.groundingme", ["--data-root", str(work / "groundingme"),
    "--model-path", str(work / "model"), "--output-dir", str(work / "groundingme-results"),
    "--nodes", "2", "--gpus-per-node", "4", "--launch-id", "groundingme-smoke"])
assert len(RecordedCluster.commands) == 5
print("INSTALLED_EVALUATION_ACCEPTED")
'''
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-I", "-c", script, str(installed), str(tmp_path)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "INSTALLED_EVALUATION_ACCEPTED" in result.stdout
    (tmp_path / "package-acceptance.json").write_text(json.dumps({
        "wheel": str(wheel), "installed": str(installed),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "repository_scripts_blocked": True, "entrypoint_help_checks": 7,
        "coordinated_branches_checked": 4, "weights_loaded": False}, indent=2) + "\n")
