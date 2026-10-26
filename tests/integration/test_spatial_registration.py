"""Replacement replicas cannot launch a group with stale identities or overlapping GPUs."""
import json
from pathlib import Path

import pytest

from scripts.train.schedule_spatial import refresh_group_registrations


def test_registration_waits_for_all_four_and_replaces_the_old_mapping(tmp_path):
    state = {"allocations": [{"name": name, "gpus": 8, "nodes": 2} for name in ("a", "b")],
        "execution_groups": [{"name": "det", "members": ["a", "b"], "nodes": 4, "gpus": 16,
            "registration_dir": str(tmp_path), "node_rank_map": {}}]}
    group = state["execution_groups"][0]
    for index, (name, rank) in enumerate((('a', 0), ('a', 1), ('b', 0), ('b', 1))):
        row = {"allocation": name, "replica": rank, "hostname": f"host-{index}",
               "gpu_uuids": [f"gpu-{index}-{i}" for i in range(4)]}
        (tmp_path / f"{name}.node-{rank}.json").write_text(json.dumps(row))
        refresh_group_registrations(state)
        if index < 3:
            assert not group["node_rank_map"]
    assert group["node_rank_map"] == {f"host-{i}": i for i in range(4)}
    assert state["allocations"][1]["hostnames"] == ["host-2", "host-3"]
    last = tmp_path / 'b.node-1.json'
    row = json.loads(last.read_text())
    row["gpu_uuids"][0] = "gpu-0-0"
    last.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="overlap"):
        refresh_group_registrations(state)
