import pandas as pd

from aero_optim.main.doe import merge


def test_merge_shards(tmp_path):
    pd.DataFrame({"blade": ["baseline", "0000", "0002"], "status": ["ok", "ok", "mesh_failed"]}).to_csv(
        tmp_path / "doe_results_shard0of2.csv", index=False)
    pd.DataFrame({"blade": ["0001", "0003"], "status": ["ok", "ok"]}).to_csv(tmp_path / "doe_results_shard1of2.csv", index=False)
    res = merge(str(tmp_path), 2)
    assert list(res.blade) == ["baseline", "0000", "0001", "0002", "0003"]
    assert (tmp_path / "doe_results.csv").exists()
