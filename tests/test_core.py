import copy

import numpy as np
import pandas as pd
import pytest

from windcast.config import LOCAL_OFFSET, PUBLICATION_DELAY_H, to_utc
from windcast.data import load_scada
from windcast.model import MODEL_FILE, WindModel
from windcast.turbines import get_turbine
from windcast.weather import asof, lead_for, target_hours

ISSUE = pd.Timestamp("2026-02-10 10:00")


def test_lead_mapping_boundaries():
    assert list(lead_for([1, 17, 18, 41, 42, 48])) == [1, 1, 2, 2, 3, 3]


def test_asof_uses_only_runs_available_at_issue():
    """Для каждого часа прогона с заблаговременностью N: t + 1ч − 24N ≤ T − задержка публикации."""
    issue = to_utc(ISSUE)
    w = asof("T1", issue)
    run_time = w.index + pd.Timedelta(hours=1) - pd.to_timedelta(24 * w["lead_day"], unit="h")
    assert (run_time <= issue - pd.Timedelta(hours=PUBLICATION_DELAY_H)).all()
    assert len(w) == 48 and w.index[0] == issue.floor("h") + pd.Timedelta(hours=1)


def test_asof_ignores_data_published_after_issue():
    """Подмена всех значений с лидом 1 не должна менять часы, для которых нужен лид ≥ 2."""
    from unittest import mock

    from windcast import weather
    issue = to_utc(ISSUE)
    base = asof("T1", issue)
    caches = {m: weather.load_cache("T1", m) for m in ("gfs", "icon", "ecmwf")}
    spoiled = caches["gfs"].copy()
    spoiled[[c for c in spoiled.columns if c.endswith("_d1")]] = 999.0
    caches["gfs"] = spoiled
    with mock.patch.object(weather, "load_cache", lambda tid, m: caches[m]):
        w = asof("T1", issue)
    later = base["lead_day"] >= 2
    assert np.allclose(w.loc[later, "gfs_wind_speed_100m"], base.loc[later, "gfs_wind_speed_100m"])
    assert (w.loc[~later, "gfs_wind_speed_100m"] == 999.0).all()


def test_scada_converted_from_utc6():
    raw = pd.read_csv(get_turbine("T1").history_path, nrows=1)
    df = load_scada(get_turbine("T1"))
    assert df.index[0] == pd.Timestamp(raw.iloc[0, 1]) - LOCAL_OFFSET


def test_target_hours_start_next_full_hour():
    t = target_hours(pd.Timestamp("2026-02-01 04:30"))
    assert t[0] == pd.Timestamp("2026-02-01 05:00") and len(t) == 48


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="сначала windcast train")
def test_agent_recomputes_only_when_inputs_change(tmp_path):
    from windcast.agent.loop import RuleAgent
    model = WindModel.load()
    r1 = RuleAgent(ISSUE, runs_dir=tmp_path, online=False, model=model).run()
    assert r1["status"] == "ok"
    tot = r1["result"]
    assert set(tot.turbine) == {"T1", "T2", "ВЭС"}
    assert tot[["p10", "p50", "p90"]].le(1).all().all() and tot.p10.le(tot.p50 + 1e-9).all()

    r2 = RuleAgent(ISSUE, runs_dir=tmp_path, online=False, model=model).run()
    assert r2["status"] == "unchanged"

    m2 = copy.deepcopy(model)
    m2.meta["trained_at"] = "retrained"
    r3 = RuleAgent(ISSUE, runs_dir=tmp_path, online=False, model=m2).run()
    assert r3["status"] == "ok"
    assert (tmp_path / ISSUE.strftime("%Y%m%d_%H%M") / "versions").exists()
