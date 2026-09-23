"""Воспроизведение теста «как в прошлом»: ежедневные выпуски прогноза через агента."""
from pathlib import Path

import pandas as pd

from .config import OUTPUTS_DIR
from .forecast import TOTAL
from .model import WindModel


def run_backtest(start: str, end: str, hours=("10:00",), out_dir: Path = OUTPUTS_DIR / "feb2026",
                 agent: str = "rules", online: bool = False, period=("2026-02-01", "2026-03-01"),
                 log=print) -> pd.DataFrame:
    """Выпуски с `start` по `end` (даты, UTC+6) в каждый из `hours`; итоговые файлы — в `out_dir`.
    `period` — полуинтервал целевых часов (UTC+6) для сводной таблицы."""
    from .agent import make_agent

    model = WindModel.load()
    runs = out_dir / "runs"
    issues = [pd.Timestamp(f"{d.date()} {h}") for d in pd.date_range(start, end, freq="D") for h in hours]
    frames = []
    for issue in issues:
        res = make_agent(agent, issue, runs_dir=runs, online=online, model=model).run()
        log(f"  выпуск {issue:%d.%m.%Y %H:%M}: {res['status']}")
        if res["result"] is not None:
            frames.append(res["result"].assign(issue_local=issue))
    all_ = pd.concat(frames)
    all_.to_csv(out_dir / "all_issues.csv", index=False)
    # для каждого часа — прогноз из самого свежего выпуска
    latest = (all_.sort_values("issue_local").groupby(["time_local", "turbine"], as_index=False).last())
    if period:
        latest = latest[(latest.time_local >= period[0]) & (latest.time_local < period[1])]
    wide = latest.pivot(index="time_local", columns="turbine", values="p50")
    cols = [c for c in wide.columns if c != TOTAL] + [TOTAL]
    wide = wide[cols].round(4)
    wide.to_csv(out_dir / "forecast_hourly_p50.csv")
    latest.to_csv(out_dir / "forecast_hourly_latest.csv", index=False)
    return wide
