"""Честная проверка на прошлых периодах: модель обучается только на данных до первого выпуска,
затем прогноз выпускается каждый день, как в тесте, и сравнивается с фактом и базовыми моделями."""
import pandas as pd

from .config import OUTPUTS_DIR, to_utc
from .data import load_hourly
from .forecast import TOTAL, forecast
from .metrics import coverage, score_table
from .model import WindModel, training_frame
from .turbines import load_turbines

FOLDS = [
    ("Февраль 2025", "2025-01-31 10:00", "2025-02-27 10:00"),
    ("Январь 2026", "2025-12-31 10:00", "2026-01-29 10:00"),
]
METHODS = {"p50": "Модель (P50)", "curve": "Прогноз погоды → кривая мощности",
           "clim": "Климатология (месяц × час)", "persist": "Персистентность (последний час)"}


def _actuals(turbines):
    act = {t.id: load_hourly(t)["power"] for t in turbines}
    a = pd.DataFrame(act)
    a[TOTAL] = a.mean(axis=1, skipna=False)
    return a


def run_fold(name, first_issue, last_issue, log=print) -> pd.DataFrame:
    turbines = [t for t in load_turbines() if t.history_path]
    cutoff = to_utc(first_issue)
    log(f"[{name}] обучение на данных до {first_issue} (UTC+6)")
    model = WindModel().fit(training_frame(turbines, end=cutoff))
    act = _actuals(turbines)
    hist = act[act.index < cutoff]
    local = hist.index + pd.Timedelta(hours=6)
    clim = hist.groupby([local.month, local.hour]).mean()

    rows = []
    for issue in pd.date_range(first_issue, last_issue, freq="D"):
        fc = forecast(issue, model, turbines)
        known = act[act.index < to_utc(issue).floor("h")]
        last = known.ffill().iloc[-1]
        fc["persist"] = fc["turbine"].map(last)
        fc["clim"] = [clim.loc[(t.month, t.hour), tb] if (t.month, t.hour) in clim.index else float("nan")
                      for t, tb in zip(fc["time_local"], fc["turbine"])]
        fc["actual"] = [act.at[t, tb] if t in act.index else float("nan")
                        for t, tb in zip(fc["time_utc"], fc["turbine"])]
        fc["issue_local"] = issue
        rows.append(fc)
    df = pd.concat(rows)
    df["fold"] = name
    df["horizon"] = df["hours_ahead"].map(lambda h: "D+1 (1–24 ч)" if h <= 24 else "D+2 (25–48 ч)")
    return df.dropna(subset=["actual"])


def run_all(log=print) -> str:
    out = OUTPUTS_DIR / "validation"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([run_fold(*f, log=log) for f in FOLDS])
    df.to_csv(out / "validation_forecasts.csv", index=False)

    tab = score_table(df, list(METHODS), ["fold", "turbine", "horizon"])
    tab.to_csv(out / "validation_scores.csv", index=False)

    lines = ["# Валидация на прошлых периодах", "",
             "Схема: модель обучается только на данных до первого момента выпуска; прогноз выпускается",
             "ежедневно в 10:00 (UTC+6) на 48 ч вперёд с погодой «как на момент выпуска». Метрики — в долях",
             "номинальной мощности (0.10 = 10% номинала). Персистентность использует фактическую мощность",
             "до момента выпуска, которой в тесте (февраль 2026) нет.", ""]
    for fold, g in df.groupby("fold", sort=False):
        lines += [f"## {fold}", ""]
        t = tab[(tab["fold"] == fold) & (tab["turbine"] == TOTAL)]
        piv = t.pivot(index="method", columns="horizon", values="NMAE").reindex(list(METHODS))
        piv.index = [METHODS[m] for m in piv.index]
        lines += ["**NMAE, итог по ВЭС**", "", piv.round(3).to_markdown(), ""]
        t2 = tab[(tab["fold"] == fold) & (tab["method"] == "p50")]
        piv2 = t2.pivot(index="turbine", columns="horizon", values=["NMAE", "NRMSE", "bias"]).round(3)
        piv2.columns = [f"{a} {b}" for a, b in piv2.columns]
        lines += ["**Модель (P50) по турбинам**", "", piv2.to_markdown(), ""]
        cov = coverage(g["actual"], g["p10"], g["p90"])
        lines += [f"Доля фактов внутри интервала P10–P90: **{cov:.0%}** (цель ≈ 80%).", ""]
    text = "\n".join(lines)
    (out / "validation.md").write_text(text, encoding="utf-8")
    return text
