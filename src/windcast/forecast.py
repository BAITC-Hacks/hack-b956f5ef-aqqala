"""Прогноз на 48 часов от момента выпуска T: по турбинам и итог по ВЭС."""
import pandas as pd

from .config import HORIZON_H, LOCAL_OFFSET, to_utc
from .features import build_features
from .model import WindModel
from .turbines import Turbine, load_turbines
from .weather import asof

TOTAL = "ВЭС"


def forecast(issue_local, model: WindModel, turbines: list[Turbine] | None = None,
             horizon: int = HORIZON_H, nwp_models=None) -> pd.DataFrame:
    """Длинная таблица: time_local, turbine, p10/p50/p90 + диагностические поля.

    Итог по ВЭС — среднее долей турбин (турбины одинаковой мощности).
    """
    issue_utc = to_utc(issue_local)
    parts = []
    for t in turbines or load_turbines():
        w = asof(t.id, issue_utc, horizon, models=nwp_models)
        X = build_features(w, t.cap)
        p = model.predict(X, t.cap)
        p["turbine"] = t.id
        p["hours_ahead"] = w["hours_ahead"].values
        p["lead_day"] = w["lead_day"].values
        p["ws_mean"] = X["ws_mean"].values
        p["ws_spread"] = (X["ws_max"] - X["ws_min"]).values
        p["temp"] = X["temp"].values
        p["icing_risk"] = X["icing_risk"].values
        parts.append(p)
    df = pd.concat(parts)
    total = df.groupby(level=0).agg({"p10": "mean", "p50": "mean", "p90": "mean", "curve": "mean",
                                     "hours_ahead": "first", "lead_day": "first", "ws_mean": "mean",
                                     "ws_spread": "mean", "temp": "mean", "icing_risk": "max"})
    total["turbine"] = TOTAL
    df = pd.concat([df, total])
    df.index.name = "time_utc"
    df = df.reset_index()
    df.insert(0, "time_local", df["time_utc"] + LOCAL_OFFSET)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Широкая таблица для выгрузки: одна строка на час, колонки <турбина>_p50 и т.д."""
    w = df.pivot(index="time_local", columns="turbine", values=["p50", "p10", "p90"])
    w.columns = [f"{t}_{q}" for q, t in w.columns]
    order = [c for t in df["turbine"].unique() for c in (f"{t}_p50", f"{t}_p10", f"{t}_p90")]
    return w[order].reset_index()
