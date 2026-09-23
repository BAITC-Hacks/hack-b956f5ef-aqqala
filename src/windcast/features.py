"""Признаки для модели из прогнозов погоды (таблица `weather.asof` / `weather.lead_table`)."""
import numpy as np
import pandas as pd

from .config import LOCAL_OFFSET, NWP_MODELS

R_DRY = 287.05
RHO0 = 1.225


def _cols(w, var):
    return [f"{m}_{var}" for m in NWP_MODELS if f"{m}_{var}" in w.columns]


def _mean(w, var):
    cols = _cols(w, var)
    return w[cols].mean(axis=1) if cols else pd.Series(np.nan, index=w.index)


def build_features(w: pd.DataFrame, cap: float) -> pd.DataFrame:
    f = pd.DataFrame(index=w.index)
    ws_cols = _cols(w, "wind_speed_100m")
    for c in ws_cols:
        f[c.replace("_wind_speed_100m", "_ws100")] = w[c]
    ws = w[ws_cols]
    f["ws_mean"] = ws.mean(axis=1)
    f["ws_std"] = ws.std(axis=1)
    f["ws_min"] = ws.min(axis=1)
    f["ws_max"] = ws.max(axis=1)

    ws80, ws120 = _mean(w, "wind_speed_80m"), _mean(w, "wind_speed_120m")
    f["shear"] = (np.log(ws120.clip(lower=0.5) / ws80.clip(lower=0.5)) / np.log(1.5)).clip(-1, 1)

    d = np.deg2rad(w[_cols(w, "wind_direction_100m")])
    f["dir_sin"] = np.sin(d).mean(axis=1)
    f["dir_cos"] = np.cos(d).mean(axis=1)

    t, p, rh = _mean(w, "temperature_2m"), _mean(w, "surface_pressure"), _mean(w, "relative_humidity_2m")
    f["temp"] = t
    f["rh"] = rh
    rho = (p * 100 / (R_DRY * (t + 273.15))).fillna(RHO0)
    f["rho"] = rho
    # эквивалентная скорость с поправкой на плотность воздуха (IEC 61400-12)
    f["ws_eq"] = f["ws_mean"] * (rho / RHO0) ** (1 / 3)
    f["gust_ratio"] = (_mean(w, "wind_gusts_10m") / _mean(w, "wind_speed_10m").clip(lower=0.5)).clip(0, 5)
    f["icing_risk"] = ((t > -6) & (t < 1.5) & (rh >= 90)).astype(float)

    local = w.index + LOCAL_OFFSET
    f["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    f["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    f["lead_day"] = w["lead_day"].astype(float)
    f["cap"] = cap
    return f
