"""Архивные прогнозы погоды Open-Meteo (Previous Runs API) и выборка «как на момент T».

Колонка `<var>_d<N>` в кеше — значение, спрогнозированное за N суток до целевого времени.
Для прогноза, выпущенного в момент T, на час t используется только лид N, при котором
прогон гарантированно существовал к T:  t + 1ч − 24·N ≤ T − PUBLICATION_DELAY_H.
"""
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from .config import (HORIZON_H, LEAD_DAYS, NWP_MODELS, NWP_VARS, PREVIOUS_RUNS_URL,
                     PUBLICATION_DELAY_H, WEATHER_DIR)
from .turbines import Turbine

MANIFEST = WEATHER_DIR / "manifest.json"


def cache_path(tid: str, model: str):
    return WEATHER_DIR / f"{tid}_{model}.parquet"


# ---------------------------------------------------------------- загрузка

def _request(turbine: Turbine, api_model: str, start: str, end: str, retries: int = 4) -> pd.DataFrame:
    cols = [f"{v}_previous_day{n}" for v in NWP_VARS for n in LEAD_DAYS]
    params = dict(latitude=turbine.lat, longitude=turbine.lon, timezone="GMT", wind_speed_unit="ms",
                  models=api_model, start_date=start, end_date=end, hourly=",".join(cols))
    for attempt in range(retries):
        try:
            r = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=120)
            if r.status_code == 429:
                raise requests.HTTPError("429 rate limit")
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame(h)
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
            df.columns = [c.replace("_previous_day", "_d") for c in df.columns]
            return df.astype("float32")
        except (requests.RequestException, KeyError, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def download(turbine: Turbine, start: str, end: str, models=None, chunk_days: int = 180, log=print) -> None:
    """Скачать прогнозы погоды для турбины и слить с кешем."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    for model in models or NWP_MODELS:
        parts = []
        for s in pd.date_range(start, end, freq=f"{chunk_days}D"):
            e = min(s + pd.Timedelta(days=chunk_days - 1), pd.Timestamp(end))
            log(f"  {turbine.id} {model}: {s.date()} … {e.date()}")
            parts.append(_request(turbine, NWP_MODELS[model], str(s.date()), str(e.date())))
        new = pd.concat(parts)
        path = cache_path(turbine.id, model)
        if path.exists():
            new = new.combine_first(pd.read_parquet(path))
        new = new[~new.index.duplicated(keep="first")].sort_index()
        new.to_parquet(path)
        _update_manifest(turbine.id, model, path, new)
    load_cache.cache_clear()


def _update_manifest(tid, model, path, df):
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m[f"{tid}_{model}"] = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(df),
        "from": str(df.index.min()),
        "to": str(df.index.max()),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": PREVIOUS_RUNS_URL,
    }
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False))


def ensure_weather(turbine: Turbine, start, end, log=print) -> bool:
    """Докачать погоду, если кеш не покрывает [start, end]. Возвращает True, если была загрузка."""
    start, end = pd.Timestamp(start).floor("D"), pd.Timestamp(end).ceil("D")
    missing = [m for m in NWP_MODELS if not _covers(turbine.id, m, start, end)]
    if missing:
        download(turbine, str(start.date()), str(end.date()), models=missing, log=log)
    return bool(missing)


def _covers(tid, model, start, end) -> bool:
    df = load_cache(tid, model)
    return df is not None and df.index.min() <= start and df.index.max() >= end - pd.Timedelta(hours=1)


# ---------------------------------------------------------------- чтение

@lru_cache(maxsize=64)
def load_cache(tid: str, model: str) -> pd.DataFrame | None:
    path = cache_path(tid, model)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # переменные, которых у модели нет (например, ветер 80/120 м у ECMWF), приходят пустыми
    return df.dropna(axis=1, how="all")


def lead_table(tid: str, lead: int, models=None) -> pd.DataFrame:
    """Признаки на час t (среднее за [t, t+1ч]) из прогноза с заблаговременностью `lead` суток.

    Колонки: `<model>_<var>`. Индекс — наивное UTC.
    """
    frames = []
    for model in models or NWP_MODELS:
        df = load_cache(tid, model)
        if df is None:
            continue
        cols = [c for c in df.columns if c.endswith(f"_d{lead}")]
        sub = df[cols].rename(columns=lambda c: f"{model}_{c.rsplit('_d', 1)[0]}")
        sub = _hour_mean(sub)
        frames.append(sub)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _hour_mean(sub: pd.DataFrame) -> pd.DataFrame:
    """Мгновенные значения в HH:00 → среднее по часу (HH:00 и HH+1:00); направление — векторно."""
    nxt = sub.shift(-1, freq="h").reindex(sub.index)
    out = (sub + nxt) / 2
    for c in [c for c in sub.columns if "direction" in c]:
        a, b = np.deg2rad(sub[c]), np.deg2rad(nxt[c])
        out[c] = np.rad2deg(np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))) % 360
    return out


def lead_for(hours_ahead) -> np.ndarray:
    """Минимальная допустимая заблаговременность (сутки) для часа, начинающегося через h ч после T."""
    h = np.asarray(hours_ahead, dtype=float)
    return np.ceil((h + 1 + PUBLICATION_DELAY_H) / 24).astype(int)


def target_hours(issue_utc, horizon: int = HORIZON_H) -> pd.DatetimeIndex:
    """Часы прогноза: начиная со следующего полного часа после T."""
    return pd.date_range(pd.Timestamp(issue_utc).floor("h") + pd.Timedelta(hours=1), periods=horizon, freq="h")


def asof(tid: str, issue_utc, horizon: int = HORIZON_H, models=None) -> pd.DataFrame:
    """Прогноз погоды на следующие `horizon` часов в том виде, в каком он был доступен в момент T."""
    issue = pd.Timestamp(issue_utc)
    targets = target_hours(issue, horizon)
    hours = (targets - issue) / pd.Timedelta(hours=1)
    leads = lead_for(hours)
    if leads.max() > max(LEAD_DAYS):
        raise ValueError("Горизонт больше доступной заблаговременности архива")
    rows = []
    for lead in np.unique(leads):
        mask = leads == lead
        tab = lead_table(tid, int(lead), models).reindex(targets[mask])
        rows.append(tab)
    out = pd.concat(rows).sort_index()
    out["lead_day"] = leads
    out["hours_ahead"] = hours.values
    return out


def fingerprint(df: pd.DataFrame) -> str:
    """Хеш входных данных — агент сравнивает его, чтобы решить о перерасчёте."""
    return hashlib.sha256(pd.util.hash_pandas_object(df.round(3), index=True).values.tobytes()).hexdigest()[:16]
