"""Метрики в долях номинала: NMAE, NRMSE, смещение."""
import numpy as np
import pandas as pd


def scores(y_true, y_pred) -> dict:
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    m = ~np.isnan(e)
    e = e[m]
    return {"NMAE": float(np.mean(np.abs(e))), "NRMSE": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)), "n": int(m.sum())}


def coverage(y, lo, hi) -> float:
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    m = ~np.isnan(y)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))


def score_table(df: pd.DataFrame, methods: list[str], by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for mth in methods:
            s = scores(g["actual"], g[mth])
            rows.append({**dict(zip(by, key)), "method": mth, **s})
    return pd.DataFrame(rows)
