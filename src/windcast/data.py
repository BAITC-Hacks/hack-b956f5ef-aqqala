"""Исторические данные SCADA: загрузка, перевод в UTC, агрегация по часам, чистка."""
import numpy as np
import pandas as pd

from .config import LOCAL_OFFSET
from .turbines import Turbine

COLUMNS = ["id", "time", "ws", "power", "temp"]


def load_scada(turbine: Turbine) -> pd.DataFrame:
    """10-минутные данные, индекс — наивное UTC."""
    df = pd.read_csv(turbine.history_path)
    df.columns = COLUMNS
    df["time"] = pd.to_datetime(df["time"]) - LOCAL_OFFSET
    return df.drop(columns="id").set_index("time").sort_index()


def to_hourly(df: pd.DataFrame, min_points: int = 4) -> pd.DataFrame:
    """Среднее за час [HH:00, HH:50]; час без достаточного числа точек отбрасывается."""
    g = df.resample("h")
    out = g.mean()
    out["n"] = g["power"].count()
    return out[out["n"] >= min_points].drop(columns="n")


def flag_anomalies(h: pd.DataFrame) -> pd.Series:
    """Простои и ограничения мощности: мощность сильно ниже кривой по измеренному ветру."""
    bins = np.arange(0, 30.5, 0.5)
    idx = np.digitize(h["ws"], bins)
    curve = h.groupby(idx)["power"].median()
    expected = pd.Series(idx, index=h.index).map(curve)
    return (expected > 0.15) & (h["power"] < 0.5 * expected - 0.05)


def load_hourly(turbine: Turbine) -> pd.DataFrame:
    """Часовые данные турбины с флагом аномалии (`bad`)."""
    h = to_hourly(load_scada(turbine))
    h["bad"] = flag_anomalies(h)
    return h
