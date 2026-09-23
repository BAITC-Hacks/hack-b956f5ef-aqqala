"""Модель: эмпирическая кривая мощности + градиентный бустинг с квантилями P10/P50/P90."""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from .config import LEAD_DAYS, MODELS_DIR, NWP_MODELS, TRAIN_START
from .data import load_hourly
from .features import build_features
from .turbines import Turbine
from .weather import lead_table

QUANTILES = (0.1, 0.5, 0.9)
MODEL_FILE = MODELS_DIR / "windcast.joblib"


def training_frame(turbines: list[Turbine], start=TRAIN_START, end=None) -> pd.DataFrame:
    """Строки (час, заблаговременность) для всех турбин с историей; аномальные часы исключены."""
    rows = []
    for t in turbines:
        if not t.history_path:
            continue
        h = load_hourly(t)
        h = h.loc[~h["bad"], ["power"]]
        for lead in LEAD_DAYS:
            w = lead_table(t.id, lead)
            w = w[(w.index >= pd.Timestamp(start)) & (w.index < pd.Timestamp(end) if end else True)]
            w = w.assign(lead_day=lead)
            X = build_features(w, t.cap).join(h, how="inner")
            X["turbine"] = t.id
            rows.append(X)
    df = pd.concat(rows)
    return df.dropna(subset=["ws_mean", "power"])


@dataclass
class WindModel:
    curve: IsotonicRegression | None = None
    regs: dict = field(default_factory=dict)
    features: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    scale: tuple = (1.0, 1.0)

    def _with_curve(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["pc_ens"] = self.curve.predict(X["ws_eq"].fillna(X["ws_mean"]))
        for m in NWP_MODELS:
            c = f"{m}_ws100"
            X[f"pc_{m}"] = self.curve.predict(X[c].fillna(X["ws_mean"])) if c in X else np.nan
        return X

    def fit(self, df: pd.DataFrame, half_life_days: float | None = 365, calib_days: int | None = 60) -> "WindModel":
        """half_life_days — вес свежих данных: наблюдение возрастом half_life весит вдвое меньше.
        calib_days — последние N дней откладываются, чтобы откалибровать ширину интервала P10–P90
        (конформная поправка), затем модель переобучается на всех данных."""
        self.scale = (1.0, 1.0)
        if calib_days:
            split = df.index.max() - pd.Timedelta(days=calib_days)
            probe = WindModel()._fit_core(df[df.index <= split], half_life_days)
            cal = df[df.index > split]
            p = probe.predict(cal.drop(columns=["power", "turbine"]), cal["cap"].values)
            y = cal["power"].values
            k_lo = np.quantile((p.p50 - y) / np.maximum(p.p50 - p.p10, 0.01), 0.9)
            k_hi = np.quantile((y - p.p50) / np.maximum(p.p90 - p.p50, 0.01), 0.9)
            self.scale = (float(max(k_lo, 1.0)), float(max(k_hi, 1.0)))
        self._fit_core(df, half_life_days)
        self.meta |= {"half_life_days": half_life_days, "calib_days": calib_days, "interval_scale": self.scale}
        return self

    def _fit_core(self, df: pd.DataFrame, half_life_days) -> "WindModel":
        w = None
        if half_life_days:
            age = (df.index.max() - df.index) / pd.Timedelta(days=1)
            w = 0.5 ** (np.asarray(age) / half_life_days)
        self.curve = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
        self.curve.fit(df["ws_eq"].fillna(df["ws_mean"]), df["power"], sample_weight=w)
        X = self._with_curve(df.drop(columns=["power", "turbine"]))
        self.features = list(X.columns)
        for q in QUANTILES:
            reg = HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05, max_iter=500,
                                                max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
                                                random_state=42)
            self.regs[q] = reg.fit(X[self.features], df["power"], sample_weight=w)
        self.meta |= {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_from": str(df.index.min()), "train_to": str(df.index.max()), "rows": len(df),
            "turbines": sorted(df["turbine"].unique()),
        }
        return self

    def predict(self, X: pd.DataFrame, cap) -> pd.DataFrame:
        Xc = self._with_curve(X)[self.features]
        preds = np.column_stack([self.regs[q].predict(Xc) for q in QUANTILES])
        preds = np.sort(preds, axis=1)  # квантили не пересекаются
        k_lo, k_hi = getattr(self, "scale", (1.0, 1.0))
        preds[:, 0] = preds[:, 1] - k_lo * (preds[:, 1] - preds[:, 0])
        preds[:, 2] = preds[:, 1] + k_hi * (preds[:, 2] - preds[:, 1])
        preds = np.clip(preds, 0, np.asarray(cap, dtype=float).reshape(-1, 1) if np.ndim(cap) else cap)
        out = pd.DataFrame(preds, index=X.index, columns=["p10", "p50", "p90"])
        out["curve"] = np.minimum(Xc["pc_ens"].values, cap)
        return out

    def save(self, path=MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=MODEL_FILE) -> "WindModel":
        return joblib.load(path)
