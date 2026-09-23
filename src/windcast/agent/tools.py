"""Инструменты агента. Каждый инструмент — метод `Toolbox`, возвращает JSON-совместимый dict.

Числа считает только код; оркестратор (правила или LLM) решает, какие инструменты вызвать,
в каком порядке и что делать с результатом.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import HORIZON_H, LOCAL_OFFSET, LOCAL_TZ_LABEL, NWP_MODELS, OUTPUTS_DIR, to_utc
from ..data import load_hourly
from ..forecast import TOTAL, forecast
from ..metrics import scores
from ..model import MODEL_FILE, WindModel
from ..turbines import load_turbines
from ..weather import asof, ensure_weather, fingerprint

# Пороги анализа (доли номинала / м/с), выбраны по распределениям на валидации:
# ширина P10–P90 — верхний квартиль (≈0.8), ошибка там в 3.5 раза выше, чем в нижнем квартиле;
# разброс ветра между моделями — верхние 10% (≈7 м/с).
WIDE_INTERVAL = 0.8
CONF_HIGH, CONF_MID = 0.4, 0.65  # средняя ширина P10–P90 за 48 ч
DISAGREE_WS = 7.0
RAMP = 0.25
MAX_MISSING = 0.25
CHANGE_NOTABLE = 0.03


def issue_dir(runs_dir: Path, issue_local: pd.Timestamp) -> Path:
    return runs_dir / pd.Timestamp(issue_local).strftime("%Y%m%d_%H%M")


@dataclass
class Toolbox:
    issue_local: pd.Timestamp
    runs_dir: Path = OUTPUTS_DIR / "runs"
    online: bool = True
    model: WindModel | None = None
    turbines: list = field(default_factory=load_turbines)
    # состояние между вызовами
    nwp_models: list = field(default_factory=lambda: list(NWP_MODELS))
    result: pd.DataFrame | None = None
    input_hash: str | None = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.issue_local = pd.Timestamp(self.issue_local)
        self.issue_utc = to_utc(self.issue_local)
        self.dir = issue_dir(self.runs_dir, self.issue_local)
        if self.model is None:
            self.model = WindModel.load(MODEL_FILE)

    # ------------------------------------------------------------ погода
    def check_weather(self) -> dict:
        """Есть ли в кеше прогнозы погоды на 48 ч от момента выпуска, по турбинам и моделям."""
        report = {}
        for t in self.turbines:
            try:
                w = asof(t.id, self.issue_utc)
            except Exception as e:  # кеша нет совсем
                report[t.id] = {"error": str(e)}
                continue
            report[t.id] = {m: round(float(w[f"{m}_wind_speed_100m"].isna().mean()), 3)
                            if f"{m}_wind_speed_100m" in w else 1.0 for m in NWP_MODELS}
        missing = sorted({f"{tid}:{m}" for tid, r in report.items() for m, v in r.items()
                          if m == "error" or v > 0})
        return {"missing_share": report, "incomplete": missing}

    def fetch_weather(self, turbine_ids: list | None = None) -> dict:
        """Докачать прогнозы погоды из Open-Meteo для окна выпуска (нужна сеть)."""
        if not self.online:
            return {"ok": False, "reason": "офлайн-режим: загрузка запрещена"}
        ids = turbine_ids or [t.id for t in self.turbines]
        start = self.issue_utc - pd.Timedelta(days=4)
        end = self.issue_utc + pd.Timedelta(hours=HORIZON_H + 24)
        done, errors = [], {}
        for t in self.turbines:
            if t.id not in ids:
                continue
            try:
                if ensure_weather(t, start, end, log=lambda *_: None):
                    done.append(t.id)
            except Exception as e:
                errors[t.id] = str(e)[:200]
        return {"ok": not errors, "downloaded": done, "errors": errors}

    def validate_inputs(self) -> dict:
        """Качество прогнозов погоды: пропуски, физически невозможные значения, расхождение моделей."""
        per_model, spread = {}, []
        for t in self.turbines:
            w = asof(t.id, self.issue_utc)
            cols = {m: f"{m}_wind_speed_100m" for m in NWP_MODELS if f"{m}_wind_speed_100m" in w}
            ws = w[list(cols.values())]
            spread.append(ws.max(axis=1) - ws.min(axis=1))
            for m, c in cols.items():
                s = w[c]
                bad = ((s < 0) | (s > 50)).mean()
                dev = (s - ws.drop(columns=c).mean(axis=1)).abs().mean()
                r = per_model.setdefault(m, {"missing": 0.0, "impossible": 0.0, "mean_abs_dev_ms": 0.0})
                r["missing"] = max(r["missing"], round(float(s.isna().mean()), 3))
                r["impossible"] = max(r["impossible"], round(float(bad), 3))
                r["mean_abs_dev_ms"] = max(r["mean_abs_dev_ms"], round(float(dev), 2))
        for m in NWP_MODELS:
            per_model.setdefault(m, {"missing": 1.0, "impossible": 0.0, "mean_abs_dev_ms": 0.0})
        usable = [m for m, r in per_model.items() if r["missing"] <= MAX_MISSING and r["impossible"] == 0]
        sp = pd.concat(spread)
        return {"per_model": per_model, "usable_models": usable,
                "ws_spread_ms": {"mean": round(float(sp.mean()), 2), "max": round(float(sp.max()), 2)}}

    # ------------------------------------------------------------ прогноз
    def run_forecast(self, nwp_models: list | None = None) -> dict:
        """Запустить модель. nwp_models — какие модели погоды использовать (по умолчанию все пригодные)."""
        if nwp_models:
            self.nwp_models = [m for m in nwp_models if m in NWP_MODELS]
        self.result = forecast(self.issue_local, self.model, self.turbines, nwp_models=self.nwp_models)
        self.input_hash = self._input_hash()
        return self._summary()

    def check_previous_version(self) -> dict:
        """Есть ли уже прогноз для этого момента выпуска и изменились ли с тех пор входные данные."""
        h = self._input_hash()
        meta_file = self.dir / "meta.json"
        if not meta_file.exists():
            return {"exists": False, "input_hash": h}
        old = json.loads(meta_file.read_text())
        return {"exists": True, "inputs_changed": old.get("input_hash") != h,
                "old_hash": old.get("input_hash"), "new_hash": h}

    def _input_hash(self) -> str:
        parts = [asof(t.id, self.issue_utc, models=self.nwp_models) for t in self.turbines]
        h = fingerprint(pd.concat(parts, keys=[t.id for t in self.turbines]))
        return f"{h}-{self.model.meta.get('trained_at', '')}"

    def _summary(self) -> dict:
        df = self.result
        out = {"issue_local": str(self.issue_local), "hours": HORIZON_H, "nwp_models": self.nwp_models}
        for tb, g in df.groupby("turbine", sort=False):
            out[tb] = {"mean_p50": round(float(g.p50.mean()), 3), "min_p50": round(float(g.p50.min()), 3),
                       "max_p50": round(float(g.p50.max()), 3),
                       "energy_frac_hours": round(float(g.p50.sum()), 1)}
        return out

    def analyze_forecast(self) -> dict:
        """Проверка результата: правдоподобность, широкие интервалы, расхождение погоды, обледенение, рампы."""
        df = self.result
        tot = df[df.turbine == TOTAL].set_index("time_local")
        per = df[df.turbine != TOTAL]
        sane = bool(df[["p10", "p50", "p90"]].notna().all().all()
                    and (df.p10 <= df.p50 + 1e-9).all() and (df.p50 <= df.p90 + 1e-9).all()
                    and df.p50.between(0, 1).all())
        wide = tot.index[(tot.p90 - tot.p10) > WIDE_INTERVAL]
        disagree = tot.index[tot.ws_spread > DISAGREE_WS]
        icing = tot.index[tot.icing_risk > 0]
        ramps = tot.p50.diff().abs()
        ramp_hours = ramps.index[ramps >= RAMP]
        width = float((tot.p90 - tot.p10).mean())
        conf = "высокая" if width < CONF_HIGH else "средняя" if width < CONF_MID else "низкая"
        return {
            "sane": sane, "confidence": conf, "mean_interval_width": round(width, 3),
            "wide_interval_hours": _ranges(wide), "nwp_disagreement_hours": _ranges(disagree),
            "icing_risk_hours": _ranges(icing), "ramp_hours": _ranges(ramp_hours),
            "turbine_mean_p50": {tb: round(float(g.p50.mean()), 3) for tb, g in per.groupby("turbine")},
            "total_profile_6h": {str(k): round(float(v), 2)
                                 for k, v in tot.p50.resample("6h", origin="start").mean().items()},
        }

    def compare_with_previous(self) -> dict:
        """Сравнить с последним ранее выпущенным прогнозом (общие часы) и с прошлой версией этого же выпуска."""
        out = {"same_issue": None, "previous_issue": None}
        meta_file = self.dir / "meta.json"
        if meta_file.exists():
            old = json.loads(meta_file.read_text())
            changed = old.get("input_hash") != self.input_hash
            out["same_issue"] = {"exists": True, "inputs_changed": changed, "old_hash": old.get("input_hash"),
                                 "new_hash": self.input_hash}
            if changed and (self.dir / "forecast.csv").exists():
                out["same_issue"]["mean_abs_change"] = self._diff(pd.read_csv(self.dir / "forecast.csv"))
        prev = sorted(p for p in self.runs_dir.glob("*_*") if p.is_dir() and p.name < self.dir.name)
        if prev and (prev[-1] / "forecast.csv").exists():
            out["previous_issue"] = {"issue": prev[-1].name,
                                     "mean_abs_change_overlap": self._diff(pd.read_csv(prev[-1] / "forecast.csv"))}
        return out

    def _diff(self, old: pd.DataFrame):
        old = old[old.turbine == TOTAL].assign(time_local=lambda d: pd.to_datetime(d.time_local))
        new = self.result[self.result.turbine == TOTAL]
        j = new.merge(old, on="time_local", suffixes=("", "_old"))
        return None if j.empty else round(float((j.p50 - j.p50_old).abs().mean()), 3)

    def evaluate_recent(self, days: int = 7) -> dict:
        """Ошибка прошлых выпусков за `days` суток, если факт уже известен (дрейф модели/погоды)."""
        act = {t.id: load_hourly(t)["power"] for t in self.turbines if t.history_path}
        if not act:
            return {"available": False, "reason": "нет турбин с историей"}
        a = pd.DataFrame(act)
        a[TOTAL] = a.mean(axis=1, skipna=False)
        known_until = min(a.index.max(), self.issue_utc)
        if known_until < self.issue_utc - pd.Timedelta(days=days):
            return {"available": False, "reason": f"факт за последние {days} сут. недоступен "
                                                  f"(исторические данные заканчиваются {known_until + LOCAL_OFFSET:%d.%m.%Y %H:%M})"}
        errs = []
        for p in sorted(self.runs_dir.glob("*_*")):
            f = p / "forecast.csv"
            if not f.exists() or p.name >= self.dir.name:
                continue
            fc = pd.read_csv(f, parse_dates=["time_utc"])
            fc = fc[(fc.turbine == TOTAL) & (fc.time_utc < known_until)
                    & (fc.time_utc >= known_until - pd.Timedelta(days=days))]
            fc["actual"] = fc.time_utc.map(a[TOTAL])
            errs.append(fc.dropna(subset=["actual"]))
        if not errs or sum(map(len, errs)) == 0:
            return {"available": False, "reason": f"нет факта для прошлых выпусков (факт известен до {known_until} UTC)"}
        e = pd.concat(errs)
        s = scores(e.actual, e.p50)
        return {"available": True, "hours": s["n"], "NMAE": round(s["NMAE"], 3), "bias": round(s["bias"], 3)}

    # ------------------------------------------------------------ выход
    def save(self, report_md: str, decisions: list) -> dict:
        """Сохранить прогноз (CSV), отчёт (MD) и метаданные; прошлую версию — в versions/."""
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / "forecast.csv"
        if f.exists() and (self.dir / "meta.json").exists():
            old = json.loads((self.dir / "meta.json").read_text())
            if old.get("input_hash") != self.input_hash:
                v = self.dir / "versions"
                v.mkdir(exist_ok=True)
                f.rename(v / f"forecast_{old.get('input_hash', 'old')[:8]}.csv")
        self.result.to_csv(f, index=False)
        (self.dir / "report.md").write_text(report_md, encoding="utf-8")
        meta = {"issue_local": str(self.issue_local), "tz": LOCAL_TZ_LABEL, "input_hash": self.input_hash,
                "nwp_models": self.nwp_models, "model": self.model.meta, "decisions": decisions}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        return {"dir": str(self.dir), "files": ["forecast.csv", "report.md", "meta.json", "agent_log.jsonl"]}


def _ranges(idx) -> list[str]:
    """Список часов → компактные интервалы «дд.мм чч:00–чч:00»."""
    idx = pd.DatetimeIndex(idx).sort_values()
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for t in list(idx[1:]) + [None]:
        if t is not None and t - prev == pd.Timedelta(hours=1):
            prev = t
            continue
        end = prev + pd.Timedelta(hours=1)
        out.append(f"{start:%d.%m %H:00}–{end:%H:00}" if end.date() == start.date()
                   else f"{start:%d.%m %H:00}–{end:%d.%m %H:00}")
        if t is not None:
            start = prev = t
    return out
