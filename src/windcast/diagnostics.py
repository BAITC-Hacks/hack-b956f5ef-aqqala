"""Проверки данных, на которых основаны решения: часовой пояс SCADA, точность моделей погоды, потолок мощности."""
import numpy as np
import pandas as pd

from .config import LEAD_DAYS, NWP_MODELS, OUTPUTS_DIR
from .data import COLUMNS, load_hourly
from .turbines import load_turbines
from .weather import lead_table


def _raw(t):
    df = pd.read_csv(t.history_path)
    df.columns = COLUMNS
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def timezone_offset(t, var="ws", nwp_var="wind_speed_100m", minutes=range(240, 421, 10)) -> dict:
    """Сдвиг (мин) между метками SCADA и UTC, при котором ряд лучше всего совпадает с прогнозом погоды (лид 1)."""
    raw = _raw(t)[var].rolling(7, center=True, min_periods=4).mean()
    out = {}
    for m in NWP_MODELS:
        n = lead_table(t.id, 1, models=[m])
        col = f"{m}_{nwp_var}"
        if col not in n:
            continue
        n = n[col].dropna().resample("10min").interpolate()
        if var == "temp":  # сравниваем суточный ход, без сезонного уровня
            n = n - n.rolling("1D", center=True).mean()
            r = raw - raw.rolling("1D", center=True).mean()
        else:
            r = raw
        corr = {}
        for mm in minutes:
            s = r.copy()
            s.index = s.index - pd.Timedelta(minutes=mm)
            j = pd.concat([s, n], axis=1, join="inner").dropna()
            corr[mm] = j.iloc[:, 0].corr(j.iloc[:, 1])
        best = max(corr, key=corr.get)
        out[m] = {"best": f"UTC+{best // 60}:{best % 60:02d}", "corr": round(corr[best], 3),
                  "corr_utc5": round(corr.get(300, float("nan")), 3), "corr_utc6": round(corr.get(360, float("nan")), 3)}
    return out


def diurnal_phase(raw: pd.DataFrame) -> pd.DataFrame:
    """Время суточного максимума температуры (первая гармоника) по годовым периодам март–февраль."""
    rows = []
    for y in range(raw.index.min().year, raw.index.max().year + 1):
        s = raw.loc[f"{y}-03-01":f"{y + 1}-02-28", "temp"]
        if len(s) < 24 * 6 * 180:
            continue
        anom = s - s.rolling("1D", center=True).mean()
        hod = anom.groupby(s.index.hour + s.index.minute / 60).mean()
        ang = 2 * np.pi * hod.index / 24
        c, sn = (hod * np.cos(ang)).sum(), (hod * np.sin(ang)).sum()
        peak = (np.arctan2(sn, c) % (2 * np.pi)) * 24 / (2 * np.pi)
        rows.append({"период": f"03.{y}–02.{y + 1}", "максимум, ч:мин": f"{int(peak):02d}:{int(peak % 1 * 60):02d}"})
    return pd.DataFrame(rows)


def nwp_skill(t) -> pd.DataFrame:
    """Корреляция прогнозного ветра 100 м с измеренным (часовые, без аномалий) по моделям и заблаговременности."""
    h = load_hourly(t)
    h = h[~h["bad"]]
    rows = []
    for lead in LEAD_DAYS:
        w = lead_table(t.id, lead).join(h[["ws"]], how="inner").dropna(subset=["ws"])
        cols = [f"{m}_wind_speed_100m" for m in NWP_MODELS if f"{m}_wind_speed_100m" in w]
        row = {"лид, сут": lead}
        for c in cols:
            row[c.split("_")[0]] = round(w[c].corr(w["ws"]), 3)
        row["среднее 3 моделей"] = round(w[cols].mean(axis=1).corr(w["ws"]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def run_all(log=print) -> str:
    lines = ["# Диагностика данных", ""]
    for t in [x for x in load_turbines() if x.history_path]:
        log(f"  {t.id}…")
        h = load_hourly(t)
        raw = _raw(t)
        lines += [f"## {t.id}", "",
                  f"- Период: {raw.index.min()} — {raw.index.max()} (метки SCADA), строк: {len(raw)}",
                  f"- Плато кривой мощности (медиана 10-мин мощности при ветре > 13 м/с) — потолок `cap` "
                  f"в turbines.yaml: {raw.loc[raw['ws'] > 13, 'power'].median():.2f}",
                  f"- Доля часов, помеченных как простой/ограничение: {h['bad'].mean():.2%}", ""]
        for var, nv, name in [("ws", "wind_speed_100m", "ветер 100 м"), ("temp", "temperature_2m", "суточный ход температуры")]:
            tz = timezone_offset(t, var, nv)
            lines += [f"**Сдвиг меток SCADA относительно UTC — {name}**", "",
                      pd.DataFrame(tz).T.to_markdown(), ""]
        lines += ["**Фаза суточного хода температуры по годам (время максимума первой гармоники, метки SCADA)**",
                  "", "Если бы часы SCADA перевели 01.03.2024 на UTC+5, фаза сместилась бы на ~1 ч.", "",
                  diurnal_phase(raw).to_markdown(index=False), ""]
        lines += ["**Точность прогноза ветра 100 м (корреляция с измеренным, часовые)**", "",
                  nwp_skill(t).to_markdown(index=False), ""]
    text = "\n".join(lines)
    out = OUTPUTS_DIR / "diagnostics.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
