"""Streamlit-интерфейс: прогноз по дате и времени выпуска, результаты теста, валидация, турбины."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from windcast.agent import make_agent
from windcast.config import LOCAL_TZ_LABEL, OUTPUTS_DIR, WEATHER_START
from windcast.forecast import TOTAL, to_wide
from windcast.model import WindModel
from windcast.turbines import Turbine, add_turbine, load_turbines, remove_turbine

st.set_page_config(page_title="Прогноз выработки ВЭС", page_icon="🌬️", layout="wide")

COLORS = {"T1": "#2a78d6", "T2": "#e8742f", TOTAL: "#1b1b1b"}


@st.cache_resource
def get_model():
    return WindModel.load()


def forecast_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    tot = df[df.turbine == TOTAL]
    fig.add_trace(go.Scatter(x=tot.time_local, y=tot.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tot.time_local, y=tot.p10, fill="tonexty", fillcolor="rgba(120,120,120,0.18)",
                             line=dict(width=0), name=f"{TOTAL}: P10–P90"))
    for tb, g in df.groupby("turbine", sort=False):
        fig.add_trace(go.Scatter(x=g.time_local, y=g.p50, name=f"{tb} (P50)", mode="lines",
                                 line=dict(width=3 if tb == TOTAL else 1.5, color=COLORS.get(tb),
                                           dash="solid" if tb == TOTAL else "dot")))
    fig.update_layout(title=title, yaxis_title="Мощность, доля номинала", yaxis_range=[0, 1.02],
                      xaxis_title=f"Время ({LOCAL_TZ_LABEL})", hovermode="x unified", height=430,
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=50, l=10, r=10))
    return fig


st.title("🌬️ Агентный прогноз выработки ВЭС на 48 часов")
st.caption("Архивные прогнозы погоды Open-Meteo «как на момент выпуска» → ML-модель → агент проверяет, "
           f"пересчитывает и объясняет. Мощность — в долях номинала, время — {LOCAL_TZ_LABEL}.")

tab_fc, tab_test, tab_val, tab_tb = st.tabs(["Прогноз", "Тест: февраль 2026", "Валидация", "Турбины"])

# ------------------------------------------------------------------ прогноз
with tab_fc:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    d = c1.date_input("Дата выпуска", dt.date(2026, 2, 1), min_value=dt.date(2024, 3, 10))
    t = c2.time_input("Время выпуска", dt.time(10, 0), step=3600)
    mode = c3.selectbox("Агент", ["rules", "llm"], format_func=lambda m: {"rules": "Правила (без ключа)",
                                                                           "llm": "LLM"}[m])
    force = c4.checkbox("Пересчитать принудительно", False)
    if st.button("Сформировать прогноз", type="primary"):
        issue = pd.Timestamp(dt.datetime.combine(d, t))
        with st.status("Агент работает…", expanded=True) as status:
            try:
                res = make_agent(mode, issue, runs_dir=OUTPUTS_DIR / "runs", online=True,
                                 model=get_model(), force=force).run()
                st.session_state["res"] = res
                status.update(label=f"Готово: {res['status']}", state="complete")
            except Exception as e:  # показать ошибку пользователю, не роняя приложение
                status.update(label="Ошибка", state="error")
                st.exception(e)
    res = st.session_state.get("res")
    if res and res.get("result") is not None:
        df = res["result"].copy()
        df["time_local"] = pd.to_datetime(df["time_local"])
        issue_s = Path(res["dir"]).name
        if res["status"] == "unchanged":
            st.info("Входные данные не изменились — агент использовал сохранённый прогноз.")
        st.plotly_chart(forecast_chart(df, f"Выпуск {issue_s}"), use_container_width=True)
        wide = to_wide(df)
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Почасовой прогноз")
            st.dataframe(wide, height=360, hide_index=True)
            st.download_button("Скачать CSV", wide.to_csv(index=False).encode("utf-8"),
                               file_name=f"forecast_{issue_s}.csv", mime="text/csv")
        with right:
            rep = Path(res["dir"]) / "report.md"
            if rep.exists():
                st.markdown(rep.read_text(encoding="utf-8"))
        logf = Path(res["dir"]) / ("agent_log_reuse.jsonl" if res["status"] == "unchanged" else "agent_log.jsonl")
        if logf.exists():
            with st.expander("Журнал решений агента"):
                log = pd.read_json(logf, lines=True)
                st.dataframe(log[["step", "tool", "reason"]], hide_index=True, use_container_width=True)
                for _, r in log.iterrows():
                    st.markdown(f"**{r.step}. {r.tool}** — {r.reason}")
                    st.json(r.result if isinstance(r.result, (dict, list)) else {"result": r.result},
                            expanded=False)

# ------------------------------------------------------------------ тест
with tab_test:
    f = OUTPUTS_DIR / "feb2026" / "forecast_hourly_latest.csv"
    if f.exists():
        df = pd.read_csv(f, parse_dates=["time_local"])
        st.plotly_chart(forecast_chart(df, "Февраль 2026: для каждого часа — самый свежий выпуск"),
                        use_container_width=True)
        p50 = OUTPUTS_DIR / "feb2026" / "forecast_hourly_p50.csv"
        st.download_button("Скачать почасовой прогноз за февраль (P50)", p50.read_bytes(),
                           file_name="feb2026_forecast_p50.csv", mime="text/csv")
        runs = sorted((OUTPUTS_DIR / "feb2026" / "runs").glob("*_*"))
        pick = st.selectbox("Отчёт агента по выпуску", [r.name for r in runs], index=len(runs) - 1 if runs else 0)
        if pick:
            st.markdown((OUTPUTS_DIR / "feb2026" / "runs" / pick / "report.md").read_text(encoding="utf-8"))
    else:
        st.warning("Нет результатов бэктеста. Запустите: `uv run windcast backtest --hours 10:00,22:00 --offline`")

# ------------------------------------------------------------------ валидация
with tab_val:
    v = OUTPUTS_DIR / "validation" / "validation.md"
    if v.exists():
        st.markdown(v.read_text(encoding="utf-8"))
        vf = pd.read_csv(OUTPUTS_DIR / "validation" / "validation_forecasts.csv", parse_dates=["time_local"])
        fold = st.selectbox("Период", vf.fold.unique())
        g = vf[(vf.fold == fold) & (vf.turbine == TOTAL) & (vf.horizon.str.startswith("D+1"))].sort_values("time_local")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.time_local, y=g.actual, name="Факт", line=dict(color="#1b1b1b")))
        fig.add_trace(go.Scatter(x=g.time_local, y=g.p50, name="Прогноз P50 (D+1)", line=dict(color="#2a78d6")))
        fig.add_trace(go.Scatter(x=g.time_local, y=g.curve, name="Погода → кривая мощности",
                                 line=dict(color="#e8742f", dash="dot")))
        fig.update_layout(yaxis_title="Мощность ВЭС, доля номинала", hovermode="x unified", height=420,
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=30, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Нет результатов валидации. Запустите: `uv run windcast validate`")

# ------------------------------------------------------------------ турбины
with tab_tb:
    tbs = load_turbines()
    st.dataframe(pd.DataFrame([{"id": x.id, "lat": x.lat, "lon": x.lon, "потолок": x.cap,
                                "история": "есть" if x.history else "нет"} for x in tbs]), hide_index=True)
    st.map(pd.DataFrame({"lat": [x.lat for x in tbs], "lon": [x.lon for x in tbs]}), zoom=12)
    st.subheader("Добавить турбину")
    with st.form("add"):
        a1, a2, a3, a4 = st.columns(4)
        tid = a1.text_input("ID", "T3")
        lat = a2.number_input("Широта", value=43.6400, format="%.6f")
        lon = a3.number_input("Долгота", value=78.5450, format="%.6f")
        cap = a4.number_input("Потолок мощности", value=0.98, min_value=0.1, max_value=1.0)
        if st.form_submit_button("Добавить"):
            try:
                add_turbine(Turbine(id=tid.strip(), lat=lat, lon=lon, cap=cap))
                st.success(f"Турбина {tid} добавлена. Прогноз погоды для неё агент скачает при первом прогнозе.")
            except Exception as e:
                st.error(str(e))
    extra = [x.id for x in tbs if not x.history]
    if extra:
        r1, r2 = st.columns([1, 3])
        rm = r1.selectbox("Удалить добавленную турбину", extra)
        if r2.button("Удалить"):
            remove_turbine(rm)
            st.rerun()
    st.caption(f"Новая турбина прогнозируется общей моделью парка по прогнозу погоды в её координатах. "
               f"Архив погоды доступен с {WEATHER_START}.")
