"""Streamlit-интерфейс: прогноз по дате и времени выпуска, результаты теста, валидация, турбины."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import datetime as dt
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from windcast.agent import make_agent
from windcast.agent.tools import CONF_HIGH, CONF_MID
from windcast.config import LOCAL_TZ_LABEL, OUTPUTS_DIR, WEATHER_START
from windcast.forecast import TOTAL, to_wide
from windcast.model import WindModel
from windcast.turbines import Turbine, add_turbine, load_turbines, remove_turbine

st.set_page_config(page_title="windcast — прогноз выработки ВЭС", page_icon=":material/wind_power:",
                   layout="wide")

st.html(Path(__file__).parent / "assets" / "glass.css")  # стеклянный эффект и фон

# Ключи LLM на Streamlit Cloud задаются в Secrets → переносим в окружение для windcast.agent.llm
try:
    for k in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "NVIDIA_API_KEY"):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except FileNotFoundError:
    pass

COLORS = {"T1": "#34D399", "T2": "#FBBF24", TOTAL: "#22D3EE"}
BAND = "rgba(34, 211, 238, 0.14)"
GRID = "rgba(148, 197, 255, 0.08)"
FEB = OUTPUTS_DIR / "feb2026"
VAL = OUTPUTS_DIR / "validation"


@st.cache_resource
def get_model():
    return WindModel.load()


def confidence(width: float) -> tuple[str, str]:
    if width < CONF_HIGH:
        return "высокая", "green"
    return ("средняя", "orange") if width < CONF_MID else ("низкая", "red")


def forecast_chart(df: pd.DataFrame, height: int = 420, ticks: str = "%d.%m<br>%H:%M") -> go.Figure:
    fig = go.Figure()
    tot = df[df.turbine == TOTAL]
    fig.add_trace(go.Scatter(x=tot.time_local, y=tot.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tot.time_local, y=tot.p10, fill="tonexty", fillcolor=BAND, line=dict(width=0),
                             name="ВЭС: интервал P10–P90", hovertemplate="%{y:.2f}"))
    fig.add_trace(go.Scatter(x=tot.time_local, y=tot.p50, mode="lines", showlegend=False, hoverinfo="skip",
                             line=dict(width=10, color="rgba(34, 211, 238, 0.18)", shape="spline")))
    for tb, g in df.groupby("turbine", sort=False):
        fig.add_trace(go.Scatter(x=g.time_local, y=g.p50, name=f"{tb}" if tb != TOTAL else "ВЭС (итог)",
                                 mode="lines", hovertemplate="%{y:.2f}",
                                 line=dict(width=3 if tb == TOTAL else 1.5, color=COLORS.get(tb), shape="spline",
                                           dash="solid" if tb == TOTAL else "dot")))
    return _style(fig, height, "Мощность, доля номинала", ticks)


def _style(fig: go.Figure, height: int, ytitle: str, ticks: str = "%d.%m") -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=12, color="#C7D5E8"),
                      hoverlabel=dict(bgcolor="rgba(10,19,36,0.92)", bordercolor="#22D3EE", font_color="#E6EEF8"),
                      margin=dict(t=10, l=10, r=10, b=10),
                      yaxis=dict(title=ytitle, range=[0, 1.02], gridcolor=GRID, zeroline=False),
                      xaxis=dict(title=f"Время ({LOCAL_TZ_LABEL})", gridcolor=GRID, tickformat=ticks,
                                 hoverformat="%d.%m.%Y %H:%M", zeroline=False),
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def kpis(df: pd.DataFrame):
    tot = df[df.turbine == TOTAL].sort_values("time_local")
    width = float((tot.p90 - tot.p10).mean())
    label, _ = confidence(width)
    peak = tot.loc[tot.p50.idxmax()]
    low = tot.loc[tot.p50.idxmin()]
    energy = tot.p50.sum()
    with st.container(horizontal=True):
        st.metric("Средняя мощность ВЭС", f"{tot.p50.mean():.2f}", "доля номинала, P50", delta_color="off",
                  delta_arrow="off", border=True, height="stretch", help="Медианный прогноз (P50) за 48 ч")
        st.metric("Пик", f"{peak.p50:.2f}", f"{peak.time_local:%d.%m %H:00}", delta_color="off", delta_arrow="off", border=True, height="stretch")
        st.metric("Минимум", f"{low.p50:.2f}", f"{low.time_local:%d.%m %H:00}", delta_color="off", delta_arrow="off", border=True, height="stretch")
        st.metric("Выработка за период", f"{energy:.1f}", "номинал·часов", delta_color="off", delta_arrow="off", border=True, height="stretch",
                  help="Сумма почасовых долей: 1.0 = один час работы на номинале")
        st.metric("Уверенность", label, f"ширина P10–P90: {width:.2f}", delta_color="off", delta_arrow="off", border=True, height="stretch")


# ------------------------------------------------------------------ боковая панель
with st.sidebar:
    st.markdown("### :material/wind_power: windcast")
    st.caption("Агентный прогноз выработки ветроэлектростанции на 48 часов")
    st.markdown("**Параметры выпуска**")
    d = st.date_input("Дата выпуска", dt.date(2026, 2, 1), min_value=dt.date(2024, 3, 10), format="DD.MM.YYYY")
    t = st.time_input("Время выпуска", dt.time(10, 0), step=3600, help=f"Время {LOCAL_TZ_LABEL}, как в данных SCADA")
    mode = st.segmented_control("Режим агента", ["rules", "llm"], default="rules", required=True,
                                format_func=lambda m: {"rules": "Правила", "llm": "LLM"}[m],
                                help="Правила — детерминированно, без ключа. LLM — модель сама выбирает шаги "
                                     "и пишет сводку; без ключа агент откатится на правила.")
    force = st.toggle("Пересчитать принудительно", False)
    run = st.button("Сформировать прогноз", type="primary", icon=":material/play_arrow:", width="stretch", key="run")
    st.space("small")
    st.caption(f"Мощность — в долях номинала. Время — {LOCAL_TZ_LABEL}. "
               "Погода — архивные прогнозы Open-Meteo (GFS, ICON, ECMWF), доступные на момент выпуска.")

st.title("Прогноз выработки ВЭС")
st.markdown(":blue-badge[:material/cloud: Open-Meteo Previous Runs] :blue-badge[:material/model_training: Бустинг P10/P50/P90] "
            ":blue-badge[:material/smart_toy: Агент: правила / LLM] :gray-badge[2 турбины · Алматинская обл.]")

tab_fc, tab_test, tab_val, tab_tb = st.tabs([":material/query_stats: Прогноз", ":material/calendar_month: Тест: февраль 2026",
                                             ":material/fact_check: Валидация", ":material/wind_power: Турбины"])

# ------------------------------------------------------------------ прогноз
with tab_fc:
    if run:
        issue = pd.Timestamp(dt.datetime.combine(d, t))
        with st.status(f"Агент выпускает прогноз на {issue:%d.%m.%Y %H:%M}…", expanded=False) as status:
            try:
                res = make_agent(mode, issue, runs_dir=OUTPUTS_DIR / "runs_app", online=True,
                                 model=get_model(), force=force).run()
                st.session_state["res"] = res
                status.update(label={"ok": "Прогноз рассчитан", "unchanged": "Входы не изменились — прогноз переиспользован"}
                              .get(res["status"], res["status"]), state="complete")
            except Exception as e:  # показать ошибку пользователю, не роняя приложение
                status.update(label="Ошибка при выпуске прогноза", state="error")
                st.exception(e)
    res = st.session_state.get("res")
    if not res or res.get("result") is None:
        with st.container(border=True, key="glass_1"):
            st.markdown("**:material/info: Как получить прогноз**")
            st.markdown(
                "1. Слева выберите дату и время выпуска (например, 01.02.2026 10:00) и режим агента.\n"
                "2. Нажмите **Сформировать прогноз**.\n"
                "3. Агент проверит архив погоды, при нехватке докачает его, оценит качество входов, запустит модель, "
                "проанализирует риски и сохранит отчёт. Каждый шаг — в журнале решений.")
    else:
        df = res["result"].copy()
        df["time_local"] = pd.to_datetime(df["time_local"])
        issue_s = pd.to_datetime(Path(res["dir"]).name, format="%Y%m%d_%H%M")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(f"Выпуск {issue_s:%d.%m.%Y %H:%M}")
            st.badge("LLM" if res.get("usage") else "Правила", icon=":material/smart_toy:",
                     color="violet" if res.get("usage") else "gray")
            if res["status"] == "unchanged":
                st.badge("Входы не изменились — прогноз переиспользован", icon=":material/history:", color="gray")
            if res.get("fallback"):
                st.badge("LLM недоступна — режим правил", icon=":material/warning:", color="orange")
            if res.get("usage"):
                u = res["usage"]
                st.caption(f"{u['calls']} запросов к LLM · {u['prompt_tokens']:,} + {u['completion_tokens']:,} токенов"
                           .replace(",", " "))
        kpis(df)
        with st.container(border=True, key="glass_2"):
            st.markdown("**Почасовой прогноз мощности на 48 ч**")
            st.plotly_chart(forecast_chart(df), width="stretch")

        t_rep, t_tab, t_log = st.tabs([":material/description: Отчёт агента", ":material/table_chart: Таблица",
                                       ":material/account_tree: Журнал решений"])
        with t_rep:
            rep = Path(res["dir"]) / "report.md"
            if rep.exists():
                st.markdown(rep.read_text(encoding="utf-8").split("\n", 2)[2])  # без заголовка H1
        with t_tab:
            wide = to_wide(df)
            cfg = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.2f")
                   for c in wide.columns if c.endswith("_p50")}
            cfg["time_local"] = st.column_config.DatetimeColumn(f"Время ({LOCAL_TZ_LABEL})", format="DD.MM HH:mm")
            st.dataframe(wide, hide_index=True, height=420, column_config=cfg)
            st.download_button("Скачать CSV", wide.to_csv(index=False).encode("utf-8"), icon=":material/download:",
                               file_name=f"forecast_{issue_s:%Y%m%d_%H%M}.csv", mime="text/csv")
        with t_log:
            logf = Path(res["dir"]) / ("agent_log_reuse.jsonl" if res["status"] == "unchanged" else "agent_log.jsonl")
            if logf.exists():
                log = pd.read_json(logf, lines=True)
                for _, r in log.iterrows():
                    with st.expander(f"**{r.step}. {r.tool}** — {r.reason}", icon=":material/check_circle:"):
                        st.json(r.result if isinstance(r.result, (dict, list)) else {"result": r.result},
                                expanded=True)

# ------------------------------------------------------------------ тест
with tab_test:
    f = FEB / "forecast_hourly_latest.csv"
    if f.exists():
        df = pd.read_csv(f, parse_dates=["time_local"])
        runs = sorted((FEB / "runs").glob("*_*"))
        tot = df[df.turbine == TOTAL]
        st.caption("Воспроизведение теста «как в прошлом»: выпуски 31.01–27.02.2026 в 10:00 и 22:00, "
                   "для каждого часа — прогноз из самого свежего выпуска. Команда: `windcast backtest --offline`.")
        with st.container(horizontal=True):
            st.metric("Выпусков прогноза", len(runs), "31.01–27.02, 10:00 и 22:00", delta_color="off", delta_arrow="off",
                      border=True, height="stretch")
            st.metric("Часов в тесте", tot.time_local.nunique(), "01.02–28.02.2026", delta_color="off", delta_arrow="off",
                      border=True, height="stretch")
            st.metric("Средняя мощность ВЭС", f"{tot.p50.mean():.2f}", "доля номинала, P50", delta_color="off",
                      delta_arrow="off", border=True, height="stretch")
            st.metric("Выработка за февраль", f"{tot.p50.sum():.0f}", "номинал·часов", delta_color="off", delta_arrow="off", border=True, height="stretch")
        with st.container(border=True, key="glass_3"):
            st.markdown("**Февраль 2026 — почасовой прогноз P50 с интервалом P10–P90**")
            st.plotly_chart(forecast_chart(df, height=380, ticks="%d.%m"), width="stretch")
            st.download_button("Скачать прогноз за февраль (P50, CSV)", (FEB / "forecast_hourly_p50.csv").read_bytes(),
                               file_name="feb2026_forecast_p50.csv", mime="text/csv", icon=":material/download:")
        with st.container(border=True, key="glass_4"):
            names = [r.name for r in runs]
            pick = st.selectbox("Отчёт агента по выпуску", names, index=len(names) - 1 if names else 0,
                                format_func=lambda n: f"{pd.to_datetime(n, format='%Y%m%d_%H%M'):%d.%m.%Y %H:%M}")
            if pick:
                st.markdown((FEB / "runs" / pick / "report.md").read_text(encoding="utf-8").split("\n", 2)[2])
    else:
        st.warning("Нет результатов бэктеста. Запустите `uv run windcast backtest --offline`.", icon=":material/warning:")

# ------------------------------------------------------------------ валидация
with tab_val:
    v = VAL / "validation_scores.csv"
    if v.exists():
        sc = pd.read_csv(v)
        st.caption("Честная проверка: модель обучена только на данных до первого выпуска, прогноз выпускался ежедневно "
                   "в 10:00 на 48 ч с погодой «как на момент выпуска». NMAE — средняя ошибка в долях номинала, меньше — лучше.")
        tot = sc[sc.turbine == TOTAL]
        with st.container(horizontal=True):
            for fold in tot.fold.unique():
                for hz in sorted(tot.horizon.unique()):
                    g = tot[(tot.fold == fold) & (tot.horizon == hz)].set_index("method")["NMAE"]
                    gain = 1 - g["p50"] / g["curve"]
                    st.metric(f"{fold} · {hz.split(' ')[0]}", f"{g['p50']:.3f}", f"-{gain:.0%} к «погода → кривая»",
                              delta_color="inverse", border=True, height="stretch", help="NMAE модели по ВЭС")
        vf = pd.read_csv(VAL / "validation_forecasts.csv", parse_dates=["time_local"])
        with st.container(border=True, key="glass_5"):
            fold = st.segmented_control("Период", list(vf.fold.unique()), default=vf.fold.unique()[0], required=True)
            g = vf[(vf.fold == fold) & (vf.turbine == TOTAL) & (vf.horizon.str.startswith("D+1"))].sort_values("time_local")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g.time_local, y=g.actual, name="Факт", line=dict(color="#E6EEF8", width=1.3)))
            fig.add_trace(go.Scatter(x=g.time_local, y=g.p50, name="Модель P50 (на сутки вперёд)",
                                     line=dict(color="#22D3EE", width=2)))
            fig.add_trace(go.Scatter(x=g.time_local, y=g.curve, name="Погода → кривая мощности",
                                     line=dict(color="#FBBF24", dash="dot", width=1.5)))
            st.plotly_chart(_style(fig, 380, "Мощность ВЭС, доля номинала"), width="stretch")
        with st.expander("Полные таблицы метрик", icon=":material/table_view:"):
            st.markdown((VAL / "validation.md").read_text(encoding="utf-8").split("\n", 2)[2])
    else:
        st.warning("Нет результатов валидации. Запустите `uv run windcast validate`.", icon=":material/warning:")

# ------------------------------------------------------------------ турбины
with tab_tb:
    tbs = load_turbines()
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True, key="glass_6"):
            st.markdown("**Турбины ВЭС**")
            st.dataframe(pd.DataFrame([{"ID": x.id, "Широта": x.lat, "Долгота": x.lon, "Потолок мощности": x.cap,
                                        "История SCADA": "есть" if x.history else "нет"} for x in tbs]),
                         hide_index=True, column_config={"Потолок мощности": st.column_config.NumberColumn(format="%.2f")})
            st.map(pd.DataFrame({"lat": [x.lat for x in tbs], "lon": [x.lon for x in tbs]}), zoom=11, height=320, size=45,
                   color="#22D3EE")
    with right:
        with st.form("add", border=True):
            st.markdown("**Добавить турбину**")
            tid = st.text_input("ID", "T3")
            lat = st.number_input("Широта", value=43.6400, format="%.6f")
            lon = st.number_input("Долгота", value=78.5450, format="%.6f")
            cap = st.number_input("Потолок мощности", value=0.98, min_value=0.1, max_value=1.0)
            if st.form_submit_button("Добавить", icon=":material/add:", width="stretch"):
                try:
                    add_turbine(Turbine(id=tid.strip(), lat=lat, lon=lon, cap=cap))
                    st.success(f"Турбина {tid} добавлена. Погоду для неё агент скачает при первом прогнозе.",
                               icon=":material/check_circle:")
                except Exception as e:
                    st.error(str(e), icon=":material/error:")
        extra = [x.id for x in load_turbines() if not x.history]
        if extra:
            with st.container(border=True, key="glass_7"):
                rm = st.selectbox("Удалить добавленную турбину", extra)
                if st.button("Удалить", icon=":material/delete:"):
                    remove_turbine(rm)
                    st.rerun()
        st.caption(f"Новая турбина прогнозируется общей моделью парка по прогнозу погоды в её координатах. "
                   f"Архив погоды доступен с {WEATHER_START}.")
