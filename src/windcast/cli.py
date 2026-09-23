"""Командная строка: windcast <команда> [опции]."""
import argparse
import sys

import pandas as pd

from .config import LOCAL_TZ_LABEL, WEATHER_END, WEATHER_START, to_utc


def cmd_download(a):
    from .turbines import load_turbines
    from .weather import download
    for t in load_turbines():
        if a.turbine and t.id != a.turbine:
            continue
        print(f"Загрузка прогнозов погоды для {t.id} ({t.lat}, {t.lon})")
        download(t, a.start, a.end)


def cmd_train(a):
    from .model import MODEL_FILE, WindModel, training_frame
    from .turbines import load_turbines
    df = training_frame(load_turbines(), end=to_utc(a.until))
    print(f"Обучение: {len(df)} строк, {df.index.min()} … {df.index.max()} UTC")
    m = WindModel().fit(df)
    m.save()
    print(f"Модель сохранена: {MODEL_FILE}\nКалибровка интервала: {m.scale}")


def cmd_validate(a):
    from .validation import run_all
    print(run_all())


def cmd_diagnose(a):
    from .diagnostics import run_all
    print(run_all())


def cmd_forecast(a):
    from .agent import make_agent
    from .forecast import to_wide
    res = make_agent(a.agent, pd.Timestamp(a.issue), online=not a.offline, force=a.force).run()
    print(f"Статус: {res['status']} → {res['dir']}")
    if res["result"] is not None:
        with pd.option_context("display.max_rows", 60, "display.width", 160):
            print(to_wide(res["result"]).to_string(index=False))
    if res.get("report"):
        print("\n" + res["report"])


def cmd_backtest(a):
    from .backtest import run_backtest
    wide = run_backtest(a.start, a.end, hours=tuple(a.hours.split(",")), agent=a.agent, online=not a.offline)
    print(wide.describe().round(3).to_string())


def cmd_add_turbine(a):
    from .turbines import Turbine, add_turbine
    from .weather import download
    t = Turbine(id=a.id, lat=a.lat, lon=a.lon, cap=a.cap)
    add_turbine(t)
    print(f"Турбина {t.id} добавлена в turbines.yaml")
    if not a.no_download:
        download(t, a.start, a.end)


def main(argv=None):
    p = argparse.ArgumentParser(prog="windcast", description="Агентный прогноз выработки ВЭС на 24–48 ч")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="скачать архивные прогнозы погоды в data/weather/")
    s.add_argument("--turbine")
    s.add_argument("--start", default=WEATHER_START)
    s.add_argument("--end", default=WEATHER_END)
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("train", help="обучить модель на истории до момента --until")
    s.add_argument("--until", default="2026-01-31 10:00", help=f"граница обучения, {LOCAL_TZ_LABEL}")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("validate", help="честная проверка на феврале 2025 и январе 2026")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("diagnose", help="проверки данных: часовой пояс, точность погоды, потолок мощности")
    s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser("forecast", help="прогноз на 48 ч от момента выпуска")
    s.add_argument("--issue", required=True, help=f"момент выпуска, «ГГГГ-ММ-ДД ЧЧ:ММ» ({LOCAL_TZ_LABEL})")
    s.add_argument("--agent", choices=["rules", "llm"], default="rules")
    s.add_argument("--offline", action="store_true", help="не обращаться к сети, только кеш")
    s.add_argument("--force", action="store_true", help="пересчитать даже без изменения входов")
    s.set_defaults(fn=cmd_forecast)

    s = sub.add_parser("backtest", help="ежедневные выпуски за период (тест: 31.01–27.02.2026)")
    s.add_argument("--start", default="2026-01-31")
    s.add_argument("--end", default="2026-02-27")
    s.add_argument("--hours", default="10:00,22:00", help="время выпуска (UTC+6) через запятую")
    s.add_argument("--agent", choices=["rules", "llm"], default="rules")
    s.add_argument("--offline", action="store_true")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("add-turbine", help="добавить турбину по координатам")
    s.add_argument("--id", required=True)
    s.add_argument("--lat", type=float, required=True)
    s.add_argument("--lon", type=float, required=True)
    s.add_argument("--cap", type=float, default=1.0, help="максимальная доля мощности (потолок)")
    s.add_argument("--start", default=WEATHER_START)
    s.add_argument("--end", default=WEATHER_END)
    s.add_argument("--no-download", action="store_true")
    s.set_defaults(fn=cmd_add_turbine)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
