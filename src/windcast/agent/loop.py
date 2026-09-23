"""Оркестратор агента: цикл «погода → проверка → прогноз → анализ → перерасчёт → отчёт».

`RuleAgent` принимает решения по фиксированным правилам (детерминированно, без ключей API).
LLM-агент использует те же инструменты и тот же журнал решений.
"""
import json
import time
from pathlib import Path

import pandas as pd

from ..config import LOCAL_TZ_LABEL, OUTPUTS_DIR
from ..forecast import TOTAL
from .tools import CHANGE_NOTABLE, Toolbox


class AgentLog:
    """Журнал решений агента (JSONL): шаг, инструмент, аргументы, результат, обоснование."""

    def __init__(self):
        self.steps = []

    def add(self, tool: str, args: dict, result, reason: str):
        self.steps.append({"step": len(self.steps) + 1, "ts": time.strftime("%H:%M:%S"), "tool": tool,
                           "args": args, "reason": reason, "result": result})
        return result

    def decisions(self):
        return [f"{s['step']}. {s['tool']}: {s['reason']}" for s in self.steps]

    def dump(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in self.steps:
                f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")


class RuleAgent:
    mode = "rules"

    def __init__(self, issue_local, runs_dir: Path = OUTPUTS_DIR / "runs", online: bool = True, model=None,
                 turbines=None, force: bool = False):
        kw = {"model": model} | ({"turbines": turbines} if turbines else {})
        self.tb = Toolbox(issue_local, runs_dir=runs_dir, online=online, **kw)
        self.log = AgentLog()
        self.force = force

    def call(self, tool: str, reason: str, **args):
        return self.log.add(tool, args, getattr(self.tb, tool)(**args), reason)

    def run(self) -> dict:
        tb = self.tb
        # 1. Погода: есть ли в кеше, при необходимости — докачать
        w = self.call("check_weather", "проверяю наличие архивных прогнозов погоды на 48 ч от момента выпуска")
        if w["incomplete"]:
            f = self.call("fetch_weather", f"в кеше не хватает данных: {', '.join(w['incomplete'])}")
            if f.get("ok"):
                self.call("check_weather", "повторная проверка после загрузки")

        # 2. Проверка входов → выбор моделей погоды
        v = self.call("validate_inputs", "проверяю пропуски, невозможные значения и расхождение моделей погоды")
        usable = v["usable_models"]
        if not usable:
            self.log.add("stop", {}, None, "нет ни одной пригодной модели погоды — прогноз невозможен")
            return self._finish(status="failed")
        dropped = [m for m in v["per_model"] if m not in usable]
        tb.nwp_models = usable

        # 3. Перерасчёт только при изменении входов
        prev = self.call("check_previous_version", "есть ли прогноз этого выпуска и изменились ли входные данные")
        if prev["exists"] and not prev["inputs_changed"] and not self.force:
            self.log.add("reuse", {}, None, "входные данные не изменились — перерасчёт не нужен")
            tb.result = pd.read_csv(tb.dir / "forecast.csv", parse_dates=["time_local", "time_utc"])
            tb.input_hash = prev["old_hash"]
            return self._finish(status="unchanged")
        reason = ("входные данные изменились — пересчитываю" if prev.get("inputs_changed")
                  else "первый расчёт для этого момента выпуска")
        if dropped:
            reason += f"; исключены модели погоды: {', '.join(dropped)}"
        self.call("run_forecast", reason, nwp_models=usable)

        # 4. Анализ; при неправдоподобном результате — повтор с одной лучшей моделью
        a = self.call("analyze_forecast", "проверяю правдоподобность и уверенность прогноза")
        if not a["sane"] and len(usable) > 1:
            best = min(usable, key=lambda m: v["per_model"][m]["mean_abs_dev_ms"])
            self.call("run_forecast", f"результат неправдоподобен — пересчёт только на {best}", nwp_models=[best])
            a = self.call("analyze_forecast", "повторная проверка")

        # 5. Контекст: изменения относительно прошлых выпусков, ошибка на известном факте
        c = self.call("compare_with_previous", "сравниваю с предыдущим выпуском на общих часах")
        e = self.call("evaluate_recent", "считаю ошибку прошлых выпусков, где факт уже известен")
        return self._finish(status="ok", analysis=a, compare=c, recent=e, validation=v, recomputed=prev)

    def _finish(self, status, **ctx) -> dict:
        tb = self.tb
        if status == "unchanged":
            self.log.dump(tb.dir / "agent_log_reuse.jsonl")
            return {"status": status, "dir": str(tb.dir), "result": tb.result}
        if status == "failed":
            self.log.dump(tb.dir / "agent_log.jsonl")
            return {"status": status, "dir": str(tb.dir), "result": None}
        report = render_report(tb, **ctx, mode=self.mode)
        saved = self.log.add("save", {}, tb.save(report, self.log.decisions()), "сохраняю прогноз, отчёт и журнал")
        self.log.dump(tb.dir / "agent_log.jsonl")
        return {"status": status, "dir": saved["dir"], "result": tb.result, "report": report}


def render_report(tb: Toolbox, analysis, compare, recent, validation, recomputed, mode="rules",
                  summary: str | None = None) -> str:
    df = tb.result
    tot = df[df.turbine == TOTAL]
    lines = [f"# Прогноз выработки ВЭС — выпуск {tb.issue_local:%d.%m.%Y %H:%M} ({LOCAL_TZ_LABEL})", "",
             f"Период: {tot.time_local.min():%d.%m %H:00} — {tot.time_local.max():%d.%m %H:00}, 48 ч. "
             f"Мощность — в долях номинала. Режим агента: **{mode}**.", ""]
    if summary:
        lines += ["## Сводка агента", "", summary.strip(), ""]
    lines += ["## Итог", "", f"- Уверенность: **{analysis['confidence']}**",
              f"- Средняя мощность ВЭС: **{tot.p50.mean():.2f}** (P10–P90: {tot.p10.mean():.2f}–{tot.p90.mean():.2f})",
              "- По турбинам: " + ", ".join(f"{k} {v:.2f}" for k, v in analysis["turbine_mean_p50"].items()),
              f"- Модели погоды: {', '.join(tb.nwp_models)}; среднее расхождение ветра "
              f"{validation['ws_spread_ms']['mean']} м/с (макс. {validation['ws_spread_ms']['max']})", ""]
    lines += ["## Профиль по 6 часов (ВЭС, P50)", "", "| Начало | Мощность |", "|---|---|"]
    lines += [f"| {pd.Timestamp(k):%d.%m %H:00} | {v:.2f} |"
              for k, v in analysis["total_profile_6h"].items()]
    lines += ["", "## Риски и предупреждения", ""]
    risks = [("Широкий интервал неопределённости", analysis["wide_interval_hours"]),
             ("Модели погоды расходятся (>3 м/с)", analysis["nwp_disagreement_hours"]),
             ("Риск обледенения (T ≈ 0 °C, влажность ≥ 90%)", analysis["icing_risk_hours"]),
             ("Резкие изменения мощности (≥ 0.25 за час)", analysis["ramp_hours"])]
    any_risk = False
    for name, hours in risks:
        if hours:
            any_risk = True
            lines.append(f"- **{name}:** {', '.join(hours[:6])}{' …' if len(hours) > 6 else ''}")
    if not any_risk:
        lines.append("- Существенных рисков не обнаружено.")
    lines += ["", "## Перерасчёт и сравнение", ""]
    if recomputed.get("exists"):
        ch = (compare.get("same_issue") or {}).get("mean_abs_change")
        lines.append(f"- Прогноз этого выпуска уже был; входные данные изменились → пересчитан"
                     f"{f' (среднее изменение {ch:.3f})' if ch is not None else ''}.")
    pi = compare.get("previous_issue")
    if pi and pi["mean_abs_change_overlap"] is not None:
        d = pi["mean_abs_change_overlap"]
        note = "заметно" if d >= CHANGE_NOTABLE else "незначительно"
        lines.append(f"- Относительно выпуска {pi['issue']} прогноз на общих часах изменился {note}: {d:.3f}.")
    if recent.get("available"):
        lines.append(f"- Ошибка прошлых выпусков за 7 суток (факт известен): NMAE {recent['NMAE']:.3f}, "
                     f"смещение {recent['bias']:+.3f} ({recent['hours']} ч).")
    else:
        lines.append(f"- Оценка по факту недоступна: {recent.get('reason')}.")
    return "\n".join(lines) + "\n"
