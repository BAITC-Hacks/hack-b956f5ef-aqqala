"""LLM-агент: модель сама выбирает инструменты, решает о перерасчёте и пишет сводку диспетчеру.

Провайдер — любой OpenAI-совместимый API (OpenAI или NVIDIA NIM), настройки из окружения:
  LLM_PROVIDER=openai|nvidia, LLM_MODEL, OPENAI_API_KEY / NVIDIA_API_KEY.
Числа считает только код инструментов. При ошибке LLM агент откатывается на режим правил.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ..config import OUTPUTS_DIR, ROOT
from .loop import AgentLog, RuleAgent, render_report
from .tools import Toolbox

load_dotenv(ROOT / ".env")

PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "model": "gpt-4.1-mini"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY",
               "model": "meta/llama-3.3-70b-instruct"},
}
MAX_STEPS = 14
MAX_RESULT_CHARS = 3500

SYSTEM = """Ты — агент-диспетчер прогноза выработки ветроэлектростанции (ВЭС) в Казахстане.
Задача: выпустить почасовой прогноз мощности на 48 часов от момента выпуска (время UTC+6) по каждой
турбине и итог по ВЭС, в долях номинала. Прогнозы погоды — архивные, ровно те, что были доступны
на момент выпуска (это обеспечивают инструменты).

Порядок работы (решения принимаешь сам, но все шаги должны быть осмысленны):
1. Проверь наличие прогнозов погоды (check_weather). Если чего-то не хватает — fetch_weather, затем проверь снова.
2. Проверь качество входов (validate_inputs). Исключи модели погоды с пропусками или невозможными значениями.
3. Проверь, есть ли уже прогноз этого выпуска и изменились ли входы (check_previous_version).
   Если входы не изменились — вызови reuse_previous и заверши работу. Иначе — run_forecast.
4. Проанализируй результат (analyze_forecast). Если результат неправдоподобен (sane=false) —
   пересчитай на меньшем наборе моделей погоды.
5. Сравни с прошлым выпуском (compare_with_previous) и проверь ошибку прошлых выпусков (evaluate_recent).
6. Заверши вызовом finish со сводкой для диспетчера.

Правила:
- Используй только числа из результатов инструментов, ничего не выдумывай.
- В каждом вызове заполняй поле reason: коротко, почему ты это делаешь (по-русски).
- Сводка в finish: 4–8 предложений по-русски — когда ожидается высокая/низкая выработка (с часами),
  уверенность и её причины, риски (обледенение, расхождение моделей, резкие изменения),
  что изменилось относительно прошлого выпуска и была ли ошибка в прошлых прогнозах."""


def _tool(name, desc, props=None, required=None):
    props = dict(props or {})
    props["reason"] = {"type": "string", "description": "почему вызывается инструмент (по-русски)"}
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": (required or []) + ["reason"]}}}


TOOLS = [
    _tool("check_weather", "Проверить наличие архивных прогнозов погоды в кеше на 48 ч от момента выпуска."),
    _tool("fetch_weather", "Докачать прогнозы погоды из Open-Meteo Previous Runs API для окна выпуска.",
          {"turbine_ids": {"type": "array", "items": {"type": "string"}, "description": "какие турбины (все по умолчанию)"}}),
    _tool("validate_inputs", "Проверить качество прогнозов погоды: пропуски, невозможные значения, расхождение моделей."),
    _tool("check_previous_version", "Есть ли уже прогноз этого выпуска и изменились ли входные данные с тех пор."),
    _tool("reuse_previous", "Взять ранее сохранённый прогноз этого выпуска (только если входы не изменились)."),
    _tool("run_forecast", "Запустить ML-модель прогноза на 48 ч.",
          {"nwp_models": {"type": "array", "items": {"type": "string", "enum": ["gfs", "icon", "ecmwf"]},
                          "description": "модели погоды для прогноза"}}, ["nwp_models"]),
    _tool("analyze_forecast", "Проверить результат: правдоподобность, уверенность, расхождение погоды, обледенение, рампы."),
    _tool("compare_with_previous", "Сравнить с предыдущим выпуском на общих часах и с прошлой версией этого выпуска."),
    _tool("evaluate_recent", "Ошибка прошлых выпусков за 7 суток, если факт уже известен."),
    _tool("finish", "Завершить работу: сохранить прогноз и отчёт со сводкой для диспетчера.",
          {"summary": {"type": "string", "description": "сводка для диспетчера, 4–8 предложений"}}, ["summary"]),
]


def llm_config():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    p = PROVIDERS[provider]
    return {"provider": provider, "base_url": p["base_url"], "api_key": os.getenv(p["key_env"]),
            "key_env": p["key_env"], "model": os.getenv("LLM_MODEL") or p["model"]}


class LLMAgent:
    def __init__(self, issue_local, runs_dir: Path = OUTPUTS_DIR / "runs", online: bool = True, model=None,
                 turbines=None, force: bool = False):
        self.args = dict(issue_local=issue_local, runs_dir=runs_dir, online=online, model=model,
                         turbines=turbines, force=force)
        kw = {"model": model} | ({"turbines": turbines} if turbines else {})
        self.tb = Toolbox(issue_local, runs_dir=runs_dir, online=online, **kw)
        self.log = AgentLog()
        self.cfg = llm_config()
        self.mode = f"llm:{self.cfg['provider']}/{self.cfg['model']}"
        self.ctx = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    # ------------------------------------------------------------ цикл
    def run(self) -> dict:
        if not self.cfg["api_key"]:
            return self._fallback(f"нет ключа {self.cfg['key_env']}")
        try:
            return self._loop()
        except Exception as e:  # сеть, лимиты, неверный ответ модели
            return self._fallback(f"ошибка LLM: {type(e).__name__}: {str(e)[:200]}")

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.cfg["api_key"], base_url=self.cfg["base_url"], timeout=60, max_retries=2)

    def _loop(self) -> dict:
        client = self._client()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Момент выпуска: {self.tb.issue_local:%Y-%m-%d %H:%M} (UTC+6). "
                                                f"Турбины: {', '.join(t.id for t in self.tb.turbines)}. "
                                                f"Режим сети: {'онлайн' if self.tb.online else 'офлайн'}."}]
        extra = {} if self.cfg["model"].startswith(("gpt-5", "o")) else {"temperature": 0}
        # NIM-модели не везде поддерживают tool_choice="required"
        extra["tool_choice"] = "required" if self.cfg["provider"] == "openai" else "auto"
        for _ in range(MAX_STEPS):
            t0 = time.time()
            resp = client.chat.completions.create(model=self.cfg["model"], messages=messages, tools=TOOLS, **extra)
            if resp.usage:
                self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                self.usage["completion_tokens"] += resp.usage.completion_tokens
            self.usage["calls"] += 1
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                messages.append({"role": "user", "content": "Вызови инструмент (или finish)."})
                continue
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                reason = args.pop("reason", "") or "(без обоснования)"
                if name == "finish":
                    return self._finish(args.get("summary", ""), reason)
                if name == "reuse_previous":
                    result = self._reuse()
                    if result.get("ok"):
                        self.log.add(name, args, result, reason)
                        return self._done_reuse()
                else:
                    result = self._exec(name, args)
                self.log.add(name, args, result, reason + f" [{time.time() - t0:.1f} с]")
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)[:MAX_RESULT_CHARS]})
        return self._fallback(f"превышен лимит шагов ({MAX_STEPS})")

    def _exec(self, name, args):
        allowed = {"check_weather", "fetch_weather", "validate_inputs", "check_previous_version", "run_forecast",
                   "analyze_forecast", "compare_with_previous", "evaluate_recent"}
        if name not in allowed:
            return {"error": f"неизвестный инструмент {name}"}
        if name in {"analyze_forecast", "compare_with_previous"} and self.tb.result is None:
            return {"error": "сначала нужен run_forecast"}
        try:
            res = getattr(self.tb, name)(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        self.ctx[name] = res
        return res

    def _reuse(self):
        prev = self.ctx.get("check_previous_version") or self.tb.check_previous_version()
        if not prev.get("exists") or prev.get("inputs_changed") or self.args["force"]:
            return {"ok": False, "error": "повторное использование невозможно: прогноза нет или входы изменились"}
        self.tb.result = pd.read_csv(self.tb.dir / "forecast.csv", parse_dates=["time_local", "time_utc"])
        self.tb.input_hash = prev["old_hash"]
        return {"ok": True}

    def _done_reuse(self):
        self.log.dump(self.tb.dir / "agent_log_reuse.jsonl")
        return {"status": "unchanged", "dir": str(self.tb.dir), "result": self.tb.result, "usage": self.usage}

    def _finish(self, summary: str, reason: str) -> dict:
        tb = self.tb
        if tb.result is None:  # модель попыталась завершить без прогноза — делаем обязательный шаг сами
            self.log.add("run_forecast", {}, tb.run_forecast(), "обязательный шаг: LLM завершила без прогноза")
        # недостающие для отчёта проверки выполняются детерминированно
        need = {"validate_inputs": {}, "analyze_forecast": {}, "compare_with_previous": {}, "evaluate_recent": {},
                "check_previous_version": {"exists": False}}
        for k in need:
            if k not in self.ctx or (k == "analyze_forecast" and self.ctx[k].get("error")):
                self.ctx[k] = getattr(tb, k)() if k != "check_previous_version" else need[k]
        self.log.add("finish", {"summary": summary}, {"usage": self.usage}, reason)
        report = render_report(tb, analysis=self.ctx["analyze_forecast"], compare=self.ctx["compare_with_previous"],
                               recent=self.ctx["evaluate_recent"], validation=self.ctx["validate_inputs"],
                               recomputed=self.ctx["check_previous_version"], mode=self.mode, summary=summary)
        saved = tb.save(report, self.log.decisions())
        self.log.dump(tb.dir / "agent_log.jsonl")
        return {"status": "ok", "dir": saved["dir"], "result": tb.result, "report": report, "usage": self.usage}

    def _fallback(self, why: str) -> dict:
        agent = RuleAgent(**self.args)
        agent.log.add("fallback", {}, None, f"LLM недоступна ({why}) — работаю по правилам")
        agent.mode = f"rules (откат с {self.mode})"
        res = agent.run()
        res["fallback"] = why
        return res
