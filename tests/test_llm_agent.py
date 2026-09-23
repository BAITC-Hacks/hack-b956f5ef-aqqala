"""LLM-агент без сети: подменяем клиента сценарием вызовов инструментов."""
import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from windcast.model import MODEL_FILE, WindModel

ISSUE = pd.Timestamp("2026-02-12 10:00")


class FakeMessage(NS):
    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in self.tool_calls or []]}


def fake_client(script):
    calls = iter(script)

    def create(**kw):
        name, args = next(calls)
        tc = NS(id=f"c{name}", function=NS(name=name, arguments=json.dumps({**args, "reason": f"тест {name}"})))
        return NS(choices=[NS(message=FakeMessage(tool_calls=[tc]))],
                  usage=NS(prompt_tokens=100, completion_tokens=10))
    return NS(chat=NS(completions=NS(create=create)))


pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="сначала windcast train")


def test_llm_agent_scripted_run(tmp_path, monkeypatch):
    from windcast.agent.llm import LLMAgent
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    script = [("check_weather", {}), ("validate_inputs", {}), ("check_previous_version", {}),
              ("run_forecast", {"nwp_models": ["gfs", "icon", "ecmwf"]}), ("analyze_forecast", {}),
              ("finish", {"summary": "Тестовая сводка."})]
    agent = LLMAgent(ISSUE, runs_dir=tmp_path, online=False, model=WindModel.load())
    monkeypatch.setattr(agent, "_client", lambda: fake_client(script))
    res = agent.run()
    assert res["status"] == "ok" and "fallback" not in res
    assert "Тестовая сводка." in res["report"]
    log = [json.loads(line) for line in open(tmp_path / ISSUE.strftime("%Y%m%d_%H%M") / "agent_log.jsonl")]
    assert [s["tool"] for s in log][:4] == ["check_weather", "validate_inputs", "check_previous_version", "run_forecast"]
    assert all(s["reason"] for s in log)


def test_llm_agent_falls_back_without_key(tmp_path, monkeypatch):
    from windcast.agent.llm import LLMAgent
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    res = LLMAgent(ISSUE, runs_dir=tmp_path, online=False, model=WindModel.load()).run()
    assert res["status"] == "ok" and "OPENAI_API_KEY" in res["fallback"]
