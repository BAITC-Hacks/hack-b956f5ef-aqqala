"""Агент прогноза: инструменты + оркестратор (правила или LLM)."""


def make_agent(kind: str, issue_local, **kw):
    """kind: 'rules' — детерминированные правила; 'llm' — LLM выбирает инструменты (нужен ключ API)."""
    if kind == "llm":
        from .llm import LLMAgent
        return LLMAgent(issue_local, **kw)
    from .loop import RuleAgent
    return RuleAgent(issue_local, **kw)
