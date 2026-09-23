"""Реестр турбин (turbines.yaml)."""
from dataclasses import asdict, dataclass

import yaml

from .config import ROOT, TURBINES_FILE


@dataclass
class Turbine:
    id: str
    lat: float
    lon: float
    cap: float = 1.0
    history: str | None = None

    @property
    def history_path(self):
        return ROOT / self.history if self.history else None


def load_turbines(path=TURBINES_FILE) -> list[Turbine]:
    with open(path, encoding="utf-8") as f:
        return [Turbine(**t) for t in yaml.safe_load(f)["turbines"]]


def get_turbine(tid: str) -> Turbine:
    for t in load_turbines():
        if t.id == tid:
            return t
    raise KeyError(f"Турбина {tid} не найдена в {TURBINES_FILE.name}")


def add_turbine(t: Turbine, path=TURBINES_FILE) -> None:
    items = load_turbines(path)
    if any(x.id == t.id for x in items):
        raise ValueError(f"Турбина {t.id} уже есть в реестре")
    if not (-90 <= t.lat <= 90 and -180 <= t.lon <= 180):
        raise ValueError("Некорректные координаты")
    items.append(t)
    rows = [{k: v for k, v in asdict(x).items() if v is not None} for x in items]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Реестр турбин ВЭС. Добавить турбину: `windcast add-turbine --id T3 --lat .. --lon ..`\n")
        f.write("# cap — максимальная нормализованная мощность (потолок), history — CSV со SCADA (если есть).\n")
        yaml.safe_dump({"turbines": rows}, f, allow_unicode=True, sort_keys=False)
