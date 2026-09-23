"""Константы проекта: пути, часовой пояс, модели погоды."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
WEATHER_DIR = DATA_DIR / "weather"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
TURBINES_FILE = ROOT / "turbines.yaml"

# Метки времени SCADA — фиксированный UTC+6 (проверено кросс-корреляцией с прогнозами погоды,
# сдвига при переходе Казахстана на UTC+5 01.03.2024 в данных нет). Ввод/вывод — тоже UTC+6.
LOCAL_OFFSET = pd.Timedelta(hours=6)
LOCAL_TZ_LABEL = "UTC+6"

HORIZON_H = 48
# Запас на публикацию прогона погоды: прогон с init-временем X считается доступным с X + 6 ч.
PUBLICATION_DELAY_H = 6

# Open-Meteo Previous Runs API
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
NWP_MODELS = {"gfs": "gfs_seamless", "icon": "icon_seamless", "ecmwf": "ecmwf_ifs025"}
NWP_VARS = [
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_speed_120m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
]
LEAD_DAYS = [1, 2, 3]
WEATHER_START = "2023-03-10"
WEATHER_END = "2026-03-02"
# С этой даты в архиве есть прогнозы с заблаговременностью 1–3 суток (проверено по API).
TRAIN_START = "2024-02-20"


def to_utc(local) -> pd.Timestamp:
    """Наивное время UTC+6 → наивное UTC."""
    return pd.Timestamp(local) - LOCAL_OFFSET


def to_local(utc) -> pd.Timestamp:
    return pd.Timestamp(utc) + LOCAL_OFFSET
