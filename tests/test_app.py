import pytest
from streamlit.testing.v1 import AppTest

from windcast.model import MODEL_FILE


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="сначала windcast train")
def test_app_renders_and_forecasts():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
    at.button(key="run").click().run()  # «Сформировать прогноз» (правила, 01.02.2026 10:00)
    assert not at.exception
    assert any("Итог" in m.value for m in at.markdown)
