import json
import urllib.error
import urllib.request

import streamlit as st


API_URL = "http://localhost:8000/diagnose"


def call_diagnose_api(symptoms: str) -> dict:
    payload = json.dumps({"symptoms": symptoms}).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="M",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at 10% 10%, rgba(4, 120, 87, 0.18), transparent 34%),
          radial-gradient(circle at 90% 20%, rgba(14, 116, 144, 0.16), transparent 36%),
          linear-gradient(180deg, #f4fafb 0%, #edf6fb 100%);
      }
      .hero {
        background: linear-gradient(130deg, #0f3d56 0%, #0f766e 100%);
        color: #ffffff;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(15, 61, 86, 0.22);
      }
      .result-card {
        background: #ffffff;
        border: 1px solid #d8e7f0;
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 12px;
        box-shadow: 0 4px 12px rgba(15, 36, 61, 0.08);
      }
      .icd-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #0f766e;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.3px;
      }
      .hint {
        color: #486073;
        font-size: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">Медицинский AI-ассистент</h2>
      <p style="margin:8px 0 0 0;">
        Введите анамнез и симптомы пациента. Система вернет вероятные диагнозы и коды МКБ-10.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Параметры")
    st.caption("Endpoint")
    st.code(API_URL)
    st.caption("Запуск")
    st.code(
        "uv run uvicorn app:app --host 0.0.0.0 --port 8000\n"
        "uv run streamlit run streamlit_app.py"
    )

st.markdown('<div class="hint">Опишите жалобы, длительность, сопутствующие симптомы и факторы риска.</div>', unsafe_allow_html=True)
symptoms_input = st.text_area(
    "Анамнез и симптомы",
    height=220,
    placeholder=(
        "Пример: Мужчина, 52 года. Давление до 180/100, головная боль, шум в ушах, "
        "отеки по утрам, утомляемость..."
    ),
)

analyze_clicked = st.button("Анализировать", type="primary", use_container_width=True)

if analyze_clicked:
    symptoms = symptoms_input.strip()
    if not symptoms:
        st.warning("Введите текст симптомов перед анализом.")
    else:
        with st.spinner("Выполняю анализ..."):
            try:
                result = call_diagnose_api(symptoms)
                diagnoses = result.get("diagnoses", [])

                if not diagnoses:
                    st.error("API вернул пустой список диагнозов.")
                else:
                    st.success("Анализ завершен.")
                    for item in diagnoses:
                        rank = item.get("rank", "?")
                        code = item.get("icd10_code", "R69")
                        diagnosis = item.get("diagnosis", "Не указано")
                        explanation = item.get("explanation", "")
                        st.markdown(
                            f"""
                            <div class="result-card">
                              <div><strong>#{rank}. {diagnosis}</strong></div>
                              <div style="margin:8px 0;"><span class="icd-badge">{code}</span></div>
                              <div style="color:#41596b;">{explanation}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with st.expander("Показать raw JSON"):
                    st.json(result)

            except urllib.error.HTTPError as exc:
                st.error(f"HTTP ошибка от API: {exc.code}")
            except urllib.error.URLError as exc:
                st.error(f"Не удалось подключиться к API: {exc.reason}")
            except Exception as exc:
                st.error(f"Внутренняя ошибка: {exc}")
