import streamlit as st
import pandas as pd
import numpy as np
from functions import custom_input_prediction  # импортируем свою функцию
from PIL import Image
import pickle

st.set_page_config(page_title="Детектор экстремистов", layout="centered")
st.title("🛡️ Модель по распознованию экстремистких высказываний")

st.markdown("""
Это приложение определяет характер экстремистского или оскорбительного текста.  
Возможные категории:
- **Политический экстремизм**
- **Этническая ненависть**
- **Религиозная ненависть**
- **Прочий экстремизм**
- **Не экстремистский контент**
***
""")

# Ввод текста
st.header("🔍 Введите текст для анализа")
tweet_input = st.text_area("Текст сообщения", height=150)

# Отображение введённого текста
if tweet_input.strip():
    st.subheader("Вы ввели:")
    st.info(tweet_input)
else:
    st.warning("⛔ Пожалуйста, введите текст для анализа.")

st.markdown("***")


# Обработка предсказания
if tweet_input.strip():
    st.header("📊 Результат анализа")
    try:
        prediction = custom_input_prediction(tweet_input)
        st.write(f"Предсказание модели: {prediction}")  # Для отладки

        if prediction is None:
            st.error("⚠️ Не удалось получить результат — проверь модель и векторизатор.")
        elif prediction == "political":
            st.success("🟥 Обнаружен политический экстремизм")
        elif prediction == "ethnicity":
            st.success("🟧 Обнаружена этническая ненависть")
        elif prediction == "religious":
            st.success("🟨 Обнаружена религиозная ненависть")
        elif prediction == "non_extremist":
            st.info("🟩 Экстремизма не обнаружено")
        elif prediction.lower() == "other extremism":
            st.warning("⬜ Обнаружен другой тип экстремизма")
        elif prediction == "Unknown":
            st.error("❓ Неизвестный результат: Обработанный текст не соответствует ни одной из категорий.")
        else:
            st.error(f"❓ Неизвестный результат: {prediction}")

    except Exception as e:
        st.error(f"🚫 Произошла ошибка при анализе текста: {e}")
