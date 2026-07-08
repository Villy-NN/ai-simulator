# Импортируем библиотеку Streamlit для создания веб-интерфейса
import streamlit as st
# Импортируем библиотеку для работы с API Google Gemini
import google.generativeai as genai
# Импортируем requests для выполнения HTTP-запросов (скачивание страниц)
import requests
# Импортируем BeautifulSoup для парсинга HTML-кода
from bs4 import BeautifulSoup
# Импортируем модуль json для работы с JSON-данными
import json
# Импортируем Typing для аннотации типов (хороший тон в архитектуре)
from typing import Dict, Any

# ==========================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ И ДИЗАЙН (CSS)
# ==========================================

# Настраиваем базовые параметры страницы (название, иконка, на всю ширину)
st.set_page_config(page_title="AI A/B Test Simulator", page_icon="📈", layout="wide")

# Задаем кастомный CSS для создания строгого корпоративного стиля (синие/серые тона)
# Скрываем дефолтное меню (MainMenu) и футер (footer) Streamlit для B2B-вида
CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {background-color: #f8f9fa;}
    h1, h2, h3 {color: #1e3a5f; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;}
    .stButton>button {
        background-color: #1e3a5f; 
        color: white; 
        border-radius: 4px; 
        padding: 0.5rem 1rem; 
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {background-color: #112238;}
    div[data-testid="stMetricValue"] {color: #1e3a5f; font-size: 4rem; font-weight: 800;}
    .insight-card {
        background-color: white; 
        border: 1px solid #dee2e6; 
        border-radius: 8px; 
        padding: 20px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
</style>
"""
# Применяем CSS стили к нашему приложению через HTML
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ПАРСИНГ И МАТЕМАТИКА)
# ==========================================

def fetch_url_content(url: str) -> str:
    """Функция для извлечения текстового контента с переданного URL."""
    # Блок try...except для перехвата любых ошибок сети или парсинга
    try:
        # Отправляем GET-запрос с таймаутом 10 секунд и поддельным User-Agent, чтобы не блокировали
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        # Проверяем, что сервер вернул успешный статус-код (200 OK)
        response.raise_for_status()
        # Инициализируем BeautifulSoup для разбора HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # Удаляем теги скриптов и стилей, так как в них нет полезного текста
        for script_or_style in soup(["script", "style", "nav", "footer"]):
            script_or_style.extract()
        # Извлекаем текст, разделяя блоки пробелами, и очищаем от лишних пробелов по краям
        text = soup.get_text(separator=' ', strip=True)
        # Возвращаем полученный текст
        return text
    # Обрабатываем ошибки таймаута
    except requests.exceptions.Timeout:
        raise Exception("Ошибка: Время ожидания ответа от сайта истекло (Timeout).")
    # Обрабатываем остальные ошибки requests
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка при доступе к сайту: {e}")
    # Обрабатываем непредвиденные ошибки
    except Exception as e:
        raise Exception(f"Критическая ошибка парсинга: {e}")

def calculate_lift_score(metrics: Dict[str, int]) -> int:
    """Детерминированная функция расчета LIFT Score на основе оценок нейросети."""
    # Извлекаем оценки из словаря
    value = metrics.get('value', 5)
    relevance = metrics.get('relevance', 5)
    clarity = metrics.get('clarity', 5)
    friction = metrics.get('friction', 5)
    anxiety = metrics.get('anxiety', 5)
    
    # Считаем базовый потенциал (Base) по заданной в ТЗ формуле
    base = (value * 0.4) + (relevance * 0.3) + (clarity * 0.3)
    # Считаем штрафные баллы (Penalty) по заданной в ТЗ формуле
    penalty = (friction * 1.5) + (anxiety * 1.5)
    # Считаем сырой итоговый балл (Score)
    raw_score = (base * 10) - penalty
    
    # Ограничиваем балл в диапазоне от 1 до 100 с помощью max/min и приводим к целому числу
    final_score = max(1, min(100, int(raw_score)))
    # Возвращаем итоговый балл
    return final_score

# ==========================================
# 3. ИНТЕГРАЦИЯ С GOOGLE GEMINI API
# ==========================================

def run_ai_simulation(api_key: str, text_content: str, personas: str) -> Dict[str, Any]:
    """Функция для обращения к Gemini API и получения JSON-ответа."""
    # Инициализируем API-клиент Google Gemini с переданным ключом
    genai.configure(api_key=api_key)
    
    # Формируем системный промпт (инструкцию для нейросети) с жестким B2B-тоном
    system_instruction = """
    Ты — циничный, прагматичный и ориентированный на ROI Chief Marketing Officer (CMO). 
    Твоя задача — оценить маркетинговый текст/лендинг глазами указанных персон.
    Тебя интересует только одно: 'Заставит ли этот текст пользователя совершить целевое действие и принесет ли он деньги бизнесу?'
    
    Оцени метрики (от 1 до 10, где 10 - максимум):
    - value (ценность оффера)
    - relevance (релевантность аудитории)
    - clarity (понятность, отсутствие воды)
    - friction (трение: сложность интерфейса/призыва к действию, 10 - ужасное трение)
    - anxiety (тревожность: вызывает ли оффер сомнения, 10 - максимальное недоверие)
    
    ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ СТРОГО В ФОРМАТЕ JSON. НИКАКОГО ДОПОЛНИТЕЛЬНОГО ТЕКСТА.
    """
    
    # Создаем конфигурацию генерации, принудительно устанавливая формат ответа JSON
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.4 # Низкая температура для более предсказуемого и аналитического ответа
    )
    
    # Инициализируем модель (используем быструю и дешевую flash-модель, идеальную для JSON)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
        generation_config=generation_config
    )
    
    # Формируем структуру JSON, которую ожидаем получить (передаем в промпт для надежности)
    json_schema = """
    {
      "metrics": {
        "value": int,
        "relevance": int,
        "clarity": int,
        "friction": int,
        "anxiety": int
      },
      "persona_feedback": [
        {"persona": "Имя персоны", "feedback": "Жесткий отзыв с упором на конверсию (2-3 предложения)"}
      ],
      "insights": [
        "Конкретный бизнес-совет 1",
        "Конкретный бизнес-совет 2",
        "Конкретный бизнес-совет 3"
      ]
    }
    """
    
    # Составляем финальный промпт пользователя с контентом и персонами
    prompt = f"""
    Оцени следующий материал от лица этих персон: {personas}.
    
    МАТЕРИАЛ ДЛЯ ОЦЕНКИ:
    {text_content}
    
    Верни результат, СТРОГО соответствующий этой JSON схеме:
    {json_schema}
    """
    
    # Отправляем запрос к модели
    response = model.generate_content(prompt)
    
    # Парсим текстовый ответ модели в Python-словарь (JSON)
    result_dict = json.loads(response.text)
    # Возвращаем готовый словарь
    return result_dict

# ==========================================
# 4. ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС (UI/UX)
# ==========================================

# Основной заголовок приложения на главном экране
st.title("📈 AI A/B Test Simulator")
# Подзаголовок, объясняющий суть инструмента
st.markdown("**Виртуальная Фокус-Группа для оценки Landing Pages & Ad Copy**")
# Визуальный разделитель
st.divider()

# --- БОКОВАЯ ПАНЕЛЬ (SIDEBAR) ---
# Создаем сайдбар для настроек
with st.sidebar:
    st.header("⚙️ Конфигурация")
    
    # Незаметно достаем ключ из файла секретов .streamlit/secrets.toml
    # Если файла нет, приложение вежливо попросит его создать и остановится
    api_key_input = st.secrets.get("GEMINI_API_KEY")
    if not api_key_input:
        st.error("🚨 Ключ API не найден в скрытом файле .streamlit/secrets.toml")
        st.stop()
    
    # Выпадающий список для выбора источника данных
    source_type = st.selectbox(
        "Источник данных", 
        ["Текст объявления", "Ссылка на лендинг (URL)", "Описание продукта"]
    )
    
    # Логика отображения полей ввода в зависимости от выбранного источника
    if source_type == "Ссылка на лендинг (URL)":
        input_data = st.text_input("Введите URL (начиная с http/https):")
    else:
        input_data = st.text_area(f"Введите {source_type.lower()}:", height=150)
    
    # Радио-кнопка для выбора режима настройки персон
    persona_mode = st.radio(
        "Настройка ИИ-персон",
        ["Использовать стандартные персоны", "Задать свои персоны"]
    )
    
    active_personas = "Скептик, Лояльный покупатель, Экономный скряга, Рационализатор"
    
    if persona_mode == "Задать свои персоны":
        active_personas = st.text_input("Введите персоны (через запятую):", "Инвестор в крипту, Студент")
        
    run_button = st.button("🚀 Запустить симуляцию", type="primary", use_container_width=True)

# --- ГЛАВНАЯ ОБЛАСТЬ (ОБРАБОТКА И ВЫВОД) ---
# Проверяем, нажал ли пользователь на кнопку запуска
if run_button:
    # Валидация: проверяем наличие API ключа
    if not api_key_input:
        st.error("Пожалуйста, введите API ключ Google Gemini в боковой панели.")
    # Валидация: проверяем наличие входных данных (текста или URL)
    elif not input_data:
        st.warning("Пожалуйста, предоставьте данные для анализа (текст или URL).")
    else:
        # Показываем спиннер (индикатор загрузки), пока идет обработка
        with st.spinner("🧠 ИИ анализирует контент. Проводим симуляцию..."):
            try:
                # Переменная для хранения итогового текста, который пойдет в ИИ
                content_to_analyze = ""
                
                # Если выбран URL, запускаем парсинг
                if source_type == "Ссылка на лендинг (URL)":
                    st.info("Скачиваем и парсим контент сайта...")
                    content_to_analyze = fetch_url_content(input_data)
                else:
                    # Если текст, просто присваиваем его переменной
                    content_to_analyze = input_data
                
                # Обращаемся к Gemini API через нашу функцию-обертку
                ai_result = run_ai_simulation(api_key=api_key_input, text_content=content_to_analyze, personas=active_personas)
                
                # Вызываем математическую функцию для расчета итогового балла
                lift_score = calculate_lift_score(ai_result['metrics'])
                
                # Очищаем экран от спиннера и инфо-сообщений, выводим успешный статус
                st.success("Симуляция успешно завершена!")
                
                # --- ОТРИСОВКА РЕЗУЛЬТАТОВ ---
                
                # Разделяем экран на две колонки: для метрики и для инсайтов
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Рисуем красивую карточку с помощью HTML/CSS (class="insight-card")
                    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                    st.subheader("Conversion LIFT Score")
                    # Выводим главный KPI крупным шрифтом с помощью st.metric
                    st.metric(label="Индекс вероятности конверсии", value=f"{lift_score} / 100")
                    st.caption("Рассчитано на основе метрик: Привлекательность оффера, Попадание в аудиторию, Ясность сообщения, Барьеры на пути, Факторы недоверия.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Выводим сырые оценки в виде таблицы для аналитиков
                    st.markdown("**Детальная оценка ИИ (из 10):**")

                    # Переводим ключи на понятный бизнес-язык для визуального отображения
                    translated_metrics = {
                        "Привлекательность оффера": ai_result['metrics'].get('value', 0),
                        "Попадание в аудиторию": ai_result['metrics'].get('relevance', 0),
                        "Ясность сообщения": ai_result['metrics'].get('clarity', 0),
                        "Барьеры на пути (сложность)": ai_result['metrics'].get('friction', 0),
                        "Факторы недоверия (сомнения)": ai_result['metrics'].get('anxiety', 0)
                    }
                    st.json(translated_metrics)
                
                with col2:
                    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                    st.subheader("💡 Actionable Insights (План действий)")
                    # Проходимся циклом по инсайтам и выводим их как маркированный список
                    for insight in ai_result.get('insights', []):
                        st.markdown(f"- {insight}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.divider()
                st.subheader("🗣️ Отзывы Виртуальной Фокус-Группы")
                
                # Проходимся циклом по отзывам персон
                for p_data in ai_result.get('persona_feedback', []):
                    # Используем st.expander (раскрывающийся блок) для каждой персоны
                    with st.expander(f"Персона: {p_data.get('persona', 'Аноним')}", expanded=True):
                        # Выводим сам отзыв
                        st.write(p_data.get('feedback', 'Нет отзыва.'))
                        
            # Блок перехвата любых ошибок в процессе (парсинг, сеть, API, JSON)
            except Exception as e:
                # В случае ошибки выводим красное сообщение, не ломая интерфейс
                st.error(f"🚨 Произошла ошибка во время симуляции: {str(e)}")
else:
    # Состояние покоя (до нажатия кнопки): выводим инструкцию для пользователя
    st.info("👈 Настройте параметры в боковой панели и запустите симуляцию.")