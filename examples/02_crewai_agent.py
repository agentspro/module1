"""
CrewAI Multi-Agent System - БЕЗ LangChain
Демонстрація роботи команди агентів
"""

import os
import json
from datetime import datetime
from typing import Dict, List
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Завантажуємо змінні середовища
from dotenv import load_dotenv
load_dotenv()

# ===========================
# ІНСТРУМЕНТИ ДЛЯ АГЕНТІВ
# ===========================

@tool("Web Search Tool")
def search_web(query: str) -> str:
    """Пошук інформації в інтернеті через DuckDuckGo"""
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"- {r['title']}: {r['body'][:150]}...")
        
        return f"Результати пошуку для '{query}':\n" + "\n".join(results)
    except:
        # Демо результати якщо пошук не працює
        return f"""
        Демо результати для '{query}':
        - AI в освіті: Персоналізація навчання через штучний інтелект
        - Тренди 2025: 85% університетів використовують AI
        - Виклики: Етика та приватність в AI системах
        """

@tool("Data Analyzer")
def analyze_data(text: str) -> str:
    """Аналіз тексту та витягування ключової інформації"""
    # Підрахунок статистики
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    # Пошук ключових слів
    keywords = {
        'технології': ['AI', 'штучний інтелект', 'machine learning', 'ML'],
        'освіта': ['навчання', 'студенти', 'університет', 'освіта'],
        'тренди': ['тренд', 'майбутнє', '2025', '2024', 'інновація']
    }
    
    found_keywords = {}
    text_lower = text.lower()
    
    for category, words_list in keywords.items():
        count = sum(1 for word in words_list if word.lower() in text_lower)
        if count > 0:
            found_keywords[category] = count
    
    analysis = f"""
    Аналіз тексту:
    - Слів: {words}
    - Речень: {sentences}
    - Ключові теми: {', '.join(found_keywords.keys()) if found_keywords else 'не виявлено'}
    """
    
    return analysis

@tool("Report Generator")
def generate_report(data: str, filename: str = None) -> str:
    """Створення та збереження звіту"""
    if not filename:
        filename = f"crewai_report_{datetime.now():%Y%m%d_%H%M%S}.md"
    
    report_content = f"""
# Звіт CrewAI
**Дата:** {datetime.now():%Y-%m-%d %H:%M}

## Зміст
{data}

---
*Згенеровано CrewAI Multi-Agent System*
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return f"[OK] Звіт збережено: {filename}"

# ===========================
# СТВОРЕННЯ АГЕНТІВ
# ===========================

def create_research_team():
    """Створення команди агентів для дослідження"""
    
    # Агент 1: Дослідник
    researcher = Agent(
        role='Головний Дослідник',
        goal='Знайти найактуальнішу інформацію про AI в освіті',
        backstory="""Ви - досвідчений дослідник з 15-річним стажем.
        Спеціалізуєтесь на освітніх технологіях та штучному інтелекті.
        Ваша сильна сторона - знаходження найсвіжіших даних та трендів.""",
        tools=[search_web],
        verbose=True,
        max_iter=3
    )
    
    # Агент 2: Аналітик
    analyst = Agent(
        role='Старший Аналітик',
        goal='Проаналізувати зібрані дані та виявити ключові інсайти',
        backstory="""Ви - експерт з data science та аналізу трендів.
        Маєте унікальну здатність знаходити приховані патерни в даних.
        Ваші аналітичні звіти завжди точні та корисні.""",
        tools=[analyze_data],
        verbose=True,
        max_iter=3
    )
    
    # Агент 3: Репортер
    reporter = Agent(
        role='Технічний Письменник',
        goal='Створити чіткий та зрозумілий звіт для широкої аудиторії',
        backstory="""Ви - професійний технічний письменник.
        Вмієте перетворювати складні технічні дані на зрозумілі звіти.
        Ваші звіти читають і технічні спеціалісти, і звичайні користувачі.""",
        tools=[generate_report],
        verbose=True,
        max_iter=2
    )
    
    return researcher, analyst, reporter

# ===========================
# СТВОРЕННЯ ЗАДАЧ
# ===========================

def create_research_tasks(researcher, analyst, reporter, topic):
    """Створення задач для команди"""
    
    # Задача 1: Дослідження
    research_task = Task(
        description=f"""
        Проведіть глибоке дослідження на тему: {topic}
        
        Використайте Web Search Tool для пошуку:
        1. Останніх новин та статей
        2. Статистики та фактів
        3. Думок експертів
        
        Зберіть мінімум 5 ключових фактів.
        """,
        expected_output="Детальний список знайденої інформації з джерелами",
        agent=researcher
    )
    
    # Задача 2: Аналіз
    analysis_task = Task(
        description="""
        Проаналізуйте зібрану інформацію від дослідника.
        
        Використайте Data Analyzer для:
        1. Виявлення ключових тем
        2. Підрахунку статистики
        3. Визначення трендів
        
        Створіть структурований аналіз з висновками.
        """,
        expected_output="Аналітичний висновок з ключовими інсайтами",
        agent=analyst,
        context=[research_task]  # Залежить від результатів дослідження
    )
    
    # Задача 3: Звіт
    report_task = Task(
        description="""
        Створіть фінальний звіт на основі дослідження та аналізу.
        
        Використайте Report Generator для:
        1. Форматування результатів
        2. Створення структурованого документа
        3. Збереження у файл
        
        Звіт має бути зрозумілим для всіх рівнів читачів.
        """,
        expected_output="Професійний звіт збережений у файл",
        agent=reporter,
        context=[research_task, analysis_task]  # Залежить від обох попередніх
    )
    
    return [research_task, analysis_task, report_task]

# ===========================
# ГОЛОВНА ФУНКЦІЯ
# ===========================

def main():
    """Запуск мультиагентної системи CrewAI"""
    
    print("""
    ============================================================
            CREWAI MULTI-AGENT SYSTEM v2.0                    
                    (БЕЗ LANGCHAIN)                           
    ============================================================
    """)
    
    # Тема дослідження
    topic = "Штучний інтелект в освіті України 2025: можливості та виклики"
    
    print(f"\n[TOPIC] Тема дослідження: {topic}")
    print("=" * 60)
    
    # Створюємо команду
    print("\n[TEAM] Формування команди агентів...")
    researcher, analyst, reporter = create_research_team()
    print("[OK] Команда готова:")
    print("   - Головний Дослідник")
    print("   - Старший Аналітик")
    print("   - Технічний Письменник")
    
    # Створюємо задачі
    print("\n[TASKS] Створення задач...")
    tasks = create_research_tasks(researcher, analyst, reporter, topic)
    print(f"[OK] Створено {len(tasks)} задачі")
    
    # Формуємо екіпаж
    print("\n[CREW] Запуск CrewAI...")
    crew = Crew(
        agents=[researcher, analyst, reporter],
        tasks=tasks,
        process=Process.sequential,  # Послідовне виконання
        verbose=True,  # Показувати деталі роботи
        memory=True,   # Використовувати пам'ять між задачами
        cache=True,    # Кешувати результати
        max_rpm=10     # Обмеження запитів
    )
    
    print("\n[WORKING] Команда працює...")
    print("-" * 60)
    
    try:
        # Запускаємо роботу команди
        result = crew.kickoff()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ДОСЛІДЖЕННЯ ЗАВЕРШЕНО!")
        print("=" * 60)
        
        print("\n[RESULT] Результат:")
        print(str(result)[:500] + "...")
        
        # Зберігаємо фінальний результат
        final_report = {
            "topic": topic,
            "result": str(result),
            "agents_count": 3,
            "tasks_count": len(tasks),
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"crewai_final_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n[SAVED] Фінальний звіт: {filename}")
        print("[INFO] Також перевірте згенеровані .md файли")
        
    except Exception as e:
        print(f"\n[ERROR] Помилка: {e}")
        print("\nПідказки:")
        print("1. Перевірте OPENAI_API_KEY в .env файлі")
        print("2. Встановіть: pip install crewai crewai-tools")
        print("3. Спробуйте зменшити max_iter для агентів")
    
    print("\n" + "=" * 60)
    print("Навчальні поради:")
    print("- Змініть ролі агентів в create_research_team()")
    print("- Додайте нові інструменти через @tool декоратор")
    print("- Спробуйте Process.hierarchical для ієрархічної роботи")
    print("- Експериментуйте з context між задачами")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nРоботу перервано")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Критична помилка: {e}")
        import traceback
        traceback.print_exc()
