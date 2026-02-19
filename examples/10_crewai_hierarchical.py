"""
Модуль 1.3: Ієрархічний шаблон (Hierarchical Pattern) на CrewAI
Менеджер-агент делегує задачі спеціалізованим worker-агентам,
контролює виконання та збирає результати.

Шаблон: Ієрархічний (Supervisor → Workers)

                      ┌→ [Tech Worker]
  [In] → [Manager] ──┤→ [Edu Worker]    → [Manager] → [Out]
                      └→ [Policy Worker]
"""

import json
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Завантажуємо змінні середовища
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ===========================
# ІНСТРУМЕНТИ ДЛЯ АГЕНТІВ
# ===========================

@tool("Web Search Tool")
def search_web(query: str) -> str:
    """Пошук інформації в інтернеті через DuckDuckGo. Передайте пошуковий запит як рядок."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"- {r['title']}: {r['body'][:150]}...")
        return f"Результати пошуку для '{query}':\n" + "\n".join(results)
    except Exception:
        return f"""Демо результати для '{query}':
- AI в освіті: Персоналізація навчання через штучний інтелект, 85% ВНЗ впроваджують
- Тренди 2025: Адаптивні системи, AI-тьютори, автоматичне оцінювання
- Виклики: Етика, приватність, підготовка кадрів, цифрова нерівність"""

@tool("Data Analyzer")
def analyze_data(text: str) -> str:
    """Аналіз тексту та витягування ключової інформації. Передайте текст для аналізу."""
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')

    keywords = {
        'технології': ['AI', 'штучний інтелект', 'machine learning', 'ML', 'платформ'],
        'освіта': ['навчання', 'студенти', 'університет', 'освіта', 'викладач'],
        'політика': ['стратегія', 'фінансування', 'регуляція', 'закон', 'бюджет']
    }

    found = {}
    text_lower = text.lower()
    for category, words_list in keywords.items():
        count = sum(1 for word in words_list if word.lower() in text_lower)
        if count > 0:
            found[category] = count

    return f"""Аналіз тексту:
- Слів: {words}, Речень: {sentences}
- Ключові теми: {', '.join(found.keys()) if found else 'не виявлено'}
- Деталі: {', '.join([f'{k}({v})' for k, v in found.items()])}"""

@tool("Report Generator")
def generate_report(data: str) -> str:
    """Створення структурованого звіту. Передайте дані для включення у звіт."""
    filename = f"hierarchical_report_{datetime.now():%Y%m%d_%H%M%S}.json"

    report_data = {
        "pattern": "hierarchical",
        "content": data,
        "timestamp": datetime.now().isoformat(),
        "framework": "CrewAI"
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    return f"[OK] Звіт збережено: {filename}\n\nЗміст звіту:\n{data[:500]}"

@tool("Quality Checker")
def check_quality(text: str) -> str:
    """Перевірка якості тексту за критеріями. Передайте текст для перевірки."""
    checks = {
        "Довжина": len(text.split()) >= 50,
        "Структура": any(c in text for c in ['#', '-', '•', '1.', '2.']),
        "Дані": any(w in text.lower() for w in ['%', 'млрд', 'млн', 'статистик']),
        "Висновки": any(w in text.lower() for w in ['висновок', 'рекомендац', 'підсумок']),
    }

    passed = sum(checks.values())
    total = len(checks)

    result = f"Перевірка якості: {passed}/{total}\n"
    for check, ok in checks.items():
        result += f"  {'[OK]' if ok else '[!!]'} {check}\n"

    return result

# ===========================
# СТВОРЕННЯ ІЄРАРХІЧНОЇ КОМАНДИ
# ===========================

def create_hierarchical_team():
    """
    Створення ієрархічної команди:
    Manager (супервайзер) → Workers (спеціалісти)
    """

    # === MANAGER (Супервайзер) ===
    manager = Agent(
        role='Керівник Дослідницького Проекту',
        goal='Координувати роботу команди дослідників та забезпечити якісний результат',
        backstory="""Ви - досвідчений керівник дослідницьких проектів з 20-річним стажем.
        Ваша роль - розподіляти завдання між спеціалістами, контролювати якість
        та синтезувати результати в єдиний звіт.
        Ви делегуєте конкретні аспекти дослідження відповідним спеціалістам.""",
        tools=[check_quality, generate_report],
        verbose=True,
        max_iter=3,
        allow_delegation=True  # Може делегувати задачі іншим агентам
    )

    # === WORKERS (Спеціалісти) ===

    # Worker 1: Технічний дослідник
    tech_worker = Agent(
        role='Технічний Дослідник AI',
        goal='Дослідити технологічні аспекти AI в освіті: платформи, інструменти, архітектуру',
        backstory="""Ви - спеціаліст з AI технологій в EdTech з досвідом роботи
        в провідних технологічних компаніях. Знаєте всі сучасні AI-платформи
        для освіти та розумієте їх технічну архітектуру.""",
        tools=[search_web, analyze_data],
        verbose=True,
        max_iter=3
    )

    # Worker 2: Освітній експерт
    edu_worker = Agent(
        role='Освітній Методист',
        goal='Дослідити педагогічні аспекти: методики, студентський досвід, ефективність',
        backstory="""Ви - провідний освітній методист з PhD в педагогічних науках.
        Спеціалізуєтесь на інтеграції технологій у навчальний процес.
        Розумієте як AI змінює педагогічні підходи.""",
        tools=[search_web, analyze_data],
        verbose=True,
        max_iter=3
    )

    # Worker 3: Аналітик політик
    policy_worker = Agent(
        role='Аналітик Освітніх Політик',
        goal='Дослідити регуляторне середовище: стратегії, фінансування, стандарти',
        backstory="""Ви - аналітик з МОН України з глибоким розумінням
        освітньої політики та регуляцій. Відстежуєте законодавчі ініціативи
        та міжнародний досвід у сфері AI в освіті.""",
        tools=[search_web, analyze_data],
        verbose=True,
        max_iter=3
    )

    return manager, tech_worker, edu_worker, policy_worker

# ===========================
# СТВОРЕННЯ ЗАДАЧ
# ===========================

def create_hierarchical_tasks(manager, tech_worker, edu_worker, policy_worker, topic):
    """Створення задач для ієрархічної команди"""

    # Worker задачі (виконуються спеціалістами під контролем менеджера)

    tech_task = Task(
        description=f"""
        Проведіть технічне дослідження на тему: {topic}

        Використайте Web Search Tool для пошуку:
        1. Сучасних AI платформ для освіти
        2. Технічних рішень та архітектур
        3. Порівняння інструментів

        Після пошуку використайте Data Analyzer для аналізу.
        Створіть структурований технічний огляд з конкретними даними.
        """,
        expected_output="Технічний огляд AI платформ та інструментів з аналізом",
        agent=tech_worker
    )

    edu_task = Task(
        description=f"""
        Проведіть педагогічне дослідження на тему: {topic}

        Використайте Web Search Tool для пошуку:
        1. Ефективних методик навчання з AI
        2. Студентського досвіду та результатів
        3. Найкращих практик інтеграції AI

        Після пошуку використайте Data Analyzer для аналізу.
        Створіть огляд з конкретними прикладами та метриками.
        """,
        expected_output="Педагогічний огляд з методиками та результатами",
        agent=edu_worker,
        context=[tech_task]  # Може використовувати технічний контекст
    )

    policy_task = Task(
        description=f"""
        Проведіть дослідження політик на тему: {topic}

        Використайте Web Search Tool для пошуку:
        1. Державних стратегій та програм
        2. Фінансування та бюджетів
        3. Міжнародного досвіду регуляцій

        Після пошуку використайте Data Analyzer для аналізу.
        Створіть огляд регуляторного середовища з порівняннями.
        """,
        expected_output="Огляд політик та регуляцій з міжнародними порівняннями",
        agent=policy_worker,
        context=[tech_task, edu_task]
    )

    # Задача менеджера: синтез та фінальний звіт
    synthesis_task = Task(
        description=f"""
        Як керівник проекту, синтезуйте результати всіх трьох спеціалістів:
        1. Технічного дослідника - AI платформи та інструменти
        2. Освітнього методиста - педагогічні методики та результати
        3. Аналітика політик - регуляторне середовище

        Використайте Quality Checker для перевірки якості.
        Потім використайте Report Generator для збереження.

        Створіть executive summary з:
        - Ключовими знахідками кожного напрямку
        - Перехресними інсайтами
        - Стратегічними рекомендаціями
        """,
        expected_output="Комплексний синтезований звіт з рекомендаціями",
        agent=manager,
        context=[tech_task, edu_task, policy_task]  # Залежить від всіх worker-задач
    )

    return [tech_task, edu_task, policy_task, synthesis_task]

# ===========================
# ГОЛОВНА ФУНКЦІЯ
# ===========================

def main():
    """Запуск ієрархічної мультиагентної системи"""

    print("""
╔══════════════════════════════════════════════════════════════╗
║     CREWAI HIERARCHICAL PATTERN                              ║
║     Ієрархічний шаблон: Manager → Workers                    ║
╚══════════════════════════════════════════════════════════════╝

                      ┌→ [Tech Worker]
  [In] → [Manager] ──┤→ [Edu Worker]    → [Manager] → [Out]
                      └→ [Policy Worker]
    """)

    topic = "Штучний інтелект в освіті України 2025: можливості та виклики"
    print(f"[TOPIC] Тема: {topic}")
    print("="*60)

    # Створюємо ієрархічну команду
    print("\n[TEAM] Формування ієрархічної команди...")
    manager, tech_worker, edu_worker, policy_worker = create_hierarchical_team()
    print("[OK] Команда готова:")
    print("   [Manager] Керівник Дослідницького Проекту")
    print("   [Worker]  Технічний Дослідник AI")
    print("   [Worker]  Освітній Методист")
    print("   [Worker]  Аналітик Освітніх Політик")

    # Створюємо задачі
    print("\n[TASKS] Створення ієрархічних задач...")
    tasks = create_hierarchical_tasks(manager, tech_worker, edu_worker, policy_worker, topic)
    print(f"[OK] Створено {len(tasks)} задачі (3 worker + 1 manager)")

    # Формуємо crew з ієрархічним процесом
    print("\n[CREW] Запуск CrewAI в ієрархічному режимі...")
    crew = Crew(
        agents=[manager, tech_worker, edu_worker, policy_worker],
        tasks=tasks,
        process=Process.hierarchical,  # ІЄРАРХІЧНИЙ процес (ключова відмінність!)
        manager_agent=manager,         # Явно вказуємо менеджера
        verbose=True,
        memory=True,
        cache=True,
        max_rpm=10
    )

    print("\n[WORKING] Ієрархічна команда працює...")
    print("   Manager делегує завдання Workers...")
    print("-"*60)

    try:
        result = crew.kickoff()

        print("\n" + "="*60)
        print("[SUCCESS] ІЄРАРХІЧНЕ ДОСЛІДЖЕННЯ ЗАВЕРШЕНО!")
        print("="*60)

        print("\n[RESULT] Результат:")
        print(str(result)[:500] + "...")

        # Зберігаємо фінальний результат
        final_report = {
            "pattern": "hierarchical",
            "topic": topic,
            "result": str(result),
            "agents": {
                "manager": "Керівник Дослідницького Проекту",
                "workers": [
                    "Технічний Дослідник AI",
                    "Освітній Методист",
                    "Аналітик Освітніх Політик"
                ]
            },
            "tasks_count": len(tasks),
            "timestamp": datetime.now().isoformat()
        }

        filename = f"crewai_hierarchical_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        print(f"\n[SAVED] Звіт: {filename}")

    except Exception as e:
        print(f"\n[ERROR] Помилка: {e}")
        print("\nПідказки:")
        print("1. Перевірте OPENAI_API_KEY в .env файлі")
        print("2. CrewAI потребує API ключ (немає демо режиму)")
        print("3. Встановіть: pip install crewai crewai-tools")

    print("\n" + "="*60)
    print("Навчальні поради (Ієрархічний шаблон):")
    print("- Process.hierarchical дозволяє Manager делегувати задачі")
    print("- allow_delegation=True дає агенту право передавати роботу")
    print("- context=[] визначає залежності між задачами")
    print("- Порівняйте з Process.sequential (04_crewai_agents.py)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nРоботу перервано")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Критична помилка: {e}")
        import traceback
        traceback.print_exc()
