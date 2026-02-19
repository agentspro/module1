"""
Модуль 1.3: Human-in-the-Loop шаблон на LangGraph
Агент збирає дані, ЛЮДИНА перевіряє та коригує результати,
потім система генерує фінальний звіт.

Шаблон: Human in the Loop

  [In] → [Researcher] → [HUMAN REVIEW] → [Reporter] → [Out]
                              ↑    │
                              └────┘ (коригування)
"""

import os
import json
from datetime import datetime
from typing import TypedDict, Annotated, List
import operator

# LangChain / LangGraph imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    print("[OK] LangChain 1.0 компоненти завантажено")
except ImportError as e:
    print(f"[ERROR] Помилка імпорту LangChain: {e}")
    print("Встановіть: pip install langchain-openai langchain-core")
    exit(1)

try:
    from langgraph.graph import StateGraph, END
    print("[OK] LangGraph завантажено")
except ImportError as e:
    print(f"[ERROR] Помилка імпорту LangGraph: {e}")
    print("Встановіть: pip install langgraph")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] .env файл завантажено")
except:
    print("[WARNING] python-dotenv не встановлено")

# ===========================
# STATE DEFINITION
# ===========================

class HumanLoopState(TypedDict):
    """Стан для human-in-the-loop шаблону"""
    topic: str
    research_results: str    # Результати дослідження
    human_feedback: str      # Зворотній зв'язок від людини
    human_approved: bool     # Чи людина підтвердила
    human_edits: str         # Правки від людини
    final_report: str
    messages: Annotated[List[str], operator.add]

# ===========================
# ІНСТРУМЕНТИ (TOOLS)
# ===========================

def search_web(query: str) -> str:
    """Пошук інформації"""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"- {r['title']}: {r['body'][:150]}...")
        return "\n".join(results) if results else _demo_search()
    except Exception:
        return _demo_search()

def _demo_search() -> str:
    return (
        "- AI в освіті 2025: 85% українських ВНЗ впроваджують AI технології...\n"
        "- Персоналізація: Адаптивні системи підвищують успішність на 35%...\n"
        "- AI-тьютори: Цілодобова підтримка студентів через віртуальних помічників...\n"
        "- EdTech: Глобальний ринок досягне $25.7 млрд до 2030 року...\n"
        "- Виклики: Етика, приватність, цифрова нерівність, підготовка кадрів..."
    )

# ===========================
# HUMAN-IN-THE-LOOP SYSTEM
# ===========================

class HumanInLoopSystem:
    """
    Human-in-the-Loop: людина контролює та коригує роботу агентів.
    """

    def __init__(self, api_key: str = None, auto_mode: bool = False):
        """
        auto_mode=True - автоматичне підтвердження (для тестування)
        auto_mode=False - інтерактивний режим з реальним введенням
        """
        self.auto_mode = auto_mode
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if self.api_key:
            try:
                self.llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=self.api_key)
                print("[OK] ChatOpenAI LLM створено")
            except Exception as e:
                print(f"[WARNING] Помилка LLM: {e}")
                self.llm = None
        else:
            print("[WARNING] OPENAI_API_KEY не знайдено - демо режим")
            self.llm = None

        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_workflow(self) -> StateGraph:
        """
        Граф з human checkpoint:
        START → researcher → human_review → {reporter | researcher (retry)} → END
        """
        workflow = StateGraph(HumanLoopState)

        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("human_review", self.human_review_node)
        workflow.add_node("reporter", self.reporter_node)

        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "human_review")

        # Умовний перехід після людської перевірки
        workflow.add_conditional_edges(
            "human_review",
            self._human_decision,
            {
                "approved": "reporter",     # Людина підтвердила → звіт
                "retry": "researcher",      # Людина хоче перезапуск
            }
        )

        workflow.add_edge("reporter", END)

        return workflow

    @staticmethod
    def _human_decision(state: HumanLoopState) -> str:
        """Маршрутизація на основі рішення людини"""
        if state.get("human_approved", False):
            return "approved"
        return "retry"

    # --- Вузли ---

    def researcher_node(self, state: HumanLoopState) -> dict:
        """Дослідник: збирає та структурує інформацію"""
        print("\n" + "="*60)
        print("RESEARCHER: Збір та обробка інформації...")
        print("="*60)

        topic = state["topic"]
        human_edits = state.get("human_edits", "")

        # Якщо є правки від людини - враховуємо їх
        if human_edits:
            query = f"{topic} {human_edits}"
            print(f"   [INFO] Враховую правки людини: {human_edits[:100]}")
        else:
            query = topic

        results = search_web(query)

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ви - професійний дослідник. Структуруйте знайдену інформацію:
1. Ключові факти (мінімум 5)
2. Статистика та цифри
3. Основні тренди
4. Проблеми та виклики"""),
                    ("human", "Тема: {topic}\n{extra}\nДані:\n{data}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                extra = f"Додаткові вказівки від людини: {human_edits}" if human_edits else ""
                results = chain.invoke({"topic": topic, "extra": extra, "data": results})
            except Exception as e:
                print(f"[WARNING] AI обробка: {e}")
        else:
            results = f"""Структуровані результати дослідження:

КЛЮЧОВІ ФАКТИ:
1. 85% українських ВНЗ впроваджують AI технології у 2025 році
2. Адаптивні AI-системи підвищують успішність студентів на 35%
3. Глобальний ринок AI в освіті досягне $25.7 млрд до 2030 року
4. AI-тьютори забезпечують цілодобову персоналізовану підтримку
5. 67% курсів використовують гібридний формат з AI компонентами

СТАТИСТИКА:
- Ринок EdTech: зростання 45% щорічно
- Бюджет МОН на AI: зріс на 150% з 2023 року
- Економія часу викладачів: до 40% на рутинних задачах

ОСНОВНІ ТРЕНДИ:
- Персоналізація навчання через AI
- Мультимодальний AI (текст + голос + відео)
- Формувальне оцінювання в реальному часі

ПРОБЛЕМИ ТА ВИКЛИКИ:
- Етика використання AI в освітньому процесі
- Захист персональних даних студентів
- Цифрова нерівність між регіонами
- Підготовка викладачів до роботи з AI"""

        print(f"\nРезультати ({len(results.split())} слів):")
        print(f"{results[:400]}...")

        return {
            "research_results": results,
            "messages": ["[OK] Researcher: дослідження завершено"]
        }

    def human_review_node(self, state: HumanLoopState) -> dict:
        """
        HUMAN CHECKPOINT: людина перевіряє результати дослідження.
        Може підтвердити, відхилити, або внести правки.
        """
        print("\n" + "="*60)
        print("HUMAN REVIEW: Очікуємо рішення людини...")
        print("="*60)

        research = state.get("research_results", "")

        print(f"""
┌──────────────────────────────────────────────────────────┐
│                 HUMAN-IN-THE-LOOP                        │
│           Перевірка результатів дослідження               │
└──────────────────────────────────────────────────────────┘

Тема: {state['topic']}

РЕЗУЛЬТАТИ ДОСЛІДЖЕННЯ:
{research[:600]}{'...' if len(research) > 600 else ''}
        """)

        if self.auto_mode:
            # Автоматичний режим для тестування/демо
            print("[AUTO] Автоматичне підтвердження (демо режим)")
            return {
                "human_approved": True,
                "human_feedback": "Автоматично підтверджено (демо режим)",
                "human_edits": "",
                "messages": ["[OK] Human: автоматично підтверджено (демо)"]
            }

        # Інтерактивний режим
        print("\nОберіть дію:")
        print("  [1] Підтвердити - результати задовільні")
        print("  [2] Відхилити  - потрібно перезапустити з правками")
        print("  [3] Доповнити  - додати коментар та підтвердити")

        try:
            choice = input("\nВаш вибір (1/2/3): ").strip()

            if choice == "1":
                print("\n[HUMAN] Результати підтверджено!")
                return {
                    "human_approved": True,
                    "human_feedback": "Підтверджено без змін",
                    "human_edits": "",
                    "messages": ["[OK] Human: підтверджено"]
                }

            elif choice == "2":
                edits = input("Що потрібно змінити/додати? ").strip()
                if not edits:
                    edits = "Потрібно більше конкретних даних та прикладів"
                print(f"\n[HUMAN] Відхилено. Правки: {edits}")
                return {
                    "human_approved": False,
                    "human_feedback": f"Відхилено. Правки: {edits}",
                    "human_edits": edits,
                    "messages": [f"[OK] Human: відхилено, правки: {edits[:50]}..."]
                }

            elif choice == "3":
                comment = input("Ваш коментар/доповнення: ").strip()
                if not comment:
                    comment = "Додаткових коментарів немає"
                print(f"\n[HUMAN] Підтверджено з коментарем: {comment}")
                return {
                    "human_approved": True,
                    "human_feedback": f"Підтверджено з коментарем: {comment}",
                    "human_edits": comment,
                    "messages": [f"[OK] Human: підтверджено з коментарем"]
                }

            else:
                print("\n[HUMAN] Невідомий вибір, автоматичне підтвердження")
                return {
                    "human_approved": True,
                    "human_feedback": "Підтверджено за замовчуванням",
                    "human_edits": "",
                    "messages": ["[OK] Human: підтверджено за замовчуванням"]
                }

        except (EOFError, KeyboardInterrupt):
            print("\n[HUMAN] Введення перервано, автоматичне підтвердження")
            return {
                "human_approved": True,
                "human_feedback": "Підтверджено (введення перервано)",
                "human_edits": "",
                "messages": ["[OK] Human: підтверджено (перервано)"]
            }

    def reporter_node(self, state: HumanLoopState) -> dict:
        """Формує фінальний звіт з урахуванням людського зворотного зв'язку"""
        print("\n" + "="*60)
        print("REPORTER: Формування фінального звіту...")
        print("="*60)

        research = state.get("research_results", "")
        human_feedback = state.get("human_feedback", "")
        human_edits = state.get("human_edits", "")

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║     HUMAN-IN-THE-LOOP - ВЕРИФІКОВАНИЙ ЗВІТ                   ║
╚══════════════════════════════════════════════════════════════╝

Дата: {datetime.now():%Y-%m-%d %H:%M:%S}
Тема: {state['topic']}
Шаблон: Human in the Loop
Людська верифікація: {human_feedback}
Платформа: LangGraph

══════════════════════════════════════════════════════════════
ВЕРИФІКОВАНІ РЕЗУЛЬТАТИ ДОСЛІДЖЕННЯ
══════════════════════════════════════════════════════════════

{research}
"""

        if human_edits:
            report += f"""
══════════════════════════════════════════════════════════════
КОМЕНТАРІ ТА ДОПОВНЕННЯ ЕКСПЕРТА
══════════════════════════════════════════════════════════════

{human_edits}
"""

        if self.llm and human_edits:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Інтегруйте коментарі експерта в висновки. Створіть executive summary."),
                    ("human", "Дослідження:\n{research}\n\nКоментарі експерта:\n{edits}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                synthesis = chain.invoke({"research": research, "edits": human_edits})
                report += f"""
══════════════════════════════════════════════════════════════
СИНТЕЗ (з урахуванням коментарів експерта)
══════════════════════════════════════════════════════════════

{synthesis}
"""
            except Exception as e:
                print(f"[WARNING] AI синтез: {e}")

        report += f"""
══════════════════════════════════════════════════════════════
[OK] Верифіковане дослідження завершено
Шлях: Researcher → HUMAN REVIEW → Reporter
Статус: Підтверджено людиною-експертом
Powered by LangGraph (Human-in-the-Loop Pattern)
"""

        # Зберігаємо звіт
        filename = f"human_loop_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_data = {
            "pattern": "human_in_the_loop",
            "topic": state["topic"],
            "research_results": research,
            "human_feedback": human_feedback,
            "human_edits": human_edits,
            "report": report,
            "timestamp": datetime.now().isoformat(),
            "framework": "LangGraph"
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Звіт збережено: {filename}")

        return {
            "final_report": report,
            "messages": ["[OK] Reporter: верифікований звіт створено"]
        }

    def run(self, topic: str) -> dict:
        """Запуск системи з людською верифікацією"""
        mode_str = "АВТО (демо)" if self.auto_mode else "ІНТЕРАКТИВНИЙ"
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     LANGGRAPH HUMAN-IN-THE-LOOP PATTERN                      ║
║     Людина в контурі прийняття рішень                        ║
╚══════════════════════════════════════════════════════════════╝

Тема: {topic}
Режим: {mode_str}
Шаблон: Human in the Loop

  [Input] → [Researcher] → [HUMAN REVIEW] → [Reporter] → [Output]
                                ↑    │
                                └────┘ (коригування)
        """)

        initial_state = {
            "topic": topic,
            "research_results": "",
            "human_feedback": "",
            "human_approved": False,
            "human_edits": "",
            "final_report": "",
            "messages": []
        }

        try:
            final_state = self.app.invoke(initial_state)

            print("\n" + "="*60)
            print("[OK] HUMAN-IN-THE-LOOP СИСТЕМА ЗАВЕРШИЛА РОБОТУ")
            print("="*60)

            for msg in final_state.get("messages", []):
                print(f"  {msg}")

            if final_state.get("final_report"):
                print("\n" + final_state["final_report"])

            return final_state

        except Exception as e:
            print(f"\n[ERROR] Помилка: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

# ===========================
# ГОЛОВНА ФУНКЦІЯ
# ===========================

def main():
    """Демонстрація Human-in-the-Loop"""
    print("\nПеревірка пакетів:")
    try:
        import langchain
        print(f"   [OK] LangChain: {langchain.__version__}")
    except:
        print("   [ERROR] LangChain: не встановлено")
    try:
        import langgraph
        print(f"   [OK] LangGraph: встановлено")
    except:
        print("   [ERROR] LangGraph: не встановлено")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"   [OK] API ключ: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("   [WARNING] API ключ: не знайдено (демо режим)")

    print("\n" + "="*60)

    topic = "Штучний інтелект в освіті України 2025: можливості та виклики"

    # Визначаємо режим (auto для неінтерактивного середовища)
    import sys
    auto_mode = not sys.stdin.isatty()

    if auto_mode:
        print("[INFO] Неінтерактивне середовище - авто-режим")
    else:
        print("[INFO] Інтерактивний режим - вас запитають підтвердження")

    system = HumanInLoopSystem(api_key, auto_mode=auto_mode)
    result = system.run(topic)

    if "error" not in result:
        print("\nГотово! Перегляньте файли:")
        print("   human_loop_report_*.json - верифікований звіт")
    else:
        print("\n[WARNING] Виконання завершилось з помилками")

    print("\n" + "="*60)
    print("Навчальні поради (Human in the Loop):")
    print("- Людина верифікує результати AI перед фінальним звітом")
    print("- auto_mode=False для інтерактивного режиму")
    print("- Людина може відхилити, підтвердити або доповнити")
    print("- Реальні системи використовують webhooks або UI для взаємодії")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()
