"""
Модуль 1.3: Петля (Loop / Iterative Pattern) на LangGraph
Автор-агент пише контент, Критик перевіряє якість.
Якщо якість недостатня - повертаємось до Автора для доопрацювання.

Шаблон: Ітеративна петля (Loop with Quality Gate)

  [In] → [Researcher] → [Writer] → [Critic] ──┐
                            ↑                   │
                            └── (потрібно ще) ──┘
                                    │
                                (достатньо) → [Out]
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

class LoopState(TypedDict):
    """Стан для ітеративної петлі"""
    topic: str
    research_data: str       # Зібрані дані для написання
    draft: str               # Поточний чернетковий варіант
    feedback: str            # Зворотній зв'язок від критика
    iteration: int           # Номер ітерації
    max_iterations: int      # Максимум ітерацій
    quality_passed: bool     # Чи пройшла перевірка якості
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
            for r in ddgs.text(query, max_results=3):
                results.append(f"- {r['title']}: {r['body'][:150]}...")
        return "\n".join(results) if results else _demo_search()
    except Exception:
        return _demo_search()

def _demo_search() -> str:
    return (
        "- AI в освіті 2025: 85% університетів активно впроваджують AI технології в навчальний процес...\n"
        "- Персоналізація навчання: Адаптивні AI системи підвищують успішність студентів на 35%...\n"
        "- EdTech ринок: Глобальний ринок AI в освіті досягне $25.7 млрд до 2030 року...\n"
        "- AI-тьютори: Віртуальні помічники забезпечують цілодобову підтримку студентів...\n"
        "- Виклики: Етика використання AI, захист даних, цифрова нерівність залишаються проблемами..."
    )

def evaluate_quality(text: str) -> dict:
    """Оцінка якості тексту за кількома критеріями"""
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')
    paragraphs = text.count('\n\n') + 1

    # Перевірка ключових елементів якісного звіту
    has_intro = any(w in text.lower() for w in ["вступ", "огляд", "введення", "тема"])
    has_data = any(w in text.lower() for w in ["статистик", "дані", "дослідження", "%", "млрд"])
    has_analysis = any(w in text.lower() for w in ["аналіз", "висновок", "інсайт", "тренд"])
    has_recommendations = any(w in text.lower() for w in ["рекомендац", "пропозиц", "наступні кроки"])
    has_structure = text.count('#') >= 2 or text.count('═') >= 2 or text.count('**') >= 2

    score = 0
    feedback_items = []

    if words >= 150:
        score += 20
    else:
        feedback_items.append(f"Текст занадто короткий ({words} слів, потрібно >= 150)")

    if has_intro:
        score += 15
    else:
        feedback_items.append("Додайте вступну частину з описом теми")

    if has_data:
        score += 20
    else:
        feedback_items.append("Додайте статистику та конкретні дані")

    if has_analysis:
        score += 20
    else:
        feedback_items.append("Додайте аналітичні висновки та інсайти")

    if has_recommendations:
        score += 15
    else:
        feedback_items.append("Додайте рекомендації та наступні кроки")

    if has_structure:
        score += 10
    else:
        feedback_items.append("Покращте структуру (заголовки, розділи)")

    return {
        "score": score,
        "passed": score >= 70,
        "feedback": feedback_items,
        "stats": {"words": words, "sentences": sentences, "paragraphs": paragraphs}
    }

# ===========================
# LOOP AGENT SYSTEM
# ===========================

class LoopAgentSystem:
    """
    Ітеративна петля: Writer → Critic → (loop back або exit)
    """

    def __init__(self, api_key: str = None):
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
        Граф з петлею:
        START → researcher → writer → critic → {writer (loop) | END}
        """
        workflow = StateGraph(LoopState)

        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("writer", self.writer_node)
        workflow.add_node("critic", self.critic_node)

        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "writer")
        workflow.add_edge("writer", "critic")

        # Умовний перехід: петля або вихід
        workflow.add_conditional_edges(
            "critic",
            self._should_continue,
            {
                "continue": "writer",  # Петля назад до Writer
                "end": END,            # Вихід
            }
        )

        return workflow

    @staticmethod
    def _should_continue(state: LoopState) -> str:
        """Вирішує чи продовжувати ітерації"""
        if state.get("quality_passed", False):
            return "end"
        if state.get("iteration", 0) >= state.get("max_iterations", 3):
            return "end"
        return "continue"

    # --- Вузли ---

    def researcher_node(self, state: LoopState) -> dict:
        """Дослідник: збирає початкові дані"""
        print("\n" + "="*60)
        print("RESEARCHER: Збір даних для написання...")
        print("="*60)

        data = search_web(state["topic"])
        print(f"\n{data[:300]}...")

        return {
            "research_data": data,
            "iteration": 0,
            "messages": ["[OK] Researcher: дані зібрано"]
        }

    def writer_node(self, state: LoopState) -> dict:
        """Автор: пише/покращує чернетку на основі даних та зворотного зв'язку"""
        iteration = state.get("iteration", 0) + 1
        print("\n" + "="*60)
        print(f"WRITER: Ітерація #{iteration} - {'Перша версія' if iteration == 1 else 'Доопрацювання'}...")
        print("="*60)

        research_data = state.get("research_data", "")
        previous_draft = state.get("draft", "")
        feedback = state.get("feedback", "")

        if self.llm:
            try:
                if iteration == 1:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """Ви - професійний технічний автор. Напишіть структурований звіт.
Включіть: вступ, основні дані зі статистикою, аналіз трендів, висновки та рекомендації."""),
                        ("human", "Тема: {topic}\nДані для звіту:\n{data}")
                    ])
                    chain = prompt | self.llm | StrOutputParser()
                    draft = chain.invoke({"topic": state["topic"], "data": research_data})
                else:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """Ви - професійний технічний автор. Доопрацюйте звіт за зворотнім зв'язком.
Збережіть вже хороші частини, виправте зазначені проблеми."""),
                        ("human", "Поточна версія:\n{draft}\n\nЗворотній зв'язок:\n{feedback}\n\nДодаткові дані:\n{data}")
                    ])
                    chain = prompt | self.llm | StrOutputParser()
                    draft = chain.invoke({"draft": previous_draft, "feedback": feedback, "data": research_data})
            except Exception as e:
                print(f"[WARNING] AI генерація: {e}")
                draft = self._demo_draft(iteration, feedback)
        else:
            draft = self._demo_draft(iteration, feedback)

        print(f"\nЧернетка v{iteration} ({len(draft.split())} слів):")
        print(f"{draft[:200]}...")

        return {
            "draft": draft,
            "iteration": iteration,
            "messages": [f"[OK] Writer: ітерація #{iteration} завершена"]
        }

    def critic_node(self, state: LoopState) -> dict:
        """Критик: оцінює якість та дає зворотній зв'язок"""
        iteration = state.get("iteration", 1)
        print("\n" + "="*60)
        print(f"CRITIC: Перевірка якості (ітерація #{iteration})...")
        print("="*60)

        draft = state.get("draft", "")
        quality = evaluate_quality(draft)

        score = quality["score"]
        passed = quality["passed"]

        print(f"\n   Оцінка якості: {score}/100")
        print(f"   Статус: {'ПРИЙНЯТО' if passed else 'ПОТРЕБУЄ ДООПРАЦЮВАННЯ'}")
        print(f"   Статистика: {quality['stats']}")

        if not passed:
            feedback_text = "Зауваження для покращення:\n"
            for i, item in enumerate(quality["feedback"], 1):
                feedback_text += f"  {i}. {item}\n"
                print(f"   [{i}] {item}")
        else:
            feedback_text = "Якість звіту відповідає вимогам. Затверджено."
            print("   Всі критерії виконано!")

        # Якщо це остання ітерація і якість не пройшла - приймаємо як є
        max_iter = state.get("max_iterations", 3)
        if not passed and iteration >= max_iter:
            print(f"\n   [WARNING] Досягнуто максимум ітерацій ({max_iter}). Приймаємо поточну версію.")
            passed = True

        # Формуємо фінальний звіт якщо пройшло
        final_report = ""
        if passed:
            final_report = f"""
╔══════════════════════════════════════════════════════════════╗
║     LOOP PATTERN - ІТЕРАТИВНИЙ ЗВІТ                          ║
╚══════════════════════════════════════════════════════════════╝

Дата: {datetime.now():%Y-%m-%d %H:%M:%S}
Тема: {state['topic']}
Ітерацій: {iteration}
Фінальна оцінка: {score}/100
Шаблон: Петля (Loop / Iterative Pattern)
Платформа: LangGraph

══════════════════════════════════════════════════════════════
ФІНАЛЬНИЙ ЗВІТ
══════════════════════════════════════════════════════════════

{draft}

══════════════════════════════════════════════════════════════
[OK] Ітеративне дослідження завершено за {iteration} ітерацій
Агенти: Researcher → Writer ⟲ Critic → Output
Powered by LangGraph (Loop Pattern)
"""
            # Зберігаємо звіт
            filename = f"loop_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            report_data = {
                "pattern": "loop",
                "topic": state["topic"],
                "iterations": iteration,
                "final_score": score,
                "report": final_report,
                "timestamp": datetime.now().isoformat(),
                "framework": "LangGraph"
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            print(f"\n[OK] Звіт збережено: {filename}")

        return {
            "feedback": feedback_text,
            "quality_passed": passed,
            "final_report": final_report,
            "messages": [f"[OK] Critic: оцінка {score}/100 ({'прийнято' if passed else 'повернуто на доопрацювання'})"]
        }

    @staticmethod
    def _demo_draft(iteration: int, feedback: str) -> str:
        """Генерація демо-чернетки з покращенням на кожній ітерації"""
        if iteration == 1:
            return """AI в освіті - це важлива тема. Багато університетів використовують AI."""
        elif iteration == 2:
            return """**Огляд: Штучний інтелект в освіті України 2025**

Вступ та огляд теми:
Штучний інтелект трансформує освітній процес в Україні. Ця тема набуває
все більшого значення в контексті цифровізації.

Ключові дані та статистика:
За даними дослідження, 85% університетів активно впроваджують AI технології.
Глобальний ринок EdTech AI досягне $25.7 млрд до 2030 року.
Адаптивні системи підвищують успішність на 35%.

Аналіз трендів та висновки:
Основний тренд - персоналізація навчання через AI-тьюторів.
Інсайт: найбільший ефект дає комбінація AI + традиційних методів."""
        else:
            return """**Комплексний звіт: Штучний інтелект в освіті України 2025**

**Вступ та огляд теми:**
Штучний інтелект (ШІ) стрімко трансформує освітній простір України.
У 2025 році ми спостерігаємо безпрецедентне впровадження AI-технологій
на всіх рівнях освіти - від шкіл до університетів.

**Ключові дані та статистика:**
- 85% українських університетів впроваджують AI в навчальний процес
- Глобальний ринок AI в освіті досягне $25.7 млрд до 2030 року
- Адаптивні AI-системи підвищують успішність студентів на 35%
- AI-тьютори забезпечують цілодобову підтримку студентів
- Бюджет МОН на EdTech зріс на 150% порівняно з 2023 роком

**Аналіз трендів та інсайти:**
Основний тренд - повна персоналізація навчального процесу через AI.
AI-тьютори адаптують темп, стиль та складність матеріалу під кожного студента.
Найбільший ефект досягається при гібридному підході: AI + викладач.

**Рекомендації та наступні кроки:**
1. Розробити стратегію впровадження AI у ВНЗ
2. Інвестувати в підготовку викладачів для роботи з AI
3. Створити етичні стандарти використання AI в освіті
4. Забезпечити рівний доступ до AI-інструментів
5. Запустити пілотні проекти у 5-10 університетах"""

    def run(self, topic: str) -> dict:
        """Запуск ітеративної системи"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     LANGGRAPH LOOP PATTERN                                   ║
║     Ітеративна петля: Writer ⟲ Critic                        ║
╚══════════════════════════════════════════════════════════════╝

Тема: {topic}
Шаблон: Петля (Loop with Quality Gate)
Макс. ітерацій: 3

  [Input] → [Researcher] → [Writer] → [Critic] ──┐
                              ↑                    │
                              └── (ще раз) ────────┘
                                      │
                                  (готово) → [Output]
        """)

        initial_state = {
            "topic": topic,
            "research_data": "",
            "draft": "",
            "feedback": "",
            "iteration": 0,
            "max_iterations": 3,
            "quality_passed": False,
            "final_report": "",
            "messages": []
        }

        try:
            final_state = self.app.invoke(initial_state)

            print("\n" + "="*60)
            print("[OK] ІТЕРАТИВНА СИСТЕМА ЗАВЕРШИЛА РОБОТУ")
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
    """Демонстрація ітеративної петлі"""
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

    system = LoopAgentSystem(api_key)
    topic = "Штучний інтелект в освіті України 2025: можливості та виклики"
    result = system.run(topic)

    if "error" not in result:
        iterations = result.get("iteration", 0)
        print(f"\nГотово! Знадобилось {iterations} ітерацій")
        print("   loop_report_*.json - ітеративний звіт")
    else:
        print("\n[WARNING] Виконання завершилось з помилками")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()
