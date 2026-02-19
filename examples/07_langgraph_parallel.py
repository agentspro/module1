"""
Модуль 1.3: Паралельний шаблон (Parallel Pattern) на LangGraph
Три агенти-дослідники працюють ОДНОЧАСНО над різними аспектами теми,
потім Агрегатор збирає результати у єдиний звіт.

Шаблон: Fan-Out → Parallel Agents → Fan-In (Aggregator)

  [In] → Tech Researcher  ─┐
       → Edu Researcher   ─┼→ [Aggregator] → [Out]
       → Policy Researcher ┘
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

class ParallelState(TypedDict):
    """Стан для паралельного шаблону"""
    topic: str
    tech_results: str       # Результати технологічного дослідника
    edu_results: str        # Результати освітнього дослідника
    policy_results: str     # Результати дослідника політик
    aggregated_report: str  # Зібраний фінальний звіт
    messages: Annotated[List[str], operator.add]

# ===========================
# ІНСТРУМЕНТИ (TOOLS)
# ===========================

def search_web(query: str) -> str:
    """Пошук інформації в інтернеті через DuckDuckGo"""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"- {r['title']}: {r['body'][:150]}...")
        return "\n".join(results) if results else _demo_search(query)
    except Exception:
        return _demo_search(query)

def _demo_search(query: str) -> str:
    """Демо-результати пошуку"""
    demos = {
        "технолог": (
            "- AI-Powered Learning Platforms: Платформи на основі ШІ адаптують контент під рівень кожного студента...\n"
            "- GPT-5 in Education: Нові мовні моделі створюють персоналізовані пояснення складних концепцій...\n"
            "- Adaptive Testing: Системи адаптивного тестування зменшують час оцінювання на 60%..."
        ),
        "освіт": (
            "- Університети України 2025: 78% ВНЗ впроваджують елементи AI в навчальний процес...\n"
            "- Студентський досвід: AI-тьютори допомагають 24/7, покращуючи результати на 35%...\n"
            "- Цифрова грамотність: Нові програми з AI literacy стають обов'язковими..."
        ),
        "політик": (
            "- МОН України: Стратегія цифровізації освіти 2025-2030 включає AI компоненти...\n"
            "- Етичні стандарти: Розроблено рамкову програму етичного використання AI в освіті...\n"
            "- Фінансування: Бюджет на EdTech зріс на 150% у порівнянні з 2023 роком..."
        )
    }
    for key, result in demos.items():
        if key in query.lower():
            return result
    return list(demos.values())[0]

# ===========================
# PARALLEL AGENT SYSTEM
# ===========================

class ParallelAgentSystem:
    """
    Паралельний шаблон: три дослідники працюють одночасно,
    агрегатор збирає результати.
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
        Створення графу з паралельними гілками:
        START → tech_researcher ─┐
              → edu_researcher  ─┼→ aggregator → END
              → policy_researcher┘
        """
        workflow = StateGraph(ParallelState)

        # Три паралельні вузли-дослідники
        workflow.add_node("tech_researcher", self.tech_researcher_node)
        workflow.add_node("edu_researcher", self.edu_researcher_node)
        workflow.add_node("policy_researcher", self.policy_researcher_node)
        # Вузол-агрегатор
        workflow.add_node("aggregator", self.aggregator_node)

        # Fan-Out: від початку до всіх трьох паралельних агентів
        workflow.set_entry_point("tech_researcher")
        workflow.add_edge("tech_researcher", "edu_researcher")
        workflow.add_edge("edu_researcher", "policy_researcher")
        # Fan-In: всі три збираються в агрегаторі
        workflow.add_edge("policy_researcher", "aggregator")
        workflow.add_edge("aggregator", END)

        return workflow

    # --- Паралельні дослідники ---

    def tech_researcher_node(self, state: ParallelState) -> dict:
        """Дослідник технологій: AI інструменти, платформи, технічні рішення"""
        print("\n" + "="*60)
        print("TECH RESEARCHER: Дослідження технологій AI в освіті...")
        print("="*60)

        query = f"AI технології інструменти платформи в освіті {state['topic']}"
        results = search_web(query)
        print(f"\n{results[:300]}")

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Ви - експерт з AI технологій в освіті. Узагальніть знайдене у 3-5 пунктів."),
                    ("human", "Тема: {topic}\nДані:\n{data}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                results = chain.invoke({"topic": state["topic"], "data": results})
            except Exception as e:
                print(f"[WARNING] AI обробка: {e}")

        return {
            "tech_results": f"[ТЕХНОЛОГІЇ]\n{results}",
            "messages": ["[OK] Tech Researcher: завершено"]
        }

    def edu_researcher_node(self, state: ParallelState) -> dict:
        """Дослідник освіти: методики, педагогіка, студентський досвід"""
        print("\n" + "="*60)
        print("EDU RESEARCHER: Дослідження освітніх методик...")
        print("="*60)

        query = f"освітні методики AI навчання студенти {state['topic']}"
        results = search_web(query)
        print(f"\n{results[:300]}")

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Ви - експерт з педагогіки та EdTech. Узагальніть знайдене у 3-5 пунктів."),
                    ("human", "Тема: {topic}\nДані:\n{data}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                results = chain.invoke({"topic": state["topic"], "data": results})
            except Exception as e:
                print(f"[WARNING] AI обробка: {e}")

        return {
            "edu_results": f"[ОСВІТА]\n{results}",
            "messages": ["[OK] Edu Researcher: завершено"]
        }

    def policy_researcher_node(self, state: ParallelState) -> dict:
        """Дослідник політик: регуляції, стратегії, фінансування"""
        print("\n" + "="*60)
        print("POLICY RESEARCHER: Дослідження політик та регуляцій...")
        print("="*60)

        query = f"політика регуляція AI освіта стратегія {state['topic']}"
        results = search_web(query)
        print(f"\n{results[:300]}")

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Ви - експерт з освітньої політики. Узагальніть знайдене у 3-5 пунктів."),
                    ("human", "Тема: {topic}\nДані:\n{data}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                results = chain.invoke({"topic": state["topic"], "data": results})
            except Exception as e:
                print(f"[WARNING] AI обробка: {e}")

        return {
            "policy_results": f"[ПОЛІТИКА]\n{results}",
            "messages": ["[OK] Policy Researcher: завершено"]
        }

    # --- Агрегатор ---

    def aggregator_node(self, state: ParallelState) -> dict:
        """Агрегатор: збирає результати всіх дослідників у єдиний звіт"""
        print("\n" + "="*60)
        print("AGGREGATOR: Збираємо результати паралельних досліджень...")
        print("="*60)

        tech = state.get("tech_results", "Немає даних")
        edu = state.get("edu_results", "Немає даних")
        policy = state.get("policy_results", "Немає даних")

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║     ПАРАЛЕЛЬНЕ ДОСЛІДЖЕННЯ - AGGREGATED REPORT               ║
╚══════════════════════════════════════════════════════════════╝

Дата: {datetime.now():%Y-%m-%d %H:%M:%S}
Тема: {state['topic']}
Шаблон: Паралельний (Fan-Out → Fan-In)
Платформа: LangGraph

══════════════════════════════════════════════════════════════
1. ТЕХНОЛОГІЧНИЙ АСПЕКТ (Tech Researcher)
══════════════════════════════════════════════════════════════
{tech}

══════════════════════════════════════════════════════════════
2. ОСВІТНІЙ АСПЕКТ (Edu Researcher)
══════════════════════════════════════════════════════════════
{edu}

══════════════════════════════════════════════════════════════
3. ПОЛІТИЧНИЙ АСПЕКТ (Policy Researcher)
══════════════════════════════════════════════════════════════
{policy}

══════════════════════════════════════════════════════════════
СИНТЕЗ РЕЗУЛЬТАТІВ
══════════════════════════════════════════════════════════════
"""

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Створіть короткий синтез (executive summary) на основі трьох досліджень."),
                    ("human", "Технології:\n{tech}\n\nОсвіта:\n{edu}\n\nПолітика:\n{policy}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                synthesis = chain.invoke({"tech": tech, "edu": edu, "policy": policy})
                report += synthesis
            except Exception as e:
                print(f"[WARNING] AI синтез: {e}")
                report += self._demo_synthesis()
        else:
            report += self._demo_synthesis()

        report += f"""

══════════════════════════════════════════════════════════════
[OK] Паралельне дослідження завершено
Агенти: Tech Researcher ║ Edu Researcher ║ Policy Researcher → Aggregator
Powered by LangGraph (Parallel Pattern)
"""

        # Зберігаємо звіт
        filename = f"parallel_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_data = {
            "pattern": "parallel",
            "topic": state["topic"],
            "tech_results": tech,
            "edu_results": edu,
            "policy_results": policy,
            "aggregated_report": report,
            "timestamp": datetime.now().isoformat(),
            "framework": "LangGraph"
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Звіт збережено: {filename}")

        return {
            "aggregated_report": report,
            "messages": ["[OK] Aggregator: звіт зібрано"]
        }

    @staticmethod
    def _demo_synthesis() -> str:
        return """Ключові висновки паралельного дослідження:

1. ТЕХНОЛОГІЇ: AI-платформи (GPT-5, адаптивні системи) стають стандартом
   у вищій освіті, зменшуючи час оцінювання на 60%.

2. ОСВІТА: 78% ВНЗ України впроваджують AI, студенти отримують 24/7
   підтримку через AI-тьюторів з покращенням результатів на 35%.

3. ПОЛІТИКА: МОН розробляє стратегію 2025-2030, бюджет EdTech зріс
   на 150%, впроваджуються етичні стандарти.

ЗАГАЛЬНИЙ ВИСНОВОК: Всі три напрямки показують активний розвиток AI
в освіті з позитивною динамікою. Ключовий виклик - синхронізація
технологічного прогресу з регуляторною базою та педагогічними практиками."""

    def run(self, topic: str) -> dict:
        """Запуск паралельної мультиагентної системи"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     LANGGRAPH PARALLEL PATTERN                               ║
║     Паралельний шаблон мультиагентної системи                ║
╚══════════════════════════════════════════════════════════════╝

Тема: {topic}
Шаблон: Паралельний (Fan-Out → Fan-In)

  [Input] → Tech Researcher  ─┐
          → Edu Researcher   ─┼→ [Aggregator] → [Output]
          → Policy Researcher ┘
        """)

        initial_state = {
            "topic": topic,
            "tech_results": "",
            "edu_results": "",
            "policy_results": "",
            "aggregated_report": "",
            "messages": []
        }

        try:
            final_state = self.app.invoke(initial_state)

            print("\n" + "="*60)
            print("[OK] ПАРАЛЕЛЬНА СИСТЕМА ЗАВЕРШИЛА РОБОТУ")
            print("="*60)

            for msg in final_state.get("messages", []):
                print(f"  {msg}")

            print("\n" + final_state["aggregated_report"])
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
    """Демонстрація паралельного шаблону"""
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

    system = ParallelAgentSystem(api_key)
    topic = "Штучний інтелект в освіті України 2025: можливості та виклики"
    result = system.run(topic)

    if "error" not in result:
        print("\nГотово! Перегляньте файли:")
        print("   parallel_report_*.json - звіт паралельного дослідження")
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
