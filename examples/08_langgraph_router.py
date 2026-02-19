"""
Модуль 1.3: Маршрутизатор (Router Pattern) на LangGraph
Агент-класифікатор визначає тип запиту і направляє до
спеціалізованого агента для обробки.

Шаблон: Router (Conditional Routing)

                     ┌→ [Tech Agent]    → Out ┐
  [In] → [Router] ──┤→ [Edu Agent]     → Out ├→ [Reporter] → [Out]
                     └→ [Policy Agent]  → Out ┘
"""

import os
import json
import re
from datetime import datetime
from typing import TypedDict, Annotated, List, Literal
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

class RouterState(TypedDict):
    """Стан для маршрутизатора"""
    topic: str
    route: str              # Обраний маршрут: tech / edu / policy
    specialist_results: str  # Результати спеціалізованого агента
    final_report: str
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
    """Демо-результати"""
    if any(w in query.lower() for w in ["технолог", "ai", "gpt", "ml", "платформ"]):
        return (
            "- GPT-5 у класі: Нові мовні моделі адаптують пояснення під рівень студента...\n"
            "- AI-Proctoring: Системи моніторингу іспитів на основі ШІ зменшують шахрайство на 85%...\n"
            "- Learning Analytics: Платформи аналітики навчання прогнозують успішність з точністю 92%..."
        )
    elif any(w in query.lower() for w in ["освіт", "навчан", "студент", "педагог"]):
        return (
            "- Blended Learning 2025: 67% курсів використовують гібридний формат з AI компонентами...\n"
            "- AI-тьютори: Персоналізовані помічники покращують оцінки студентів на 28%...\n"
            "- Gamification + AI: Ігрові елементи з адаптивним AI підвищують залученість на 45%..."
        )
    else:
        return (
            "- Стратегія МОН: Програма 'Цифрова освіта 2030' виділяє 2.5 млрд грн на AI...\n"
            "- GDPR та освіта: Нові правила захисту даних студентів при використанні AI...\n"
            "- Міжнародний досвід: Естонія та Фінляндія як лідери AI-освіти в Європі..."
        )

# ===========================
# ROUTER AGENT SYSTEM
# ===========================

class RouterAgentSystem:
    """
    Маршрутизатор: класифікує запит і направляє до спеціалізованого агента.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            try:
                self.llm = ChatOpenAI(model="gpt-4", temperature=0.3, api_key=self.api_key)
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
        Граф з умовною маршрутизацією:
        START → router → {tech_agent | edu_agent | policy_agent} → reporter → END
        """
        workflow = StateGraph(RouterState)

        # Вузли
        workflow.add_node("router", self.router_node)
        workflow.add_node("tech_agent", self.tech_agent_node)
        workflow.add_node("edu_agent", self.edu_agent_node)
        workflow.add_node("policy_agent", self.policy_agent_node)
        workflow.add_node("reporter", self.reporter_node)

        # Початок → маршрутизатор
        workflow.set_entry_point("router")

        # Умовна маршрутизація від router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "tech": "tech_agent",
                "edu": "edu_agent",
                "policy": "policy_agent",
            }
        )

        # Всі спеціалісти → reporter
        workflow.add_edge("tech_agent", "reporter")
        workflow.add_edge("edu_agent", "reporter")
        workflow.add_edge("policy_agent", "reporter")
        workflow.add_edge("reporter", END)

        return workflow

    @staticmethod
    def _route_decision(state: RouterState) -> str:
        """Функція маршрутизації на основі визначеного маршруту"""
        return state.get("route", "tech")

    # --- Вузли ---

    def router_node(self, state: RouterState) -> dict:
        """Маршрутизатор: класифікує тему та обирає спеціалізованого агента"""
        print("\n" + "="*60)
        print("ROUTER: Класифікація запиту...")
        print("="*60)

        topic = state["topic"]
        route = "tech"  # за замовчуванням

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Класифікуйте тему в одну з категорій:
- tech: технології, інструменти, платформи, програмне забезпечення
- edu: освіта, навчання, педагогіка, студенти, методики
- policy: політика, регулювання, стратегія, фінансування, закони

Відповідайте ТІЛЬКИ одним словом: tech, edu або policy."""),
                    ("human", "Тема: {topic}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                result = chain.invoke({"topic": topic}).strip().lower()
                if result in ("tech", "edu", "policy"):
                    route = result
            except Exception as e:
                print(f"[WARNING] AI класифікація: {e}")

        if route == "tech":
            # Демо-класифікація за ключовими словами
            topic_lower = topic.lower()
            tech_words = ["технолог", "інструмент", "платформ", "gpt", "ml", "ai", "програм"]
            edu_words = ["освіт", "навчан", "студент", "педагог", "університет", "школ"]
            policy_words = ["політик", "закон", "стратег", "фінансув", "регуляц", "бюджет"]

            scores = {
                "tech": sum(1 for w in tech_words if w in topic_lower),
                "edu": sum(1 for w in edu_words if w in topic_lower),
                "policy": sum(1 for w in policy_words if w in topic_lower),
            }
            route = max(scores, key=scores.get) if max(scores.values()) > 0 else "tech"

        route_names = {"tech": "Технологічний", "edu": "Освітній", "policy": "Політичний"}
        print(f"\n[ROUTE] Обрано маршрут: {route_names.get(route, route)} спеціаліст")

        return {
            "route": route,
            "messages": [f"[OK] Router: обрано маршрут → {route}"]
        }

    def tech_agent_node(self, state: RouterState) -> dict:
        """Технологічний агент"""
        print("\n" + "="*60)
        print("TECH AGENT: Глибоке дослідження технологій...")
        print("="*60)

        results = search_web(f"AI технології платформи інструменти {state['topic']}")

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ви - провідний експерт з AI технологій в освіті.
Створіть детальний технічний огляд: інструменти, платформи, архітектура рішень."""),
                    ("human", "Тема: {topic}\nДані:\n{data}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                results = chain.invoke({"topic": state["topic"], "data": results})
            except Exception as e:
                print(f"[WARNING] AI обробка: {e}")
        else:
            results += """

ТЕХНІЧНИЙ ОГЛЯД:
- Провідні платформи: Coursera AI, Khan Academy GPT, Duolingo Max
- Архітектура: RAG-системи для навчального контенту
- Інтеграції: LMS + AI (Moodle, Canvas з AI-плагінами)
- Тренд: Мультимодальні AI (текст + голос + відео)
- Продуктивність: Автоматична генерація тестів (-70% часу викладача)"""

        print(f"\n{results[:300]}...")
        return {
            "specialist_results": f"[ТЕХНОЛОГІЧНИЙ ЗВІТ]\n{results}",
            "messages": ["[OK] Tech Agent: дослідження завершено"]
        }

    def edu_agent_node(self, state: RouterState) -> dict:
        """Освітній агент"""
        print("\n" + "="*60)
        print("EDU AGENT: Глибоке дослідження освітніх методик...")
        print("="*60)

        results = search_web(f"освітні методики AI педагогіка {state['topic']}")

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ви - провідний експерт з педагогіки та EdTech.
Створіть детальний огляд освітніх методик з використанням AI."""),
                    ("human", "Тема: {topic}\nДані:\n{data}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                results = chain.invoke({"topic": state["topic"], "data": results})
            except Exception as e:
                print(f"[WARNING] AI обробка: {e}")
        else:
            results += """

ОСВІТНІЙ ОГЛЯД:
- Персоналізація: AI-тьютори адаптують темп та стиль пояснень
- Оцінювання: Формувальне оцінювання в реальному часі через AI
- Інклюзія: AI допомагає студентам з особливими потребами
- Мотивація: Gamification + AI підвищує залученість на 45%
- Компетенції: Нова модель "AI literacy" для студентів та викладачів"""

        print(f"\n{results[:300]}...")
        return {
            "specialist_results": f"[ОСВІТНІЙ ЗВІТ]\n{results}",
            "messages": ["[OK] Edu Agent: дослідження завершено"]
        }

    def policy_agent_node(self, state: RouterState) -> dict:
        """Агент з політик"""
        print("\n" + "="*60)
        print("POLICY AGENT: Глибоке дослідження політик та регуляцій...")
        print("="*60)

        results = search_web(f"політика регуляція AI освіта Україна {state['topic']}")

        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ви - експерт з освітньої політики та регуляцій AI.
Створіть детальний огляд: стратегії, закони, фінансування, міжнародний досвід."""),
                    ("human", "Тема: {topic}\nДані:\n{data}")
                ])
                chain = prompt | self.llm | StrOutputParser()
                results = chain.invoke({"topic": state["topic"], "data": results})
            except Exception as e:
                print(f"[WARNING] AI обробка: {e}")
        else:
            results += """

ПОЛІТИЧНИЙ ОГЛЯД:
- МОН: Стратегія 'Цифрова освіта 2030' - 2.5 млрд грн на AI
- Захист даних: Нові правила GDPR для освітнього AI
- Стандарти: Рамкова програма етичного використання AI
- Міжнародний досвід: Естонія (100% цифровізація), Фінляндія (AI в кожній школі)
- Кадри: Програма перепідготовки 50,000 викладачів до 2027"""

        print(f"\n{results[:300]}...")
        return {
            "specialist_results": f"[ПОЛІТИЧНИЙ ЗВІТ]\n{results}",
            "messages": ["[OK] Policy Agent: дослідження завершено"]
        }

    def reporter_node(self, state: RouterState) -> dict:
        """Формує фінальний звіт"""
        print("\n" + "="*60)
        print("REPORTER: Формування фінального звіту...")
        print("="*60)

        route_names = {"tech": "Технологічний", "edu": "Освітній", "policy": "Політичний"}
        route_name = route_names.get(state.get("route", ""), "Невідомий")

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║     ROUTER PATTERN - МАРШРУТИЗОВАНИЙ ЗВІТ                    ║
╚══════════════════════════════════════════════════════════════╝

Дата: {datetime.now():%Y-%m-%d %H:%M:%S}
Тема: {state['topic']}
Обраний маршрут: {route_name} спеціаліст
Шаблон: Маршрутизатор (Router Pattern)
Платформа: LangGraph

══════════════════════════════════════════════════════════════
РЕЗУЛЬТАТИ СПЕЦІАЛІЗОВАНОГО ДОСЛІДЖЕННЯ
══════════════════════════════════════════════════════════════
{state.get('specialist_results', 'Немає даних')}

══════════════════════════════════════════════════════════════
[OK] Маршрутизоване дослідження завершено
Шлях: Router → {route_name} Agent → Reporter
Powered by LangGraph (Router Pattern)
"""

        # Зберігаємо звіт
        filename = f"router_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_data = {
            "pattern": "router",
            "topic": state["topic"],
            "route": state.get("route", ""),
            "specialist_results": state.get("specialist_results", ""),
            "report": report,
            "timestamp": datetime.now().isoformat(),
            "framework": "LangGraph"
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Звіт збережено: {filename}")

        return {
            "final_report": report,
            "messages": ["[OK] Reporter: звіт створено"]
        }

    def run(self, topic: str) -> dict:
        """Запуск системи з маршрутизатором"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     LANGGRAPH ROUTER PATTERN                                 ║
║     Маршрутизатор: розумна маршрутизація запитів              ║
╚══════════════════════════════════════════════════════════════╝

Тема: {topic}
Шаблон: Маршрутизатор (Conditional Routing)

                     ┌→ [Tech Agent]    ─┐
  [Input] → [Router]─┤→ [Edu Agent]     ├→ [Reporter] → [Output]
                     └→ [Policy Agent]  ─┘
        """)

        initial_state = {
            "topic": topic,
            "route": "",
            "specialist_results": "",
            "final_report": "",
            "messages": []
        }

        try:
            final_state = self.app.invoke(initial_state)

            print("\n" + "="*60)
            print("[OK] МАРШРУТИЗАТОР ЗАВЕРШИВ РОБОТУ")
            print("="*60)

            for msg in final_state.get("messages", []):
                print(f"  {msg}")

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
    """Демонстрація маршрутизатора"""
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

    system = RouterAgentSystem(api_key)

    # Тест 1: Технологічна тема
    print("\n\n" + "#"*60)
    print("# ТЕСТ 1: Технологічна тема")
    print("#"*60)
    system.run("AI платформи та інструменти машинного навчання в університетах")

    # Тест 2: Освітня тема
    print("\n\n" + "#"*60)
    print("# ТЕСТ 2: Освітня тема")
    print("#"*60)
    system.run("Методики навчання студентів з використанням AI-тьюторів")

    # Тест 3: Політична тема
    print("\n\n" + "#"*60)
    print("# ТЕСТ 3: Тема політик")
    print("#"*60)
    system.run("Стратегія фінансування та регуляція AI в освіті України")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()
