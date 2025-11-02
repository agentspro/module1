"""
Модуль 1: Multi-Agent System на LangChain + LangGraph 1.0
Мультиагентна система з трьома агентами: Researcher, Analyst, Reporter
"""

import os
import json
from datetime import datetime
from typing import TypedDict, Annotated, List
import operator

# LangChain 1.0 imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    print("[OK] LangChain 1.0 компоненти завантажено")
except ImportError as e:
    print(f"[ERROR] Помилка імпорту LangChain: {e}")
    print("Встановіть: pip install langchain-openai langchain-core")
    exit(1)

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    print("[OK] LangGraph завантажено")
except ImportError as e:
    print(f"[ERROR] Помилка імпорту LangGraph: {e}")
    print("Встановіть: pip install langgraph")
    exit(1)

# Завантаження .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] .env файл завантажено")
except:
    print("[WARNING] python-dotenv не встановлено")

# ===========================
# STATE DEFINITION
# ===========================

class AgentState(TypedDict):
    """Стан агентів, що передається між вузлами графу"""
    topic: str
    research_results: str
    analysis_results: str
    final_report: str
    messages: Annotated[List[str], operator.add]
    timestamp: str

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

        return f"Результати пошуку для '{query}':\n" + "\n".join(results)
    except Exception as e:
        # Демо результати якщо пошук не працює
        return f"""Демо результати для '{query}':
- AI в освіті: Персоналізація навчання через штучний інтелект. Адаптивні системи дозволяють створювати індивідуальні траєкторії...
- Тренди 2025: 85% університетів використовують AI. За даними дослідження, впровадження AI технологій зростає на 45% щороку...
- Виклики: Етика та приватність в AI системах. Основні проблеми включають захист персональних даних студентів..."""

def analyze_data(text: str) -> str:
    """Аналіз тексту та витягування ключової інформації"""
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')

    # Пошук ключових слів
    keywords = {
        'технології': ['AI', 'штучний інтелект', 'machine learning', 'ML', 'технологія'],
        'освіта': ['навчання', 'студенти', 'університет', 'освіта', 'викладач'],
        'тренди': ['тренд', 'майбутнє', '2025', '2024', 'інновація']
    }

    found_keywords = {}
    text_lower = text.lower()

    for category, words_list in keywords.items():
        count = sum(1 for word in words_list if word.lower() in text_lower)
        if count > 0:
            found_keywords[category] = count

    analysis = f"""Аналіз тексту:
- Слів: {words}
- Речень: {sentences}
- Ключові теми: {', '.join(found_keywords.keys()) if found_keywords else 'не виявлено'}
- Деталі: {', '.join([f'{k}({v})' for k, v in found_keywords.items()])}"""

    return analysis

def save_report(content: str, filename: str = None) -> str:
    """Збереження звіту у файл"""
    if not filename:
        filename = f"langgraph_report_{datetime.now():%Y%m%d_%H%M%S}.json"

    report_data = {
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "framework": "LangChain + LangGraph"
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    return f"Звіт збережено: {filename}"

# ===========================
# AGENT NODES
# ===========================

class MultiAgentSystem:
    """Мультиагентна система на LangGraph"""

    def __init__(self, api_key: str = None):
        """Ініціалізація системи"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            print("[WARNING] OPENAI_API_KEY не знайдено - працюватиме в демо режимі")
            self.llm = None
        else:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-5-nano",
                    temperature=0.7,
                    api_key=self.api_key
                )
                print("[OK] ChatOpenAI LLM створено")
            except Exception as e:
                print(f"[WARNING] Помилка створення LLM: {e}")
                self.llm = None

        # Створення графу
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_workflow(self) -> StateGraph:
        """Створення графу агентів"""
        workflow = StateGraph(AgentState)

        # Додаємо вузли (агентів)
        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("reporter", self.reporter_node)

        # Визначаємо послідовність виконання
        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "reporter")
        workflow.add_edge("reporter", END)

        return workflow

    def researcher_node(self, state: AgentState) -> AgentState:
        """Агент-дослідник: шукає інформацію"""
        print("\n" + "="*60)
        print("RESEARCHER AGENT: Пошук інформації...")
        print("="*60)

        topic = state["topic"]

        # Виконуємо пошук
        search_results = search_web(topic)
        print(f"\n{search_results[:300]}...")

        # Якщо є LLM - додаємо AI аналіз
        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ви - професійний дослідник AI в освіті.
                    Проаналізуйте знайдену інформацію та виділіть 5 ключових фактів."""),
                    ("human", f"Тема: {topic}\n\nДані:\n{search_results}")
                ])

                chain = prompt | self.llm | StrOutputParser()
                ai_summary = chain.invoke({})
                search_results = f"{search_results}\n\nAI Висновки:\n{ai_summary}"
            except Exception as e:
                print(f"[WARNING] AI обробка недоступна: {e}")

        return {
            "research_results": search_results,
            "messages": ["[OK] Researcher: Пошук завершено"]
        }

    def analyst_node(self, state: AgentState) -> AgentState:
        """Агент-аналітик: аналізує дані"""
        print("\n" + "="*60)
        print("ANALYST AGENT: Аналіз даних...")
        print("="*60)

        research_results = state["research_results"]

        # Виконуємо аналіз
        analysis = analyze_data(research_results)
        print(f"\n{analysis}")

        # Якщо є LLM - додаємо глибокий аналіз
        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ви - експерт з data science та аналізу трендів.
                    Проаналізуйте дані та виявіть ключові інсайти, тренди та закономірності."""),
                    ("human", f"Дані для аналізу:\n{research_results}")
                ])

                chain = prompt | self.llm | StrOutputParser()
                deep_analysis = chain.invoke({})
                analysis = f"{analysis}\n\nГлибокий аналіз:\n{deep_analysis}"
            except Exception as e:
                print(f"[WARNING] AI аналіз недоступний: {e}")
        else:
            # Демо аналіз
            analysis += f"""

Ключові висновки:
- Основний тренд: Персоналізація навчання через AI
- Статистика: 85% університетів впроваджують AI технології
- Виклики: Етика, приватність, підготовка кадрів
- Прогноз: Зростання ринку EdTech на 45% до 2026 року"""

        return {
            "analysis_results": analysis,
            "messages": ["[OK] Analyst: Аналіз завершено"]
        }

    def reporter_node(self, state: AgentState) -> AgentState:
        """Агент-репортер: створює фінальний звіт"""
        print("\n" + "="*60)
        print("REPORTER AGENT: Створення звіту...")
        print("="*60)

        topic = state["topic"]
        research_results = state["research_results"]
        analysis_results = state["analysis_results"]

        # Створюємо базовий звіт
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         LANGGRAPH MULTI-AGENT RESEARCH REPORT                ║
╚══════════════════════════════════════════════════════════════╝

Дата: {datetime.now():%Y-%m-%d %H:%M:%S}
Тема: {topic}
Платформа: LangChain 1.0 + LangGraph

════════════════════════════════════════════════════════════════
РЕЗУЛЬТАТИ ДОСЛІДЖЕННЯ (Researcher Agent)
════════════════════════════════════════════════════════════════

{research_results}

════════════════════════════════════════════════════════════════
АНАЛІТИКА (Analyst Agent)
════════════════════════════════════════════════════════════════

{analysis_results}
"""

        # Якщо є LLM - додаємо професійні висновки
        if self.llm:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ви - професійний технічний письменник.
                    Створіть executive summary та рекомендації на основі дослідження та аналізу."""),
                    ("human", f"Дослідження:\n{research_results}\n\nАналіз:\n{analysis_results}")
                ])

                chain = prompt | self.llm | StrOutputParser()
                summary = chain.invoke({})
                report += f"""
════════════════════════════════════════════════════════════════
EXECUTIVE SUMMARY (Reporter Agent)
════════════════════════════════════════════════════════════════

{summary}
"""
            except Exception as e:
                print(f"[WARNING] AI генерація висновків недоступна: {e}")
        else:
            # Демо висновки
            report += f"""
════════════════════════════════════════════════════════════════
ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ (Reporter Agent)
════════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY:
Дослідження показує активне впровадження AI в освітній процес.
Основний фокус - на персоналізації навчання та автоматизації.

РЕКОМЕНДАЦІЇ:
1. Розробити стратегію впровадження AI технологій
2. Інвестувати в підготовку викладацького складу
3. Створити етичні стандарти використання AI в освіті
4. Забезпечити захист персональних даних студентів

NEXT STEPS:
- Пілотні проекти в 2-3 університетах
- Створення навчальних програм для викладачів
- Розробка локальних AI рішень
"""

        report += f"""
════════════════════════════════════════════════════════════════

[OK] Дослідження завершено успішно
Агенти: Researcher → Analyst → Reporter
Powered by LangChain 1.0 + LangGraph
"""

        # Зберігаємо звіт
        save_status = save_report(report)
        print(f"\n{save_status}")

        return {
            "final_report": report,
            "messages": ["[OK] Reporter: Звіт створено та збережено"]
        }

    def run(self, topic: str) -> dict:
        """Запуск мультиагентної системи"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     LANGGRAPH MULTI-AGENT SYSTEM                             ║
║     LangChain 1.0 + LangGraph                                ║
╚══════════════════════════════════════════════════════════════╝

Тема дослідження: {topic}
Агенти: Researcher → Analyst → Reporter
Граф: StateGraph з послідовним виконанням
        """)

        # Початковий стан
        initial_state = {
            "topic": topic,
            "research_results": "",
            "analysis_results": "",
            "final_report": "",
            "messages": [],
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Запускаємо граф
            final_state = self.app.invoke(initial_state)

            print("\n" + "="*60)
            print("[OK] МУЛЬТИАГЕНТНА СИСТЕМА ЗАВЕРШИЛА РОБОТУ")
            print("="*60)

            # Виводимо повідомлення від агентів
            print("\nЛог виконання:")
            for msg in final_state.get("messages", []):
                print(f"  {msg}")

            # Виводимо фінальний звіт
            print("\n" + final_state["final_report"])

            return final_state

        except Exception as e:
            print(f"\n[ERROR] Помилка виконання: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

# ===========================
# ГОЛОВНА ФУНКЦІЯ
# ===========================

def main():
    """Запуск LangGraph мультиагентної системи"""

    # Перевірка версій
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

    try:
        import openai
        print(f"   [OK] OpenAI: {openai.__version__}")
    except:
        print("   [ERROR] OpenAI: не встановлено")

    # API ключ
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"   [OK] API ключ: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("   [WARNING] API ключ: не знайдено (демо режим)")

    print("\n" + "="*60)

    # Створення системи
    system = MultiAgentSystem(api_key)

    # Запуск дослідження
    topic = "Штучний інтелект в освіті України 2025: можливості та виклики"
    result = system.run(topic)

    if "error" not in result:
        print("\nГотово! Перегляньте файли:")
        print("   langgraph_report_*.json - повний звіт")
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
