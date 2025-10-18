"""
CrewAI Agent БЕЗ LangChain залежностей
Чистий CrewAI для студентів - простіше і зрозуміліше
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# ===========================
# ПРОСТИЙ CREWAI АГЕНТ
# ===========================

class SimpleCrewAIAgent:
    """
    Чистий CrewAI агент без LangChain
    Використовує вбудовані можливості CrewAI
    """
    
    def __init__(self):
        """Ініціалізація агента"""
        # CrewAI автоматично використає OPENAI_API_KEY з середовища
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List:
        """Створення інструментів для дослідження"""
        
        @tool("Web Search")
        def search_web(query: str) -> str:
            """Пошук інформації в інтернеті"""
            try:
                from duckduckgo_search import DDGS
                results_text = f"Результати пошуку для '{query}':\n\n"
                
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                    for i, r in enumerate(results, 1):
                        results_text += f"{i}. {r['title']}\n"
                        results_text += f"   {r['body'][:150]}...\n"
                        results_text += f"   Джерело: {r.get('href', 'N/A')}\n\n"
                
                return results_text
            except Exception as e:
                return f"Помилка пошуку: {e}. Використайте демо дані."
        
        @tool("Data Analysis")
        def analyze_data(text: str) -> str:
            """Аналіз тексту та статистика"""
            words = text.split()
            sentences = text.count('.') + text.count('!') + text.count('?')
            
            # Аналіз тональності
            positive = ["успіх", "покращення", "інновація", "прогрес", "розвиток"]
            negative = ["проблема", "виклик", "ризик", "загроза", "складність"]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive if word in text_lower)
            neg_count = sum(1 for word in negative if word in text_lower)
            
            sentiment = "позитивна" if pos_count > neg_count else "негативна" if neg_count > pos_count else "нейтральна"
            
            return f"""
            Аналіз даних:
            - Слів: {len(words)}
            - Речень: {sentences}
            - Тональність: {sentiment}
            - Позитивних слів: {pos_count}
            - Негативних слів: {neg_count}
            """
        
        @tool("Save Report")
        def save_report(content: str) -> str:
            """Зберегти звіт у файл"""
            filename = f"crewai_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            
            report = {
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "agent": "CrewAI Research Agent"
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return f"Звіт збережено в {filename}"
        
        @tool("Get Date")
        def get_date() -> str:
            """Отримати поточну дату"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return [search_web, analyze_data, save_report, get_date]
    
    def _create_agent(self) -> Agent:
        """Створення агента-дослідника"""
        return Agent(
            role='Дослідник AI',
            goal='Зібрати та проаналізувати інформацію про AI в освіті',
            backstory="""Ви - експерт з AI та освітніх технологій.
            Ваше завдання - знайти актуальну інформацію, проаналізувати її
            та створити корисний звіт для викладачів та студентів.""",
            tools=self.tools,
            verbose=True,  # Показувати процес роботи
            max_iter=5,    # Максимум 5 ітерацій
            memory=True    # Використовувати пам'ять
        )
    
    def research(self, topic: str) -> Dict[str, Any]:
        """Провести дослідження"""
        print(f"\n🚀 CrewAI: Досліджуємо '{topic}'")
        print("=" * 60)
        
        # Створюємо задачу
        task = Task(
            description=f"""
            Дослідіть тему: {topic}
            
            План дій:
            1. Використайте Web Search для пошуку інформації
            2. Проаналізуйте знайдені дані через Data Analysis
            3. Створіть структурований звіт
            4. Збережіть результати через Save Report
            
            Звіт повинен містити:
            - Основні факти
            - Тренди та тенденції
            - Переваги та недоліки
            - Рекомендації
            """,
            expected_output="""
            Структурований звіт українською мовою з:
            - Резюме (2-3 речення)
            - Основні висновки (3-5 пунктів)
            - Рекомендації (2-3 пункти)
            """,
            agent=self.agent
        )
        
        # Створюємо команду (crew)
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            process=Process.sequential,  # Послідовне виконання
            verbose=True
        )
        
        try:
            # Запускаємо дослідження
            result = crew.kickoff()
            
            # Формуємо результат
            return {
                "success": True,
                "topic": topic,
                "result": str(result),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "topic": topic,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ===========================
# МУЛЬТИАГЕНТНА СИСТЕМА
# ===========================

class MultiAgentTeam:
    """Команда з декількох агентів"""
    
    def __init__(self):
        """Створюємо команду агентів"""
        self.tools = self._create_tools()
        self.researcher = self._create_researcher()
        self.analyst = self._create_analyst()
        self.writer = self._create_writer()
    
    def _create_tools(self) -> List:
        """Спільні інструменти для всіх агентів"""
        
        @tool("Search")
        def search(query: str) -> str:
            """Пошук інформації"""
            return f"Знайдено інформацію про: {query}"
        
        @tool("Analyze")
        def analyze(data: str) -> str:
            """Аналіз даних"""
            return f"Проаналізовано: {len(data)} символів"
        
        return [search, analyze]
    
    def _create_researcher(self) -> Agent:
        """Агент-дослідник"""
        return Agent(
            role='Дослідник',
            goal='Знайти актуальну інформацію',
            backstory='Експерт з пошуку даних',
            tools=self.tools,
            verbose=True
        )
    
    def _create_analyst(self) -> Agent:
        """Агент-аналітик"""
        return Agent(
            role='Аналітик',
            goal='Проаналізувати зібрані дані',
            backstory='Експерт з аналізу даних',
            tools=self.tools,
            verbose=True
        )
    
    def _create_writer(self) -> Agent:
        """Агент-письменник"""
        return Agent(
            role='Письменник',
            goal='Створити зрозумілий звіт',
            backstory='Експерт з написання звітів',
            verbose=True
        )
    
    def research_together(self, topic: str) -> Dict:
        """Командна робота над дослідженням"""
        print(f"\n👥 Мультиагентна команда: '{topic}'")
        print("=" * 60)
        
        # Задача 1: Пошук
        task1 = Task(
            description=f"Знайдіть інформацію про: {topic}",
            expected_output="Список знайдених фактів",
            agent=self.researcher
        )
        
        # Задача 2: Аналіз
        task2 = Task(
            description="Проаналізуйте зібрану інформацію",
            expected_output="Аналітичний висновок",
            agent=self.analyst
        )
        
        # Задача 3: Звіт
        task3 = Task(
            description="Напишіть зрозумілий звіт",
            expected_output="Фінальний звіт",
            agent=self.writer
        )
        
        # Створюємо команду
        crew = Crew(
            agents=[self.researcher, self.analyst, self.writer],
            tasks=[task1, task2, task3],
            process=Process.sequential
        )
        
        # Запускаємо
        result = crew.kickoff()
        
        return {
            "topic": topic,
            "team_result": str(result),
            "agents_used": 3
        }

# ===========================
# ГОЛОВНА ПРОГРАМА
# ===========================

def main():
    """Демонстрація CrewAI без LangChain"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║               CREWAI AGENT (БЕЗ LANGCHAIN)                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Перевірка API ключа
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API ключ знайдено")
    else:
        print("⚠️ Додайте OPENAI_API_KEY в .env файл")
        print("   CrewAI спробує працювати в обмеженому режимі")
    
    # 1. Простий агент
    print("\n1️⃣ ОДИН АГЕНТ:")
    agent = SimpleCrewAIAgent()
    result = agent.research("AI асистенти для студентів")
    
    if result["success"]:
        print(f"✅ Дослідження завершено")
        print(f"📝 Результат: {result['result'][:300]}...")
    else:
        print(f"❌ Помилка: {result.get('error')}")
    
    # 2. Команда агентів (демо)
    print("\n2️⃣ КОМАНДА АГЕНТІВ:")
    team = MultiAgentTeam()
    team_result = team.research_together("Майбутнє онлайн освіти")
    print(f"✅ Команда з {team_result['agents_used']} агентів завершила роботу")
    
    print("\n" + "=" * 60)
    print("💡 Підказки:")
    print("- Перегляньте crewai_report_*.json для повних результатів")
    print("- Додайте OPENAI_API_KEY для повного функціоналу")
    print("- Спробуйте змінити ролі та завдання агентів")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 До побачення!")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        print("\n📦 Встановіть залежності:")
        print("pip install crewai crewai-tools duckduckgo-search")
