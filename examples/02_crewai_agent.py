"""
Модуль 1: AI Research Agent на CrewAI (v1.0)
Той самий агент-дослідник для порівняння з LangChain
"""

import os
from typing import Dict, List, Any
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from datetime import datetime
import json

# ===========================
# БАЗОВИЙ АГЕНТ-ДОСЛІДНИК
# ===========================

class CrewAIResearchAgent:
    """
    Агент-дослідник на CrewAI v1.0
    Використовує role-based архітектуру
    """
    
    def __init__(self, api_key: str = None):
        """Ініціалізація агента"""
        # LLM для агента
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Створення інструментів
        self.tools = self._create_tools()
        
        # Створення агента
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List:
        """Створення набору інструментів для дослідження"""
        
        @tool("Web Search")
        def search_web(query: str) -> str:
            """
            Пошук інформації в інтернеті.
            Використовується для знаходження актуальних даних.
            """
            from duckduckgo_search import DDGS
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                    formatted = "\n".join([f"- {r['title']}: {r['body'][:200]}..." for r in results])
                    return f"Результати пошуку:\n{formatted}"
            except:
                return "Пошук недоступний"
        
        @tool("Get Current Date")
        def get_current_date() -> str:
            """
            Отримати поточну дату та час.
            Корисно для часових міток та планування.
            """
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @tool("Sentiment Analysis")
        def analyze_sentiment(text: str) -> str:
            """
            Аналіз тональності тексту.
            Визначає емоційне забарвлення інформації.
            """
            positive_words = ["добре", "чудово", "успіх", "позитив", "інновація"]
            negative_words = ["проблема", "виклик", "ризик", "складність"]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "позитивна"
            elif neg_count > pos_count:
                sentiment = "негативна"
            else:
                sentiment = "нейтральна"
            
            return f"Тональність: {sentiment} (позитив: {pos_count}, негатив: {neg_count})"
        
        @tool("Save to Memory")
        def save_to_memory(data: str) -> str:
            """
            Зберегти важливу інформацію в пам'ять.
            Дозволяє зберігати ключові факти для подальшого використання.
            """
            memory_file = "crewai_memory.json"
            try:
                with open(memory_file, 'r') as f:
                    memory = json.load(f)
            except:
                memory = {"facts": []}
            
            memory["facts"].append({
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
            with open(memory_file, 'w') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            return f"Збережено в пам'ять: {data[:50]}..."
        
        return [search_web, get_current_date, analyze_sentiment, save_to_memory]
    
    def _create_agent(self) -> Agent:
        """Створення агента-дослідника"""
        agent = Agent(
            role='Професійний дослідник',
            goal='Зібрати та проаналізувати актуальну інформацію за темою дослідження',
            backstory="""Ви - досвідчений дослідник з 10-річним досвідом роботи 
            в аналітичних центрах. Ваша експертиза включає збір даних, 
            аналіз трендів та формування обґрунтованих висновків.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )
        return agent
    
    def research(self, topic: str) -> Dict[str, Any]:
        """Виконати дослідження на задану тему"""
        print(f"\n🔍 CrewAI Agent: Починаю дослідження теми '{topic}'")
        print("=" * 60)
        
        # Створення задачі дослідження
        research_task = Task(
            description=f"""
            Проведіть комплексне дослідження на тему: {topic}
            
            Ваші кроки:
            1. Знайдіть актуальну інформацію через веб-пошук
            2. Проаналізуйте тональність знайденої інформації
            3. Збережіть ключові факти в пам'ять
            4. Сформуйте структурований висновок
            """,
            expected_output="Детальний звіт дослідження з висновками",
            agent=self.agent
        )
        
        # Створення екіпажу з одним агентом
        crew = Crew(
            agents=[self.agent],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            # Виконання дослідження
            result = crew.kickoff()
            
            return {
                "topic": topic,
                "result": str(result),
                "agent_role": self.agent.role,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "topic": topic,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ===========================
# ДЕМОНСТРАЦІЯ
# ===========================

def main():
    """Демонстрація роботи агентів CrewAI"""
    
    research_topic = "Штучний інтелект в освіті 2025"
    
    print("\n🚢 CREWAI AGENT DEMO")
    print("=" * 60)
    
    agent = CrewAIResearchAgent()
    result = agent.research(research_topic)
    
    print(f"\n📄 Результат:")
    print(f"Тема: {result['topic']}")
    print(f"Роль агента: {result.get('agent_role', 'N/A')}")
    print(f"Результат: {result.get('result', 'Немає результату')[:300]}...")

if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    
    try:
        main()
    except Exception as e:
        print(f"❌ Помилка: {e}")
        print("\nПереконайтеся, що встановлено:")
        print("1. OPENAI_API_KEY")
        print("2. pip install crewai crewai-tools")
