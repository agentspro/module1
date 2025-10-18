"""
Модуль 1: AI Research Agent на LangChain (v1.0)
Єдиний приклад агента-дослідника для порівняння фреймворків
"""

import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import json

# ===========================
# БАЗОВИЙ АГЕНТ-ДОСЛІДНИК
# ===========================

class LangChainResearchAgent:
    """
    Агент-дослідник на LangChain v1.0
    Використовує нову архітектуру з tool calling
    """
    
    def __init__(self, api_key: str = None):
        """Ініціалізація агента"""
        # LLM з підтримкою tool calling
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Створення інструментів
        self.tools = self._create_tools()
        
        # Створення промпту
        self.prompt = self._create_prompt()
        
        # Створення агента з новою архітектурою
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List:
        """Створення набору інструментів для дослідження"""
        
        @tool
        def search_web(query: str) -> str:
            """Пошук інформації в інтернеті"""
            search = DuckDuckGoSearchRun()
            results = search.run(query)
            return f"Результати пошуку: {results[:500]}..."
        
        @tool
        def get_current_date() -> str:
            """Отримати поточну дату та час"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @tool
        def analyze_sentiment(text: str) -> Dict:
            """Аналіз тональності тексту"""
            # Спрощена імітація аналізу
            positive_words = ["добре", "чудово", "успіх", "позитив", "good", "great"]
            negative_words = ["погано", "проблема", "негатив", "bad", "problem"]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "позитивна"
            elif neg_count > pos_count:
                sentiment = "негативна"
            else:
                sentiment = "нейтральна"
            
            return {
                "sentiment": sentiment,
                "positive_score": pos_count,
                "negative_score": neg_count
            }
        
        @tool
        def save_to_memory(key: str, value: str) -> str:
            """Зберегти інформацію в пам'ять"""
            memory_file = "agent_memory.json"
            try:
                with open(memory_file, 'r') as f:
                    memory = json.load(f)
            except:
                memory = {}
            
            memory[key] = value
            
            with open(memory_file, 'w') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            return f"Збережено: {key}"
        
        return [search_web, get_current_date, analyze_sentiment, save_to_memory]
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Створення промпту для агента"""
        system_prompt = """Ви - професійний агент-дослідник. Ваші обов'язки:
        1. Збір актуальної інформації за темою
        2. Аналіз зібраних даних
        3. Збереження важливої інформації
        4. Формування структурованих висновків
        
        Використовуйте доступні інструменти для виконання задач."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        return prompt
    
    def _create_agent(self) -> AgentExecutor:
        """Створення агента з новою архітектурою LangChain"""
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def research(self, topic: str) -> Dict[str, Any]:
        """Виконати дослідження на задану тему"""
        print(f"\n🔍 LangChain Agent: Починаю дослідження теми '{topic}'")
        print("=" * 60)
        
        try:
            result = self.agent.invoke({
                "input": f"Проведіть дослідження на тему: {topic}"
            })
            
            return {
                "topic": topic,
                "result": result.get("output", ""),
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
    """Демонстрація роботи агентів"""
    
    research_topic = "Штучний інтелект в освіті 2025"
    
    print("\n🤖 LANGCHAIN AGENT DEMO")
    print("=" * 60)
    
    agent = LangChainResearchAgent()
    result = agent.research(research_topic)
    
    print(f"\n📄 Результат:")
    print(f"Тема: {result['topic']}")
    print(f"Висновок: {result.get('result', 'Немає результату')[:300]}...")

if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    
    try:
        main()
    except Exception as e:
        print(f"❌ Помилка: {e}")
        print("Переконайтеся, що встановлено OPENAI_API_KEY")
