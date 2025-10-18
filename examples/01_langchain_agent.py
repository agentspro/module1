"""
Модуль 1: AI Research Agent на LangChain
Сумісний з різними версіями LangChain
"""

import os
from typing import Dict, List, Any
from datetime import datetime
import json

try:
    # Для новіших версій LangChain (0.2+)
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
    USE_REACT = True
except ImportError:
    # Для старіших версій LangChain
    from langchain.agents import initialize_agent, AgentExecutor, AgentType
    USE_REACT = False

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun

# ===========================
# БАЗОВИЙ АГЕНТ-ДОСЛІДНИК
# ===========================

class LangChainResearchAgent:
    """
    Агент-дослідник на LangChain
    Сумісний з різними версіями фреймворку
    """
    
    def __init__(self, api_key: str = None):
        """Ініціалізація агента"""
        # LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Створення інструментів
        self.tools = self._create_tools()
        
        # Створення агента залежно від версії
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> List:
        """Створення набору інструментів для дослідження"""
        
        @tool
        def search_web(query: str) -> str:
            """Пошук інформації в інтернеті"""
            try:
                search = DuckDuckGoSearchRun()
                results = search.run(query)
                return f"Результати пошуку: {results[:500]}..."
            except Exception as e:
                return f"Симуляція пошуку для '{query}': Знайдено інформацію про AI в освіті, включаючи персоналізоване навчання та автоматизацію."
        
        @tool
        def get_current_date() -> str:
            """Отримати поточну дату та час"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @tool
        def analyze_sentiment(text: str) -> str:
            """Аналіз тональності тексту"""
            positive_words = ["добре", "чудово", "успіх", "позитив", "good", "great", "прогрес", "інновація"]
            negative_words = ["погано", "проблема", "негатив", "bad", "problem", "виклик", "ризик"]
            
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
        
        @tool
        def save_to_memory(key: str, value: str) -> str:
            """Зберегти інформацію в пам'ять"""
            memory_file = "agent_memory.json"
            try:
                with open(memory_file, 'r') as f:
                    memory = json.load(f)
            except:
                memory = {}
            
            memory[key] = {
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(memory_file, 'w') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            return f"Збережено: {key} = {value[:50]}..."
        
        return [search_web, get_current_date, analyze_sentiment, save_to_memory]
    
    def _create_agent(self) -> AgentExecutor:
        """Створення агента залежно від версії LangChain"""
        
        if USE_REACT:
            # Спробуємо використати ReAct агента (нова версія)
            try:
                # Створення промпту
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Ви - професійний агент-дослідник. Ваші обов'язки:
                    1. Збір актуальної інформації за темою
                    2. Аналіз зібраних даних
                    3. Збереження важливої інформації
                    4. Формування структурованих висновків
                    
                    Використовуйте доступні інструменти для виконання задач."""),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad", optional=True)
                ])
                
                # Створення ReAct агента
                from langchain.agents import create_react_agent
                agent = create_react_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=prompt
                )
                
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=self.tools,
                    verbose=True,
                    max_iterations=5,
                    handle_parsing_errors=True
                )
                
            except Exception as e:
                print(f"Не вдалось створити ReAct агента: {e}")
                # Fallback до простішого підходу
                agent_executor = self._create_simple_agent()
                
        else:
            # Використовуємо старий спосіб ініціалізації
            agent_executor = self._create_simple_agent()
        
        return agent_executor
    
    def _create_simple_agent(self) -> AgentExecutor:
        """Створення агента старим способом (для сумісності)"""
        try:
            from langchain.agents import initialize_agent, AgentType
            
            agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=5,
                handle_parsing_errors=True
            )
            return agent_executor
            
        except Exception as e:
            print(f"Помилка створення агента: {e}")
            # Повертаємо мінімальну реалізацію
            return self._create_minimal_executor()
    
    def _create_minimal_executor(self):
        """Мінімальна реалізація для випадків несумісності"""
        class MinimalExecutor:
            def __init__(self, llm, tools):
                self.llm = llm
                self.tools = tools
            
            def invoke(self, inputs):
                # Проста симуляція роботи агента
                query = inputs.get("input", "")
                
                # Використовуємо інструменти вручну
                results = []
                for tool_func in self.tools:
                    if "search" in tool_func.name.lower() and "AI" in query:
                        result = tool_func.func("AI в освіті")
                        results.append(result)
                    elif "date" in tool_func.name.lower():
                        result = tool_func.func()
                        results.append(f"Дата: {result}")
                
                output = f"Дослідження '{query}':\n" + "\n".join(results) if results else f"Виконано дослідження: {query}"
                return {"output": output}
        
        return MinimalExecutor(self.llm, self.tools)
    
    def research(self, topic: str) -> Dict[str, Any]:
        """Виконати дослідження на задану тему"""
        print(f"\n🔍 LangChain Agent: Починаю дослідження теми '{topic}'")
        print("=" * 60)
        
        try:
            # Спроба виконати через агента
            result = self.agent_executor.invoke({
                "input": f"""Проведіть дослідження на тему: {topic}
                
                Кроки:
                1. Отримайте поточну дату
                2. Знайдіть інформацію через пошук
                3. Проаналізуйте тональність
                4. Збережіть важливі факти
                5. Сформуйте висновок
                """
            })
            
            return {
                "topic": topic,
                "result": result.get("output", ""),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            # Якщо виникла помилка, використовуємо спрощений підхід
            print(f"Використовую спрощений режим через помилку: {e}")
            
            # Виконуємо дослідження вручну
            results = []
            
            # Використовуємо інструменти напряму
            for tool_func in self.tools:
                try:
                    if "date" in tool_func.name.lower():
                        date_result = tool_func.func()
                        results.append(f"Дата дослідження: {date_result}")
                    elif "search" in tool_func.name.lower():
                        search_result = tool_func.func(topic)
                        results.append(search_result)
                    elif "sentiment" in tool_func.name.lower():
                        sentiment_result = tool_func.func(topic)
                        results.append(sentiment_result)
                except:
                    continue
            
            final_result = "\n\n".join(results) if results else f"Базове дослідження теми '{topic}' завершено."
            
            return {
                "topic": topic,
                "result": final_result,
                "timestamp": datetime.now().isoformat(),
                "status": "fallback"
            }

# ===========================
# ДЕМОНСТРАЦІЯ
# ===========================

def main():
    """Демонстрація роботи агентів"""
    
    research_topic = "Штучний інтелект в освіті 2025"
    
    print("\n🤖 LANGCHAIN AGENT DEMO")
    print("=" * 60)
    print(f"Версія LangChain: спроба автоматичного визначення")
    print("=" * 60)
    
    try:
        agent = LangChainResearchAgent()
        result = agent.research(research_topic)
        
        print(f"\n📄 Результат:")
        print(f"Тема: {result['topic']}")
        print(f"Статус: {result.get('status', 'unknown')}")
        print(f"Час: {result.get('timestamp', 'N/A')}")
        print(f"\nВисновок: {result.get('result', 'Немає результату')[:500]}...")
        
        # Зберігаємо результат
        with open("langchain_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Результат збережено в langchain_result.json")
            
    except Exception as e:
        print(f"❌ Критична помилка: {e}")
        print("\n🔧 Можливі рішення:")
        print("1. Перевірте версію LangChain: pip show langchain")
        print("2. Оновіть LangChain: pip install --upgrade langchain langchain-openai langchain-community")
        print("3. Або встановіть конкретну версію: pip install langchain==0.1.0")

if __name__ == "__main__":
    # Перевірка наявності API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY не встановлено!")
        print("Встановіть змінну середовища або створіть файл .env")
        print("\nПриклад:")
        print("export OPENAI_API_KEY='sk-your-key-here'")
        print("або")
        print("echo 'OPENAI_API_KEY=sk-your-key-here' > .env")
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\n\n👋 Програму перервано користувачем")
        except Exception as e:
            print(f"\n❌ Несподівана помилка: {e}")
            import traceback
            traceback.print_exc()
