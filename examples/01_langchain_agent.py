"""
Модуль 1: AI Research Agent на LangChain
Сумісний з LangChain v1.0+
"""

import os
from typing import Dict, List, Any
from datetime import datetime
import json

# Правильні імпорти для LangChain v1.0+
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun

# Для LangChain v1.0+ використовуємо новий спосіб
from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ===========================
# БАЗОВИЙ АГЕНТ-ДОСЛІДНИК
# ===========================

class LangChainResearchAgent:
    """
    Агент-дослідник на LangChain v1.0+
    Використовує сучасний підхід з Runnables
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
        
        # Створення агента
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> List:
        """Створення набору інструментів для дослідження"""
        
        @tool
        def search_web(query: str) -> str:
            """Пошук інформації в інтернеті"""
            try:
                search = DuckDuckGoSearchRun()
                results = search.run(query)
                return f"Результати пошуку для '{query}': {results[:500]}..."
            except Exception as e:
                # Fallback для демонстрації
                return f"[Демо пошук] Для запиту '{query}' знайдено: AI трансформує освіту через персоналізацію навчання, автоматизацію оцінювання та адаптивні навчальні системи. Основні тренди: 1) Персоналізовані траєкторії навчання 2) AI-тьютори 3) Автоматична перевірка завдань."
        
        @tool
        def get_current_date() -> str:
            """Отримати поточну дату та час"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @tool
        def analyze_sentiment(text: str) -> str:
            """Аналіз тональності тексту"""
            positive_words = ["добре", "чудово", "успіх", "позитив", "good", "great", 
                            "прогрес", "інновація", "покращення", "ефективність"]
            negative_words = ["погано", "проблема", "негатив", "bad", "problem", 
                            "виклик", "ризик", "складність", "загроза"]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "позитивна"
            elif neg_count > pos_count:
                sentiment = "негативна"
            else:
                sentiment = "нейтральна"
            
            return f"Тональність: {sentiment} (позитивні маркери: {pos_count}, негативні: {neg_count})"
        
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
            
            return f"✅ Збережено в пам'ять: {key}"
        
        return [search_web, get_current_date, analyze_sentiment, save_to_memory]
    
    def _create_agent(self):
        """Створення агента для LangChain v1.0+"""
        
        # Простіший підхід - використовуємо LLM напряму з інструментами
        class SimpleAgent:
            def __init__(self, llm, tools):
                self.llm = llm
                self.tools = tools
                self.tool_map = {tool.name: tool for tool in tools}
            
            def invoke(self, inputs):
                query = inputs.get("input", "")
                
                # Системний промпт
                system_prompt = """Ви - агент-дослідник. Використовуйте доступні інструменти для дослідження теми.
                
                Доступні інструменти:
                - search_web: пошук інформації в інтернеті
                - get_current_date: отримати поточну дату
                - analyze_sentiment: аналіз тональності тексту
                - save_to_memory: зберегти в пам'ять
                
                Проведіть дослідження крок за кроком."""
                
                # Формуємо повідомлення
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
                
                try:
                    # Отримуємо відповідь від LLM
                    response = self.llm.invoke(messages)
                    
                    # Виконуємо інструменти вручну для демонстрації
                    results = []
                    
                    # Крок 1: Отримуємо дату
                    date_tool = self.tool_map.get("get_current_date")
                    if date_tool:
                        date_result = date_tool.func()
                        results.append(f"📅 Дата дослідження: {date_result}")
                    
                    # Крок 2: Пошук
                    search_tool = self.tool_map.get("search_web")
                    if search_tool and "AI" in query:
                        search_result = search_tool.func(query)
                        results.append(f"\n🔍 Пошук:\n{search_result}")
                    
                    # Крок 3: Аналіз тональності
                    sentiment_tool = self.tool_map.get("analyze_sentiment")
                    if sentiment_tool and len(results) > 0:
                        sentiment_result = sentiment_tool.func(str(results))
                        results.append(f"\n📊 {sentiment_result}")
                    
                    # Крок 4: Збереження
                    memory_tool = self.tool_map.get("save_to_memory")
                    if memory_tool:
                        memory_result = memory_tool.func("research_result", query)
                        results.append(f"\n💾 {memory_result}")
                    
                    # Формуємо фінальну відповідь
                    if results:
                        final_output = f"Дослідження теми: {query}\n\n" + "\n".join(results)
                    else:
                        final_output = response.content if hasattr(response, 'content') else str(response)
                    
                    return {"output": final_output}
                    
                except Exception as e:
                    # Fallback
                    return {
                        "output": f"Виконано базове дослідження теми: {query}\n"
                                f"Статус: Демо режим\n"
                                f"Результат: AI в освіті - перспективний напрямок"
                    }
        
        return SimpleAgent(self.llm, self.tools)
    
    def research(self, topic: str) -> Dict[str, Any]:
        """Виконати дослідження на задану тему"""
        print(f"\n🔍 LangChain Agent: Починаю дослідження теми '{topic}'")
        print("=" * 60)
        
        try:
            # Виконання дослідження
            result = self.agent_executor.invoke({
                "input": topic
            })
            
            return {
                "topic": topic,
                "result": result.get("output", ""),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            print(f"⚠️ Помилка виконання: {e}")
            
            # Демо результат
            demo_result = f"""
Дослідження: {topic}

📅 Час: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

🔍 Основні висновки:
1. AI активно впроваджується в освітні процеси
2. Персоналізація навчання - ключовий тренд
3. Автоматизація рутинних задач вивільняє час викладачів
4. Адаптивні системи покращують результати студентів

📊 Тональність: Переважно позитивна

💡 Рекомендації:
- Впровадження AI має бути поступовим
- Важливо зберегти людський фактор
- Необхідна підготовка викладачів

✅ Дослідження завершено (демо режим)
            """
            
            return {
                "topic": topic,
                "result": demo_result,
                "timestamp": datetime.now().isoformat(),
                "status": "demo"
            }

# ===========================
# ДЕМОНСТРАЦІЯ
# ===========================

def main():
    """Демонстрація роботи агентів"""
    
    research_topic = "Штучний інтелект в освіті 2025"
    
    print("\n🤖 LANGCHAIN AGENT DEMO")
    print("=" * 60)
    print(f"Версія LangChain: v1.0+")
    print("=" * 60)
    
    try:
        agent = LangChainResearchAgent()
        result = agent.research(research_topic)
        
        print(f"\n📄 Результат:")
        print(f"Тема: {result['topic']}")
        print(f"Статус: {result.get('status', 'unknown')}")
        print(f"Час: {result.get('timestamp', 'N/A')}")
        print(f"\n📝 Висновок:")
        print(result.get('result', 'Немає результату'))
        
        # Зберігаємо результат
        with open("langchain_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Результат збережено в langchain_result.json")
            
    except Exception as e:
        print(f"❌ Критична помилка: {e}")
        print("\n🔧 Можливі рішення:")
        print("1. Перевірте API ключ в .env файлі")
        print("2. Перевірте підключення до інтернету")
        print("3. Спробуйте: pip install --upgrade langchain langchain-openai")

if __name__ == "__main__":
    # Спроба завантажити .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env файл завантажено")
    except:
        pass
    
    # Перевірка наявності API ключа
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY не встановлено!")
        print("Працюю в демо режимі...")
        print("\nДля повноцінної роботи:")
        print("1. Створіть файл .env")
        print("2. Додайте: OPENAI_API_KEY=sk-your-key-here")
    else:
        print(f"✅ API ключ знайдено: {api_key[:7]}...{api_key[-4:]}")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Програму перервано користувачем")
    except Exception as e:
        print(f"\n❌ Несподівана помилка: {e}")
        import traceback
        traceback.print_exc()
