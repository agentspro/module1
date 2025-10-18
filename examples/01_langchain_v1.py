"""
Модуль 1: AI Research Agent на LangChain 1.0
Використовує найновіші версії всіх пакетів
"""

import os
import sys
from datetime import datetime
import json
from typing import Dict, List, Any

# Завантаження .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(".env файл завантажено")
except:
    print("python-dotenv не встановлено")

# Імпорти для LangChain 1.0
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    print("LangChain 1.0 компоненти завантажено")
except ImportError as e:
    print(f"Помилка імпорту LangChain: {e}")
    print("Встановіть: pip install langchain-openai langchain-core")
    sys.exit(1)

# ===========================
# LANGCHAIN 1.0 AGENT
# ===========================

class LangChain1Agent:
    """
    Агент-дослідник на LangChain 1.0
    Використовує нову архітектуру LCEL (LangChain Expression Language)
    """
    
    def __init__(self, api_key: str = None):
        """Ініціалізація агента"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("OPENAI_API_KEY не знайдено!")
            self.llm = None
        else:
            try:
                # Створення LLM для LangChain 1.0
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.7,
                    api_key=self.api_key
                )
                print(f" ChatOpenAI LLM створено")
            except Exception as e:
                print(f" Помилка створення LLM: {e}")
                self.llm = None
        
        # Створення інструментів
        self.tools = self._create_tools()
        
        # Створення ланцюгів (chains) - нова архітектура LangChain 1.0
        self.chains = self._create_chains()
    
    def _create_tools(self) -> Dict:
        """Створення інструментів для дослідження"""
        
        def search_web(query: str) -> str:
            """Пошук інформації в інтернеті"""
            try:
                from ddgs import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                    if results:
                        output = "🔍 Результати пошуку:\n\n"
                        for i, r in enumerate(results, 1):
                            output += f"{i}. {r.get('title', '')}\n"
                            output += f"   {r.get('body', '')[:200]}...\n"
                            output += f"   {r.get('href', '')}\n\n"
                        return output
            except Exception as e:
                print(f" Помилка пошуку: {e}")
            
            # Демо результат
            return f"""
Демо результати для '{query}':

1. AI трансформує освіту через персоналізацію
   Штучний інтелект дозволяє створювати індивідуальні навчальні траєкторії...

2. Статистика 2025: 85% закладів використовують AI
   За даними дослідження, більшість навчальних закладів впровадили AI-рішення...

3. Виклики впровадження AI в освіті
   Основні проблеми: підготовка кадрів, етичні питання, доступність...
"""
        
        def analyze_data(text: str) -> str:
            """Аналіз даних"""
            word_count = len(text.split())
            sentences = text.count('.') + text.count('!') + text.count('?')
            
            # Простий sentiment аналіз
            positive = ["успіх", "покращення", "інновація", "прогрес", "ефективність"]
            negative = ["проблема", "виклик", "ризик", "загроза", "складність"]
            
            pos_count = sum(1 for word in positive if word in text.lower())
            neg_count = sum(1 for word in negative if word in text.lower())
            
            sentiment = "позитивний" if pos_count > neg_count else "негативний" if neg_count > pos_count else "нейтральний"
            
            return f"""
Аналіз даних:
- Слів: {word_count}
- Речень: {sentences}
- Тональність: {sentiment}
- Позитивних маркерів: {pos_count}
- Негативних маркерів: {neg_count}
"""
        
        def save_to_memory(data: dict) -> str:
            """Збереження в пам'ять"""
            filename = "langchain1_memory.json"
            try:
                with open(filename, 'r') as f:
                    memory = json.load(f)
            except:
                memory = {"sessions": []}
            
            memory["sessions"].append(data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            return f"Збережено в {filename}"
        
        return {
            "search_web": search_web,
            "analyze_data": analyze_data,
            "save_to_memory": save_to_memory
        }
    
    def _create_chains(self):
        """Створення ланцюгів для LangChain 1.0 LCEL"""
        chains = {}
        
        if self.llm:
            # Промпт для дослідження
            research_prompt = ChatPromptTemplate.from_messages([
                ("system", "Ви - професійний дослідник AI в освіті. Аналізуйте надану інформацію та створіть структурований звіт."),
                ("human", "Тема: {topic}\n\nДані:\n{data}\n\nСтворіть детальний аналіз.")
            ])
            
            # Ланцюг дослідження (LCEL синтаксис)
            chains["research"] = research_prompt | self.llm | StrOutputParser()
            
            # Ланцюг для висновків
            conclusion_prompt = ChatPromptTemplate.from_messages([
                ("system", "Ви експерт з формування висновків. Узагальніть інформацію."),
                ("human", "{analysis}")
            ])
            
            chains["conclusion"] = conclusion_prompt | self.llm | StrOutputParser()
        
        return chains
    
    async def research_async(self, topic: str) -> dict:
        """Асинхронне дослідження (нова функція LangChain 1.0)"""
        # LangChain 1.0 підтримує async операції
        pass
    
    def research(self, topic: str) -> dict:
        """Синхронне дослідження"""
        print(f"\nLangChain 1.0 Agent: Дослідження '{topic}'")
        print("=" * 60)
        
        results = {"topic": topic, "timestamp": datetime.now().isoformat()}
        
        # Крок 1: Пошук
        print("Крок 1: Пошук інформації...")
        search_results = self.tools["search_web"](topic)
        results["search"] = search_results
        print("   Завершено")
        
        # Крок 2: Аналіз
        print("Крок 2: Аналіз даних...")
        analysis = self.tools["analyze_data"](search_results)
        results["analysis"] = analysis
        print("   Завершено")
        
        # Крок 3: AI обробка (якщо доступна)
        if self.llm and "research" in self.chains:
            print("Крок 3: AI аналіз...")
            try:
                ai_analysis = self.chains["research"].invoke({
                    "topic": topic,
                    "data": search_results
                })
                results["ai_analysis"] = ai_analysis
                print("   Завершено")
            except Exception as e:
                print(f"   Помилка AI: {e}")
                results["ai_analysis"] = "AI аналіз недоступний"
        else:
            print(" Крок 3: Демо аналіз...")
            results["ai_analysis"] = self._demo_analysis(topic)
            print("   Завершено")
        
        # Крок 4: Збереження
        print(" Крок 4: Збереження результатів...")
        save_result = self.tools["save_to_memory"](results)
        print(f"   {save_result}")
        
        # Створення звіту
        report = self._create_report(results)
        results["report"] = report
        
        # Збереження фінального звіту
        with open("langchain1_report.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\nПовний звіт: langchain1_report.json")
        
        return results
    
    def _demo_analysis(self, topic: str) -> str:
        """Демо аналіз для випадків без API"""
        return f"""
Аналіз теми '{topic}':

**Основні тренди:**
• Персоналізація навчання через AI
• Автоматизація рутинних задач
• Адаптивні навчальні системи

**Переваги:**
• Підвищення ефективності навчання на 30%
• Доступність 24/7
• Індивідуальний підхід

**Виклики:**
• Необхідність підготовки викладачів
• Питання етики та приватності
• Цифрова нерівність

**Прогноз:**
Очікується зростання ринку EdTech на 45% до 2026 року.
"""
    
    def _create_report(self, results: dict) -> str:
        """Створення фінального звіту"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              LANGCHAIN 1.0 RESEARCH REPORT                   ║
╚══════════════════════════════════════════════════════════════╝

Дата: {results['timestamp']}
Тема: {results['topic']}
Платформа: LangChain 1.0 + OpenAI GPT-4

════════════════════════════════════════════════════════════════

РЕЗУЛЬТАТИ ПОШУКУ:
{results.get('search', 'Немає даних')}

════════════════════════════════════════════════════════════════

СТАТИСТИЧНИЙ АНАЛІЗ:
{results.get('analysis', 'Немає даних')}

════════════════════════════════════════════════════════════════

AI АНАЛІТИКА:
{results.get('ai_analysis', 'Немає даних')}

════════════════════════════════════════════════════════════════

Дослідження завершено успішно
"""

# ===========================
# ГОЛОВНА ФУНКЦІЯ
# ===========================

def main():
    """Запуск LangChain 1.0 агента"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║            LANGCHAIN 1.0 RESEARCH AGENT                      ║
║                Найновіші версії пакеті                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Перевірка версій
    print("Версії пакетів:")
    try:
        import langchain
        print(f"   LangChain: {langchain.__version__}")
    except:
        print(f"   LangChain: не встановлено")
    
    try:
        import langchain_openai
        print(f"   LangChain-OpenAI: встановлено")
    except:
        print(f"   LangChain-OpenAI: не встановлено")
    
    try:
        import openai
        print(f"   OpenAI: {openai.__version__}")
    except:
        print(f"   OpenAI: не встановлено")
    
    # API ключ
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"   API ключ: {api_key[:10]}...{api_key[-4:]}")
    else:
        print(f"   API ключ: не знайдено (демо режим)")
    
    print("\n" + "=" * 60)
    
    # Створення агента
    agent = LangChain1Agent(api_key)
    
    # Дослідження
    topic = "Штучний інтелект в освіті 2025: найновіші тренди"
    result = agent.research(topic)
    
    # Виведення звіту
    print("\n" + "=" * 60)
    print(result["report"])
    
    print("\nГотово! Перегляньте файли:")
    print("   • langchain1_report.json - повні дані")
    print("   • langchain1_memory.json - збережена історія")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()
