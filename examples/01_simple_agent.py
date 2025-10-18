"""
Simple AI Research Agent для студентів
Версія: навчальна (100 рядків коду)
Мета: зрозуміти базові концепції AI агентів
"""

import os
import json
from datetime import datetime
from typing import Dict, List

# Завантажуємо змінні середовища
from dotenv import load_dotenv
load_dotenv()

# Імпортуємо LangChain компоненти
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class SimpleAgent:
    """Простий AI агент для дослідження"""
    
    def __init__(self):
        """Ініціалізація агента"""
        # Отримуємо API ключ
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Створюємо LLM (мозок агента)
        if api_key:
            self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
            self.has_ai = True
            print("✅ AI підключено")
        else:
            self.llm = None
            self.has_ai = False
            print("⚠️ Працюю без AI (демо режим)")
    
    def search(self, topic: str) -> str:
        """Інструмент 1: Пошук інформації"""
        # В реальному проекті тут був би справжній пошук
        # Для навчання використовуємо приклад
        return f"""
        Знайдено про '{topic}':
        1. AI персоналізує навчання для кожного студента
        2. 85% університетів використовують AI інструменти
        3. Головні виклики: етика, приватність, доступність
        """
    
    def analyze(self, text: str) -> Dict:
        """Інструмент 2: Аналіз тексту"""
        words = len(text.split())
        
        # Простий аналіз настрою
        positive = text.count("успіх") + text.count("покращення")
        negative = text.count("проблема") + text.count("виклик")
        
        sentiment = "позитивний" if positive > negative else "негативний"
        
        return {
            "слів": words,
            "настрій": sentiment,
            "позитив": positive,
            "негатив": negative
        }
    
    def get_ai_insights(self, topic: str, data: str) -> str:
        """Інструмент 3: AI аналіз (якщо доступний)"""
        if not self.has_ai:
            return "AI аналіз недоступний (потрібен API ключ)"
        
        # Створюємо промпт (інструкцію для AI)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ти експерт з аналізу. Дай короткий висновок."),
            ("human", f"Тема: {topic}\nДані: {data}\nЩо можеш сказати?")
        ])
        
        # Запитуємо AI
        chain = prompt | self.llm
        response = chain.invoke({"topic": topic, "data": data})
        
        return response.content
    
    def research(self, topic: str) -> Dict:
        """Головна функція: проводимо дослідження"""
        print(f"\n🔍 Досліджую: {topic}")
        print("-" * 50)
        
        # Крок 1: Пошук
        print("1️⃣ Шукаю інформацію...")
        search_results = self.search(topic)
        
        # Крок 2: Аналіз
        print("2️⃣ Аналізую дані...")
        analysis = self.analyze(search_results)
        
        # Крок 3: AI висновки
        print("3️⃣ Генерую висновки...")
        ai_insights = self.get_ai_insights(topic, search_results)
        
        # Формуємо результат
        result = {
            "тема": topic,
            "дата": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "пошук": search_results,
            "аналіз": analysis,
            "ai_висновки": ai_insights
        }
        
        # Зберігаємо в файл
        with open("agent_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("✅ Готово! Результат в agent_result.json")
        return result

def main():
    """Запуск програми"""
    print("=" * 50)
    print("     🤖 SIMPLE AI RESEARCH AGENT")
    print("=" * 50)
    
    # Створюємо агента
    agent = SimpleAgent()
    
    # Досліджуємо тему
    topic = "Штучний інтелект в освіті 2025"
    result = agent.research(topic)
    
    # Виводимо результат
    print(f"\n📊 РЕЗУЛЬТАТ:")
    print(f"Настрій: {result['аналіз']['настрій']}")
    print(f"Слів проаналізовано: {result['аналіз']['слів']}")
    print(f"\n💡 AI висновок: {result['ai_висновки'][:200]}...")

if __name__ == "__main__":
    main()
