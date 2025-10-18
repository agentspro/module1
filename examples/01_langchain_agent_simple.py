"""
Модуль 1: AI Research Agent на LangChain
Спрощена версія, яка працює з будь-якою версією LangChain
"""

import os
import sys
from datetime import datetime
import json

# Перевірка версії та імпорт
print("🔍 Перевірка встановлених пакетів...")

try:
    import langchain
    print(f"✅ LangChain версія: {langchain.__version__}")
except ImportError:
    print("❌ LangChain не встановлено")
    sys.exit(1)

try:
    from langchain_openai import ChatOpenAI
    print("✅ langchain-openai знайдено")
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        print("✅ Використовую langchain.chat_models")
    except ImportError:
        print("❌ Не можу імпортувати ChatOpenAI")
        ChatOpenAI = None

# Завантаження .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env файл завантажено")
except:
    print("⚠️ python-dotenv не встановлено")

# ===========================
# ПРОСТИЙ АГЕНТ-ДОСЛІДНИК
# ===========================

class SimpleResearchAgent:
    """
    Простий агент для дослідження - працює з будь-якою версією LangChain
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.has_api = bool(self.api_key)
        
        if self.has_api and ChatOpenAI:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.7,
                    api_key=self.api_key
                )
                print(f"✅ LLM створено з API ключем")
            except Exception as e:
                print(f"⚠️ Не вдалось створити LLM: {e}")
                self.llm = None
                self.has_api = False
        else:
            self.llm = None
            print("⚠️ Працюю в демо режимі (без API)")
    
    def search_web(self, query: str) -> str:
        """Пошук інформації (демо)"""
        return f"""
🔍 Результати пошуку для '{query}':

1. **AI трансформує освіту** - Штучний інтелект революціонізує освітні процеси через персоналізацію.

2. **Статистика 2025** - 85% навчальних закладів використовують AI-інструменти.

3. **Основні тренди**:
   - Адаптивне навчання підлаштовується під кожного студента
   - AI-тьютори доступні 24/7
   - Автоматична перевірка завдань економить час викладачів
   
4. **Виклики**: Необхідність навчання викладачів новим технологіям.

5. **Майбутнє**: Прогнозується зростання ринку EdTech на 45% до 2026 року.
"""
    
    def analyze_sentiment(self, text: str) -> str:
        """Аналіз тональності"""
        positive_words = ["революціонізує", "покращення", "зростання", "доступні", "економить"]
        negative_words = ["виклики", "проблеми", "складність"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "📊 Тональність: Позитивна (оптимістичний погляд на AI в освіті)"
        elif neg_count > pos_count:
            return "📊 Тональність: Негативна (песимістичний погляд)"
        else:
            return "📊 Тональність: Нейтральна (збалансований погляд)"
    
    def create_report(self, topic: str, search_results: str, sentiment: str) -> str:
        """Створення звіту"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║              ЗВІТ ДОСЛІДЖЕННЯ AI-АГЕНТА                   ║
╚══════════════════════════════════════════════════════════╝

📅 **Дата**: {timestamp}
🎯 **Тема**: {topic}
🤖 **Агент**: LangChain Research Agent
📊 **Режим**: {"API" if self.has_api else "Демо"}

═══════════════════════════════════════════════════════════

📌 **РЕЗУЛЬТАТИ ПОШУКУ**
{search_results}

═══════════════════════════════════════════════════════════

📈 **АНАЛІЗ**
{sentiment}

═══════════════════════════════════════════════════════════

💡 **ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ**

На основі проведеного дослідження можна зробити наступні висновки:

1. **Позитивні аспекти**:
   - AI значно покращує персоналізацію навчання
   - Автоматизація рутинних задач вивільняє час для творчості
   - Доступність навчання зростає завдяки AI-асистентам

2. **Виклики для вирішення**:
   - Необхідність підвищення цифрової грамотності викладачів
   - Забезпечення етичного використання AI
   - Збереження людського фактору в освіті

3. **Рекомендації**:
   - Поступове впровадження AI-інструментів
   - Інвестиції в навчання персоналу
   - Розробка чітких етичних стандартів

═══════════════════════════════════════════════════════════

✅ **СТАТУС**: Дослідження успішно завершено
"""
        return report
    
    def research(self, topic: str) -> dict:
        """Виконати дослідження"""
        print(f"\n🚀 Починаю дослідження: {topic}")
        print("=" * 60)
        
        # Крок 1: Пошук
        print("📍 Крок 1/3: Пошук інформації...")
        search_results = self.search_web(topic)
        print("   ✓ Завершено")
        
        # Крок 2: Аналіз
        print("📍 Крок 2/3: Аналіз тональності...")
        sentiment = self.analyze_sentiment(search_results)
        print("   ✓ Завершено")
        
        # Крок 3: Звіт
        print("📍 Крок 3/3: Формування звіту...")
        report = self.create_report(topic, search_results, sentiment)
        print("   ✓ Завершено")
        
        # Збереження результату
        result = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "report": report,
            "mode": "api" if self.has_api else "demo"
        }
        
        # Зберігаємо в файл
        with open("langchain_research_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("\n💾 Результат збережено: langchain_research_result.json")
        
        return result

# ===========================
# ГОЛОВНА ФУНКЦІЯ
# ===========================

def main():
    """Головна функція для запуску агента"""
    
    print("""
╔══════════════════════════════════════════════════════════╗
║            LANGCHAIN RESEARCH AGENT v2.0                  ║
║                  Спрощена версія                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Перевірка API ключа
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API ключ знайдено: {api_key[:10]}...")
    else:
        print("⚠️ API ключ не знайдено - працюю в демо режимі")
        print("💡 Підказка: створіть .env файл з OPENAI_API_KEY=...")
    
    # Створення агента
    print("\n🤖 Ініціалізація агента...")
    agent = SimpleResearchAgent(api_key)
    
    # Тема дослідження
    topic = "Штучний інтелект в освіті 2025: тренди та виклики"
    
    # Виконання дослідження
    print("\n" + "=" * 60)
    result = agent.research(topic)
    
    # Виведення результату
    print("\n" + "=" * 60)
    print(result["report"])
    
    print("\n✅ Програма завершена успішно!")
    print("📂 Перегляньте файл: langchain_research_result.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Програму перервано користувачем")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
