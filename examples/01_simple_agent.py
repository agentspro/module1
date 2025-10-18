"""
Повноцінний простий AI Research Agent для студентів
Реальний пошук, реальний аналіз, реальний AI - але простий код!
"""

import os
import json
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Завантажуємо змінні середовища
load_dotenv()

# Базові імпорти
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class ResearchAgent:
    """Простий але повноцінний AI агент для досліджень"""
    
    def __init__(self):
        """Ініціалізація агента"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            # Використовуємо GPT-4 для кращих результатів
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                api_key=api_key
            )
            print("✅ AI підключено (GPT-4)")
        else:
            # Можна використати GPT-3.5 для економії
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            print("⚠️ Використовую GPT-3.5 (додайте API ключ для GPT-4)")
    
    def search_web(self, query: str) -> List[Dict]:
        """Реальний пошук в інтернеті через DuckDuckGo"""
        try:
            # Спробуємо нову версію
            from duckduckgo_search import DDGS
            
            print(f"  🔍 Шукаю: {query}...")
            results = []
            
            with DDGS() as ddgs:
                # Реальний пошук - отримуємо 5 результатів
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        "title": r.get('title', ''),
                        "body": r.get('body', ''),
                        "link": r.get('href', '')
                    })
            
            print(f"  ✓ Знайдено {len(results)} результатів")
            return results
            
        except ImportError:
            print("  ⚠️ Встановіть: pip install duckduckgo-search")
            # Демо результати якщо пошук не працює
            return [
                {
                    "title": "AI трансформує освіту",
                    "body": "Штучний інтелект революціонізує навчальний процес...",
                    "link": "https://example.com/1"
                },
                {
                    "title": "85% університетів використовують AI",
                    "body": "За даними дослідження, більшість закладів...",
                    "link": "https://example.com/2"
                }
            ]
    
    def analyze_text(self, text: str) -> Dict:
        """Аналіз тексту: статистика та тональність"""
        # Базова статистика
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Аналіз тональності (українська + англійська)
        positive_words = [
            "успіх", "покращення", "інновація", "прогрес", "розвиток",
            "success", "improvement", "innovation", "progress", "growth"
        ]
        negative_words = [
            "проблема", "виклик", "ризик", "загроза", "складність",
            "problem", "challenge", "risk", "threat", "difficulty"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Визначаємо тональність
        if positive_count > negative_count:
            sentiment = "позитивна 😊"
        elif negative_count > positive_count:
            sentiment = "негативна 😟"
        else:
            sentiment = "нейтральна 😐"
        
        return {
            "слів": word_count,
            "символів": char_count,
            "тональність": sentiment,
            "позитивних_слів": positive_count,
            "негативних_слів": negative_count
        }
    
    def summarize_with_ai(self, topic: str, search_results: List[Dict]) -> str:
        """Використовуємо AI для створення аналітичного висновку"""
        # Готуємо дані для AI
        search_text = "\n".join([
            f"- {r['title']}: {r['body'][:200]}..."
            for r in search_results
        ])
        
        # Створюємо промпт
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ти - експерт аналітик. 
            Проаналізуй надані результати пошуку та створи структурований висновок.
            Відповідай українською мовою."""),
            ("human", f"""
            Тема дослідження: {topic}
            
            Результати пошуку:
            {search_text}
            
            Створи короткий аналітичний висновок з:
            1. Головними трендами
            2. Ключовими фактами
            3. Рекомендаціями
            """)
        ])
        
        # Отримуємо відповідь від AI
        try:
            chain = prompt | self.llm
            response = chain.invoke({})
            return response.content
        except Exception as e:
            print(f"  ⚠️ Помилка AI: {e}")
            return "AI аналіз тимчасово недоступний"
    
    def save_report(self, report: Dict) -> str:
        """Зберігаємо звіт у файл"""
        filename = f"research_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def research(self, topic: str) -> Dict:
        """Головна функція: повне дослідження теми"""
        print(f"\n{'='*60}")
        print(f"🔬 ДОСЛІДЖЕННЯ: {topic}")
        print(f"{'='*60}")
        
        # Крок 1: Пошук інформації
        print("\n1️⃣ ПОШУК ІНФОРМАЦІЇ")
        search_results = self.search_web(topic)
        
        # Крок 2: Аналіз даних
        print("\n2️⃣ АНАЛІЗ ДАНИХ")
        all_text = " ".join([r['title'] + " " + r['body'] for r in search_results])
        analysis = self.analyze_text(all_text)
        print(f"  📊 Проаналізовано: {analysis['слів']} слів")
        print(f"  😊 Тональність: {analysis['тональність']}")
        
        # Крок 3: AI висновки
        print("\n3️⃣ AI АНАЛІТИКА")
        ai_summary = self.summarize_with_ai(topic, search_results)
        print(f"  🤖 Згенеровано аналітичний висновок")
        
        # Крок 4: Формуємо повний звіт
        report = {
            "тема": topic,
            "дата": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "результати_пошуку": search_results,
            "аналіз": analysis,
            "ai_висновок": ai_summary,
            "джерела": [r['link'] for r in search_results if r.get('link')]
        }
        
        # Крок 5: Зберігаємо звіт
        print("\n4️⃣ ЗБЕРЕЖЕННЯ ЗВІТУ")
        filename = self.save_report(report)
        print(f"  💾 Звіт збережено: {filename}")
        
        return report

def format_report(report: Dict) -> str:
    """Красиве форматування звіту для виводу"""
    return f"""
╔══════════════════════════════════════════════════════════════╗
║                    📋 ЗВІТ ДОСЛІДЖЕННЯ                       ║
╚══════════════════════════════════════════════════════════════╝

📅 Дата: {report['дата']}
🎯 Тема: {report['тема']}

────────────────────────────────────────────────────────────────
📊 АНАЛІЗ ДАНИХ:
• Оброблено слів: {report['аналіз']['слів']}
• Тональність: {report['аналіз']['тональність']}
• Позитивних маркерів: {report['аналіз']['позитивних_слів']}
• Негативних маркерів: {report['аналіз']['негативних_слів']}

────────────────────────────────────────────────────────────────
🔍 ЗНАЙДЕНІ ДЖЕРЕЛА ({len(report['результати_пошуку'])}):
{chr(10).join([f"• {r['title']}" for r in report['результати_пошуку'][:3]])}

────────────────────────────────────────────────────────────────
🤖 AI ВИСНОВОК:
{report['ai_висновок'][:500]}...

────────────────────────────────────────────────────────────────
✅ Дослідження завершено успішно!
"""

def main():
    """Запуск програми"""
    # Заголовок
    print("\n" + "="*60)
    print(" "*15 + "🤖 AI RESEARCH AGENT v2.0")
    print(" "*18 + "Simple but Powerful")
    print("="*60)
    
    # Перевірка середовища
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API ключ знайдено: {api_key[:10]}...")
    else:
        print("⚠️ API ключ не знайдено (додайте в .env для кращих результатів)")
    
    # Створюємо агента
    agent = ResearchAgent()
    
    # Тема для дослідження
    topic = "Штучний інтелект в освіті України 2025"
    
    # Проводимо дослідження
    report = agent.research(topic)
    
    # Виводимо красивий звіт
    print(format_report(report))
    
    # Додаткова інформація
    print(f"\n💡 Підказка: Перегляньте повний звіт у файлі JSON")
    print(f"📁 Файли звітів зберігаються з префіксом 'research_'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Програму перервано користувачем")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        print("💡 Перевірте чи встановлені всі пакети:")
        print("   pip install langchain langchain-openai duckduckgo-search python-dotenv")
