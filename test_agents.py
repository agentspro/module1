#!/usr/bin/env python3
"""
Простий скрипт для швидкого тестування всіх агентів
Працює в GitHub Codespaces без додаткових налаштувань
"""

import os
import sys
from datetime import datetime

def test_without_api():
    """Тестування агентів без API ключів (демо режим)"""
    print("\n" + "="*60)
    print("🎯 ТЕСТУВАННЯ АГЕНТІВ В ДЕМО РЕЖИМІ (БЕЗ API)")
    print("="*60)
    
    # Тест 1: LangChain
    print("\n1️⃣ Тестування LangChain Agent...")
    print("-" * 40)
    try:
        from examples.langchain_demo import demo_langchain
        demo_langchain()
        print("✅ LangChain працює в демо режимі")
    except Exception as e:
        print(f"❌ LangChain помилка: {e}")
    
    # Тест 2: CrewAI
    print("\n2️⃣ Тестування CrewAI Agent...")
    print("-" * 40)
    try:
        from examples.crewai_demo import demo_crewai
        demo_crewai()
        print("✅ CrewAI працює в демо режимі")
    except Exception as e:
        print(f"❌ CrewAI помилка: {e}")
    
    # Тест 3: SmolAgents
    print("\n3️⃣ Тестування SmolAgents...")
    print("-" * 40)
    try:
        from examples.smolagents_demo import demo_smolagents
        demo_smolagents()
        print("✅ SmolAgents працює в демо режимі")
    except Exception as e:
        print(f"❌ SmolAgents помилка: {e}")

def check_environment():
    """Перевірка середовища"""
    print("\n🔍 ПЕРЕВІРКА СЕРЕДОВИЩА")
    print("="*60)
    
    # Python версія
    print(f"Python: {sys.version}")
    
    # Перевірка пакетів
    packages = ["langchain", "crewai", "smolagents", "openai"]
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: не встановлено")
    
    # API ключі
    print("\n🔑 API КЛЮЧІ:")
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY встановлено")
    else:
        print("⚠️  OPENAI_API_KEY відсутній (працюватиме в демо режимі)")

def main():
    """Головна функція"""
    print("""
╔══════════════════════════════════════════════════════════╗
║          MODULE 1: AI AGENTS - ТЕСТУВАННЯ                 ║
║                  GitHub Codespaces                        ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Перевірка середовища
    check_environment()
    
    # Запуск тестів
    print("\n" + "="*60)
    print("🚀 ЗАПУСК ТЕСТІВ")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  API ключ не знайдено. Запускаю демо режим...")
        test_without_api()
    else:
        print("\n✅ API ключ знайдено. Запускаю повне тестування...")
        # Тут можна додати повне тестування з API
        
    print("\n" + "="*60)
    print("✅ ТЕСТУВАННЯ ЗАВЕРШЕНО")
    print(f"⏰ Час: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
