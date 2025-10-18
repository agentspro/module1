#!/usr/bin/env python3
"""
Простий скрипт для швидкого тестування всіх агентів
Працює в GitHub Codespaces без додаткових налаштувань
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Завантаження .env файлу
def load_env():
    """Завантаження змінних з .env файлу"""
    env_file = Path('.env')
    if env_file.exists():
        print("📁 Знайдено .env файл, завантажую...")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ .env файл завантажено")
        except ImportError:
            print("⚠️  python-dotenv не встановлено, читаю .env вручну...")
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        if key == 'OPENAI_API_KEY':
                            print(f"✅ Завантажено {key[:15]}...")
    else:
        print("📝 .env файл не знайдено")

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

def test_with_api():
    """Тестування агентів з API ключами"""
    print("\n" + "="*60)
    print("🚀 ТЕСТУВАННЯ АГЕНТІВ З API")
    print("="*60)
    
    # Тест 1: LangChain з API
    print("\n1️⃣ Тестування LangChain Agent з API...")
    print("-" * 40)
    try:
        # Імпортуємо та запускаємо
        import sys
        sys.path.insert(0, 'examples')
        from examples.langchain_agent import main as langchain_main
        langchain_main()
        print("✅ LangChain працює з API")
    except Exception as e:
        print(f"⚠️  LangChain: {e}")
        print("Використовую демо режим...")
        from examples.langchain_demo import demo_langchain
        demo_langchain()

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
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Маскуємо ключ
        masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
        print(f"✅ OPENAI_API_KEY встановлено: {masked_key}")
    else:
        print("⚠️  OPENAI_API_KEY відсутній (працюватиме в демо режимі)")
    
    # Інші ключі
    if os.getenv("HF_TOKEN"):
        print("✅ HF_TOKEN встановлено")
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY встановлено")

def main():
    """Головна функція"""
    print("""
╔══════════════════════════════════════════════════════════╗
║          MODULE 1: AI AGENTS - ТЕСТУВАННЯ                 ║
║                  GitHub Codespaces                        ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Завантаження .env
    load_env()
    
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
        choice = input("\nВиберіть режим:\n1. Демо режим (швидко)\n2. Повний тест з API (повільно)\n\nВаш вибір (1 або 2): ").strip()
        
        if choice == "2":
            test_with_api()
        else:
            test_without_api()
        
    print("\n" + "="*60)
    print("✅ ТЕСТУВАННЯ ЗАВЕРШЕНО")
    print(f"⏰ Час: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Показати створені файли
    print("\n📁 Створені файли:")
    for file in ["langchain_demo_result.json", "crewai_demo_result.json", "smolagents_demo_result.json", "langchain_result.json", "agent_memory.json"]:
        if Path(file).exists():
            print(f"  ✓ {file}")

if __name__ == "__main__":
    main()
