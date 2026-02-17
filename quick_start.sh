#!/bin/bash
# Швидкий запуск для GitHub Codespaces

echo "🚀 Module 1: AI Agents - Quick Start"
echo "===================================="
echo ""

# Перевірка та завантаження .env
if [ -f .env ]; then
    echo "📁 Знайдено .env файл, завантажую змінні..."
    set -a
    source .env
    set +a
    echo "✅ Змінні середовища завантажено"
    
    # Перевірка API ключа
    if [ ! -z "$OPENAI_API_KEY" ]; then
        echo "✅ OPENAI_API_KEY знайдено: ${OPENAI_API_KEY:0:7}...${OPENAI_API_KEY: -4}"
    fi
else
    echo "📝 .env файл не знайдено"
    echo "   Створіть його командою: cp .env.template .env"
fi

# Встановлення базових пакетів (якщо потрібно)
echo ""
echo "📦 Встановлення залежностей..."
pip install -q python-dotenv 2>/dev/null

# Запуск тестів
echo ""
echo "🔬 Запуск тестування..."
echo ""

# Запускаємо Python скрипт який сам визначить режим
#python3 test_agents.py

echo ""
echo "===================================="
echo "✅ Готово!"
echo ""

# Підказки для користувача
if [ -z "$OPENAI_API_KEY" ]; then
    echo "💡 Для повного тестування з API:"
    echo "1. Створіть файл .env: cp .env.template .env"
    echo "2. Відредагуйте .env та додайте ваш OpenAI ключ"
    echo "3. Запустіть знову: bash quick_start.sh"
else
    echo "💡 Наступні кроки:"
    echo "1. Запустіть окремі агенти:"
    echo "   python3 examples/01_langchain_v1.py"
    echo "   python3 examples/02_langchain_langgraph.py"
    echo "   python3 examples/03_crewai_simple.py"
    echo "   python3 examples/04_crewai_agents.py"
    echo "   python3 examples/05_smolagents_agent.py"
    echo "   python3 examples/06_smolagents_multiagent.py"
    echo ""
    echo "2. Перегляньте результати:"
    echo "   ls -la *.json"
fi
