#!/bin/bash
# Швидкий запуск для GitHub Codespaces

echo "🚀 Module 1: AI Agents - Quick Start"
echo "===================================="
echo ""

# Встановлення базових пакетів (якщо потрібно)
echo "📦 Встановлення залежностей..."
pip install -q python-dotenv 2>/dev/null

# Запуск тестів без API
echo ""
echo "🔬 Запуск демо режиму (без API ключів)..."
echo ""

python3 test_agents.py

echo ""
echo "===================================="
echo "✅ Готово!"
echo ""
echo "Для повного тестування з API:"
echo "1. Створіть файл .env"
echo "2. Додайте OPENAI_API_KEY=your-key"
echo "3. Запустіть: python3 examples/01_langchain_agent.py"
