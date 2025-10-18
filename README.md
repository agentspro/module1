# Module 1: AI Agents - Порівняння фреймворків

## 🎯 Мета модуля

Навчитись створювати AI-агентів на трьох популярних фреймворках та розуміти їх відмінності.

## ✅ Перевірені версії (працює в GitHub Codespaces)

- **Python**: 3.12.1
- **LangChain**: 1.0.0
- **CrewAI**: 0.203.1
- **SmolAgents**: 1.22.0
- **OpenAI**: 1.109.1

## 📚 Зміст репозиторію

```
module1/
├── examples/                       # Приклади агентів
│   ├── 01_langchain_v1.py         # LangChain 1.0 (РЕКОМЕНДОВАНО)
│   ├── 01_universal_agent.py      # Універсальний агент (без LangChain)
│   ├── 01_langchain_agent.py      # LangChain класичний
│   ├── 02_crewai_agent.py         # CrewAI v0.203
│   ├── 03_smolagents_agent.py     # SmolAgents v1.22
│   ├── langchain_demo.py          # Демо без API
│   ├── crewai_demo.py             # Демо без API
│   └── smolagents_demo.py         # Демо без API
├── test_agents.py                  # Тестування всіх агентів
├── quick_start.sh                  # Швидкий запуск
├── requirements.txt                # Точні версії пакетів
├── .env.template                   # Шаблон для API ключів
└── README.md                       # Цей файл
```

## 🚀 Швидкий старт

### 1. Клонування репозиторію (або відкрийте в GitHub Codespaces)

```bash
git clone https://github.com/agentspro/module1.git
cd module1
```

### 2. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 3. Налаштування API ключів

```bash
cp .env.template .env
# Відредагуйте .env та додайте ваші API ключі
```

### 4. Запуск прикладів

```bash
# РЕКОМЕНДОВАНО - LangChain 1.0 агент
python3 examples/01_langchain_v1.py

# Універсальний агент (працює завжди)
python3 examples/01_universal_agent.py

# CrewAI агент
python3 examples/02_crewai_agent.py

# SmolAgents агент
python3 examples/03_smolagents_agent.py

# Або запустіть всі тести
python3 test_agents.py
```

## 📊 Порівняння фреймворків

| Фреймворк | Версія | Складність | Мультиагентність | Особливості |
|-----------|--------|------------|------------------|-------------|
| **LangChain** | 1.0.0 | Висока | Ручна | Найбільша екосистема, LCEL |
| **CrewAI** | 0.203.1 | Середня | Вбудована | Role-based агенти |
| **SmolAgents** | 1.22.0 | Низька | Ручна | Мінімалістичний, CodeAgent |

## 🎓 Для студентів

### Завдання 1: Базовий агент
1. Запустіть `examples/01_langchain_v1.py`
2. Змініть тему дослідження в коді
3. Проаналізуйте результати в `langchain1_report.json`

### Завдання 2: Порівняння фреймворків
1. Запустіть всі три агенти з однією темою
2. Порівняйте результати та швидкість
3. Визначте переваги кожного

### Завдання 3: Мультиагентна система
1. Використайте CrewAI для створення команди
2. Додайте ролі: Дослідник, Аналітик, Письменник
3. Реалізуйте складне дослідження

## 🛠 Вимоги

- **Python**: 3.10+ (тестовано на 3.12.1)
- **OpenAI API key**: обов'язково для повної функціональності
- **RAM**: мінімум 2GB
- **Інтернет**: для API викликів

## 📝 Структура проекту для навчання

1. **Початківці**: Почніть з `examples/01_universal_agent.py`
2. **Середній рівень**: Вивчіть `examples/01_langchain_v1.py`
3. **Просунуті**: Експериментуйте з CrewAI мультиагентністю

## 🔧 Вирішення проблем

### Помилка імпорту LangChain
```bash
pip install --upgrade langchain langchain-openai langchain-core
```

### API ключ не працює
```bash
# Перевірте .env файл
cat .env
# Має бути: OPENAI_API_KEY=sk-...
```

### Попередження про duckduckgo_search
```bash
pip install ddgs  # Нова версія
```

## 📊 Результати роботи

Кожен агент створює JSON файли з результатами:
- `langchain1_report.json` - звіт LangChain агента
- `crewai_result.json` - результат CrewAI
- `universal_agent_report.json` - універсальний агент

## 🤝 Внесок

Ласкаво просимо до покращень! Створіть Pull Request або Issue.

## 📝 Ліцензія

MIT - вільне використання для навчання

## 🆘 Підтримка

- **Issues**: https://github.com/agentspro/module1/issues
- **Discussions**: https://github.com/agentspro/module1/discussions

---

**Версія**: 2.0.0  
**Останнє оновлення**: Жовтень 2024  
**Автор**: AI Agents Course 2025
