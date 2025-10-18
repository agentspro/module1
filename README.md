# Module 1: AI Agents - Порівняння фреймворків

## 🎯 Мета модуля

Навчитись створювати AI-агентів на трьох популярних фреймворках та розуміти їх відмінності.

## 📚 Зміст репозиторію

```
module1/
├── examples/                     # Приклади агентів
│   ├── 01_langchain_agent.py   # LangChain v1.0
│   ├── 02_crewai_agent.py      # CrewAI v1.0
│   └── 03_smolagents_agent.py  # SmolAgents
├── docs/                        # Документація
│   └── comparison.md            # Детальне порівняння
├── requirements.txt            # Залежності
├── .env.template              # Шаблон для API ключів
└── README.md                  # Цей файл
```

## 🚀 Швидкий старт

### 1. Клонування репозиторію

```bash
git clone https://github.com/agentspro/module1.git
cd module1
```

### 2. Створення віртуального середовища

```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

### 3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 4. Налаштування API ключів

```bash
cp .env.template .env
# Відредагуйте .env та додайте ваші API ключі
```

### 5. Запуск прикладів

```bash
# LangChain агент
python examples/01_langchain_agent.py

# CrewAI агент
python examples/02_crewai_agent.py

# SmolAgents агент
python examples/03_smolagents_agent.py
```

## 📊 Порівняння фреймворків

| Фреймворк | Складність | Мультиагентність | Особливості |
|-----------|------------|------------------|-------------|
| **LangChain** | Висока | Ручна | Найбільша екосистема |
| **CrewAI** | Середня | Вбудована | Role-based агенти |
| **SmolAgents** | Низька | Ручна | Мінімалістичний, CodeAgent |

## 🎓 Для студентів

### Завдання 1: Базовий агент
1. Запустіть всі три приклади
2. Порівняйте результати
3. Додайте новий інструмент до кожного агента

### Завдання 2: Мультиагентна система
1. Розширте одного агента до команди
2. Реалізуйте делегування задач
3. Додайте координацію між агентами

## 🛠 Вимоги

- Python 3.10+
- OpenAI API key (обов'язково)
- HuggingFace token (опціонально)

## 📝 Ліцензія

MIT - вільне використання для навчання

---

**Версія**: 1.0.0  
**Курс**: AI Agents 2025
