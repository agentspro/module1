# Module 1: AI Agents - Навчальний курс

## 🎯 Мета курсу

Навчитись створювати AI-агентів від простого до складного.

## 📚 Структура навчання (три фреймворки)

### 🟢 **Фреймворк 1: LangChain** (~350 рядків)
```bash
python3 examples/01_langchain_v1.py
```
- **Що вивчаємо**: LangChain 1.0, LCEL chains, промпти
- **Особливості**: Ланцюги обробки, pipeline pattern
- **Інструменти**: Web search, data analysis, AI processing
- **Код**: 349 рядків з детальними коментарями

### 🟡 **Фреймворк 2: CrewAI** (~300 рядків)
```bash
# Простий приклад
python3 examples/02_crewai_simple.py

# Мультиагентна система
python3 examples/02_crewai_agent.py
```
- **Що вивчаємо**: Мультиагентні системи, командна робота
- **Особливості**: Агенти з ролями, задачі з контекстом
- **Інструменти**: Custom tools через @tool декоратор
- **Код**: 323/294 рядки, два приклади

### 🔴 **Фреймворк 3: SmolAgents** (~270 рядків)
```bash
python3 examples/03_smolagents_agent.py
```
- **Що вивчаємо**: Code-first підхід, CodeAgent
- **Особливості**: Генерація Python коду для задач
- **Інструменти**: Підтримка OpenAI, HuggingFace, локальних моделей
- **Код**: 267 рядків, мінімалістичний стиль

## 🚀 Швидкий старт для студентів

### Крок 1: Встановіть залежності
```bash
# Встановити всі залежності
pip install -r requirements.txt

# Або встановити вручну для конкретного фреймворка
pip install langchain==1.0.0 langchain-openai openai python-dotenv  # LangChain
pip install crewai==0.203.1 crewai-tools ddgs                       # CrewAI
pip install smolagents==1.22.0                                       # SmolAgents
```

### Крок 2: Створіть .env файл (опціонально)
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```
**Примітка**: Всі агенти працюють в демо режимі без API ключа!

### Крок 3: Швидкий старт
```bash
# Автоматичне тестування всіх агентів
bash quick_start.sh

# Або запустіть тестовий скрипт
python3 test_agents.py
```

## 📊 Що таке AI Agent?

**AI Agent** - це програма, яка:
1. **Думає** - використовує AI для аналізу
2. **Діє** - виконує дії через інструменти
3. **Вчиться** - покращується з досвідом

### Структура агента:
```python
class SimpleAgent:
    def __init__(self):
        # Мозок агента (LLM)
        self.llm = ChatOpenAI()
    
    def search(self):
        # Інструмент 1: пошук
        
    def analyze(self):
        # Інструмент 2: аналіз
        
    def research(self):
        # Головна логіка
```

## 📝 Приклади використання

### LangChain Agent
```python
# 01_langchain_v1.py - 349 рядків
from examples.langchain_v1 import LangChain1Agent

agent = LangChain1Agent()
result = agent.research("AI в освіті")
# Використовує LCEL chains, промпти, tools
# Виходи: langchain1_report.json, langchain1_memory.json
```

### CrewAI Agent (простий)
```python
# 02_crewai_simple.py - 323 рядки
from examples.crewai_simple import SimpleCrewAIAgent

agent = SimpleCrewAIAgent()
result = agent.research("AI асистенти для студентів")
# Один агент з декількома інструментами
```

### CrewAI Multi-Agent System
```python
# 02_crewai_agent.py - 294 рядки
from examples.crewai_agent import create_research_team, create_research_tasks

researcher, analyst, reporter = create_research_team()
# Команда з 3 агентів: дослідник, аналітик, репортер
# Послідовне виконання задач з контекстом
```

### SmolAgents
```python
# 03_smolagents_agent.py - 267 рядків
from examples.smolagents_agent import SmolAgentsResearchAgent

agent = SmolAgentsResearchAgent(model_type="openai")
result = agent.research("Штучний інтелект в освіті 2025")
# CodeAgent генерує Python код для вирішення задач
```

## 🎓 Навчальний план

### Тиждень 1: LangChain Basics
1. Запустіть `01_langchain_v1.py` і зрозумійте структуру
2. Вивчіть як працюють LCEL chains (research_chain, conclusion_chain)
3. Додайте новий інструмент до `_create_tools()`
4. Змініть промпти в `_create_chains()`

### Тиждень 2: CrewAI Simple Agent
1. Запустіть `02_crewai_simple.py`
2. Зрозумійте структуру Agent (role, goal, backstory)
3. Створіть власний @tool декоратор
4. Експериментуйте з Task description

### Тиждень 3: CrewAI Multi-Agent
1. Запустіть `02_crewai_agent.py`
2. Вивчіть як агенти співпрацюють через context
3. Додайте 4-го агента до команди
4. Спробуйте Process.hierarchical замість sequential

### Тиждень 4: SmolAgents
1. Запустіть `03_smolagents_agent.py`
2. Зрозумійте як CodeAgent генерує код
3. Додайте новий tool з proper docstring
4. Спробуйте локальну модель (model_type="local")

## 🛠 Версії та сумісність

| Пакет | Версія | Обов'язково |
|-------|--------|-------------|
| Python | 3.10+ | ✅ |
| LangChain | 1.0.0 | ✅ |
| OpenAI | 1.109+ | ✅ |
| CrewAI | 0.203+ | ⚪ |
| SmolAgents | 1.22+ | ⚪ |

## ❓ Часті питання

### Який файл запускати першим?
Спочатку запустіть `test_agents.py` або `bash quick_start.sh` для автоматичного тестування всіх агентів. Потім вивчайте файли в порядку: `01_langchain_v1.py` → `02_crewai_simple.py` → `02_crewai_agent.py` → `03_smolagents_agent.py`

### Чи потрібен API ключ?
Ні, агенти працюють в демо режимі без ключа

### Скільки коштує API?
GPT-4: ~$0.03 за запит
GPT-3.5: ~$0.002 за запит

### Де взяти API ключ?
https://platform.openai.com/api-keys

## 📚 Додаткові ресурси

- [LangChain Docs](https://python.langchain.com/) - Офіційна документація LangChain 1.0
- [OpenAI API](https://platform.openai.com/docs) - Документація OpenAI API
- [CrewAI Docs](https://docs.crewai.com/) - Документація CrewAI framework
- [SmolAgents GitHub](https://github.com/huggingface/smolagents) - SmolAgents від HuggingFace

## 🆘 Підтримка

Якщо виникли проблеми:
1. Перевірте версії: `pip list | grep langchain`
2. Перевірте API ключ: `echo $OPENAI_API_KEY`
3. Створіть Issue на GitHub

## 📈 Прогрес навчання

- [ ] Запустив `test_agents.py` та побачив як працюють всі агенти
- [ ] Вивчив LangChain agent та зрозумів LCEL chains
- [ ] Запустив CrewAI simple та розібрав @tool декоратор
- [ ] Запустив CrewAI multi-agent та зрозумів як агенти співпрацюють
- [ ] Запустив SmolAgents та побачив CodeAgent в дії
- [ ] Додав власний інструмент до одного з фреймворків
- [ ] Створив власного агента на основі прикладів

---

**Версія курсу**: 2.2.0
**Оновлено**: Жовтень 2024
**Автор**: AI Agents Course

💡 **Підказка**: Вивчайте фреймворки послідовно та порівнюйте їх підходи до вирішення однієї задачі!
