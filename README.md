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

## 🔧 Як працюють агенти

Всі агенти реалізують один патерн дослідження з чотирма кроками:

1. **Web Search** - пошук інформації через DuckDuckGo (або demo fallback)
2. **Data Analysis** - аналіз тональності та статистика тексту
3. **AI Processing** - генерація звіту через LLM (або demo аналіз)
4. **Memory/Storage** - збереження результатів у JSON файли

### Архітектура LangChain 1.0

**Ключові концепції**: LCEL (LangChain Expression Language), chains, промпти

```python
# Структура
class LangChain1Agent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.tools = {
            "search_web": function,
            "analyze_data": function,
            "save_to_memory": function
        }
        self.chains = {
            "research": prompt | llm | output_parser,
            "conclusion": prompt | llm | output_parser
        }

    def research(self, topic: str):
        # 1. Виконати інструменти
        search_results = self.tools["search_web"](topic)
        analysis = self.tools["analyze_data"](search_results)

        # 2. Запустити AI через chains
        ai_analysis = self.chains["research"].invoke({
            "topic": topic,
            "data": search_results
        })

        # 3. Зберегти в пам'ять
        self.tools["save_to_memory"](results)
```

**Виходи**: `langchain1_report.json`, `langchain1_memory.json`

**Pipeline**: Tools → LCEL Chains → JSON Output

---

### Архітектура CrewAI

**Ключові концепції**: Agents з ролями, Tasks з контекстом, Crew з Process

#### Простий агент (02_crewai_simple.py)

```python
# Інструменти через декоратор
@tool("Web Search")
def search_web(query: str) -> str:
    """Пошук інформації"""
    return results

# Агент з роллю
agent = Agent(
    role='Дослідник AI',
    goal='Зібрати інформацію',
    backstory='Експерт з AI',
    tools=[search_web, analyze_data],
    verbose=True,
    max_iter=5
)

# Задача
task = Task(
    description="Дослідіть тему: {topic}",
    expected_output="Структурований звіт",
    agent=agent
)

# Запуск
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

#### Мультиагентна система (02_crewai_agent.py)

```python
# Три агенти з різними ролями
researcher = Agent(role='Дослідник', tools=[search_web])
analyst = Agent(role='Аналітик', tools=[analyze_data])
reporter = Agent(role='Репортер', tools=[generate_report])

# Задачі з залежностями через context
task1 = Task(description="Пошук", agent=researcher)
task2 = Task(description="Аналіз", agent=analyst, context=[task1])
task3 = Task(description="Звіт", agent=reporter, context=[task1, task2])

# Послідовне виконання
crew = Crew(
    agents=[researcher, analyst, reporter],
    tasks=[task1, task2, task3],
    process=Process.sequential,  # або Process.hierarchical
    memory=True,
    cache=True
)
```

**Виходи**: `crewai_report_*.json`, `crewai_final_*.json`, `*.md` звіти

**Pipeline**: Agents → Tasks (sequential) → Context Sharing → JSON Output

---

### Архітектура SmolAgents

**Ключові концепції**: CodeAgent генерує Python код, мінімалістичний підхід

```python
# Інструменти з docstrings
@tool
def search_web(query: str) -> str:
    """
    Пошук інформації в інтернеті.

    Args:
        query: Пошуковий запит

    Returns:
        Результати пошуку
    """
    return results

# Агент з моделлю
agent = CodeAgent(
    tools=[search_web, analyze_sentiment, save_memory],
    model=OpenAIServerModel(model_id="gpt-4"),
    max_steps=5,
    verbose=True
)

# Задача як промпт
task = """
Проведіть дослідження:
1. Знайдіть інформацію
2. Проаналізуйте тональність
3. Збережіть в пам'ять
"""

# Агент генерує Python код для виконання
result = agent.run(task)
```

**Виходи**: `smolagents_memory.json`

**Pipeline**: Task Description → Code Generation → Tool Execution → Results

---

## 📊 Порівняння фреймворків

| Характеристика | LangChain | CrewAI | SmolAgents |
|----------------|-----------|---------|------------|
| **Підхід** | Chains & Pipelines | Multi-Agent Teams | Code Generation |
| **Складність** | Середня | Висока | Низька |
| **Інструменти** | Функції в dict | @tool декоратор | @tool з docstrings |
| **Паралелізм** | Через LCEL | Process.sequential/hierarchical | Послідовно |
| **Контекст** | Через chains | context між tasks | Через generated code |
| **LLM підтримка** | OpenAI, Anthropic | OpenAI | OpenAI, HF, Local |
| **Use case** | Pipelines, RAG | Командна робота | Швидкі прототипи |

---

## 📝 Приклади використання

### LangChain Agent
```bash
python3 examples/01_langchain_v1.py
```
Демонструє LCEL chains та pipeline pattern для дослідження.

### CrewAI Simple Agent
```bash
python3 examples/02_crewai_simple.py
```
Один агент з кількома інструментами та Task.

### CrewAI Multi-Agent
```bash
python3 examples/02_crewai_agent.py
```
Команда з 3 агентів (researcher → analyst → reporter) з context sharing.

### SmolAgents
```bash
python3 examples/03_smolagents_agent.py
```
CodeAgent генерує Python код для виконання дослідження.

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


---
**Оновлено**: Жовтень 2025
