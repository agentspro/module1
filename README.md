# Module 1: AI Agents - Навчальний курс

## 🎯 Мета курсу

Навчитись створювати AI-агентів від простого до складного.

## 📚 Структура навчання (від простого до складного)

### 🟢 **Рівень 1: Початківець** (100 рядків)
```bash
python3 examples/01_simple_agent.py
```
- **Для кого**: Початківці без досвіду
- **Що вивчаємо**: Базові концепції AI агентів
- **Код**: ~100 рядків, прості коментарі

### 🟡 **Рівень 2: Середній** (300 рядків)
```bash
python3 examples/01_langchain_v1.py
```
- **Для кого**: Ті, хто знає Python
- **Що вивчаємо**: LangChain, chains, промпти
- **Код**: ~300 рядків, детальніше

### 🔴 **Рівень 3: Просунутий** (700 рядків)
```bash
python3 examples/01_langchain_senior.py
```
- **Для кого**: Senior developers
- **Що вивчаємо**: SOLID, async/await, patterns
- **Код**: ~700 рядків, production-ready

## 🚀 Швидкий старт для студентів

### Крок 1: Встановіть залежності
```bash
pip install langchain langchain-openai openai python-dotenv
```

### Крок 2: Створіть .env файл
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Крок 3: Запустіть простий приклад
```bash
python3 examples/01_simple_agent.py
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

## 📝 Приклади для різних рівнів

### Початківці: Простий агент
```python
# 01_simple_agent.py - 100 рядків
agent = SimpleAgent()
result = agent.research("AI в освіті")
print(result)
```

### Середній рівень: LangChain
```python
# 01_langchain_v1.py - 300 рядків
agent = LangChain1Agent()
result = agent.research("AI в освіті")
# Використовує chains, промпти
```

### Просунутий: Enterprise
```python
# 01_langchain_senior.py - 700 рядків
config = AgentConfig(model=ModelType.GPT4)
app = Application(config)
await app.run("AI в освіті")
# Async, types, error handling
```

## 🎓 Навчальний план

### Тиждень 1: Основи
1. Запустіть `01_simple_agent.py`
2. Зрозумійте 3 основні методи
3. Змініть тему дослідження

### Тиждень 2: LangChain
1. Вивчіть `01_langchain_v1.py`
2. Додайте новий інструмент
3. Змініть промпти

### Тиждень 3: Мультиагентність
1. Запустіть `02_crewai_agent.py`
2. Створіть команду агентів
3. Додайте ролі

### Тиждень 4: Production
1. Аналізуйте `01_langchain_senior.py`
2. Вивчіть error handling
3. Додайте тести

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
Почніть з `01_simple_agent.py` - він найпростіший

### Чи потрібен API ключ?
Ні, агенти працюють в демо режимі без ключа

### Скільки коштує API?
GPT-4: ~$0.03 за запит
GPT-3.5: ~$0.002 за запит

### Де взяти API ключ?
https://platform.openai.com/api-keys

## 📚 Додаткові ресурси

- [LangChain Docs](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [CrewAI Docs](https://docs.crewai.com/)

## 🆘 Підтримка

Якщо виникли проблеми:
1. Перевірте версії: `pip list | grep langchain`
2. Перевірте API ключ: `echo $OPENAI_API_KEY`
3. Створіть Issue на GitHub

## 📈 Прогрес навчання

- [ ] Запустив простий агент
- [ ] Зрозумів 3 основні методи
- [ ] Змінив тему дослідження
- [ ] Додав новий інструмент
- [ ] Створив мультиагентну систему
- [ ] Написав власного агента

---

**Версія курсу**: 2.1.0  
**Оновлено**: Жовтень 2024  
**Автор**: AI Agents Course

💡 **Підказка**: Починайте з простого! Не намагайтесь одразу зрозуміти senior версію.
