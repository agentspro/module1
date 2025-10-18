"""
CrewAI Agent - Демо версія для GitHub Codespaces
Працює без API ключів
"""

from datetime import datetime
import json

def demo_crewai():
    """Демонстрація CrewAI без API"""
    
    print("🚢 CrewAI Agent (Demo Mode)")
    print("Екіпаж: Дослідник, Аналітик, Письменник")
    
    # Симуляція роботи екіпажу
    crew_actions = [
        ("👤 Дослідник", "Зібрав дані про AI тренди"),
        ("👤 Аналітик", "Проаналізував вплив на студентів"),
        ("👤 Письменник", "Створив детальний звіт")
    ]
    
    for agent, action in crew_actions:
        print(f"  {agent}: {action}")
    
    # Створення результату
    result = {
        "framework": "CrewAI",
        "crew": ["Дослідник", "Аналітик", "Письменник"],
        "topic": "AI в освіті",
        "collaboration_result": {
            "research": "15 джерел проаналізовано",
            "analysis": "3 ключові тренди виявлено",
            "report": "Звіт на 5 сторінок створено"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Збереження
    with open("crewai_demo_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Результат збережено: crewai_demo_result.json")
    
    return result

if __name__ == "__main__":
    demo_crewai()
