# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an educational AI agents course (Module 1) teaching how to build AI agents using different frameworks. The course is structured in Ukrainian and includes examples using LangChain, CrewAI, and SmolAgents. The repository demonstrates three different approaches to building AI research agents for educational purposes.

## Core Architecture

### Three AI Agent Frameworks Compared

1. **LangChain 1.0** (`examples/01_langchain_v1.py`)
   - Uses LCEL (LangChain Expression Language) chains
   - Structure: LLM + Tools + Chains pipeline
   - Pattern: `prompt | llm | output_parser`
   - Creates research/conclusion chains for sequential processing

2. **CrewAI** (`examples/02_crewai_agent.py`, `examples/02_crewai_simple.py`)
   - Multi-agent system with roles, goals, and backstories
   - Structure: Agents + Tasks + Crew with Process.sequential
   - Uses `@tool` decorator for custom tools
   - Supports agent memory and task context dependencies

3. **SmolAgents** (`examples/03_smolagents_agent.py`)
   - Minimalist code-first approach using CodeAgent
   - Structure: Model + Tools + CodeAgent
   - Generates Python code to solve tasks
   - Supports OpenAI, HuggingFace, and local models

### Common Agent Pattern

All agents follow this structure:
```python
class Agent:
    def __init__(self):
        self.tools = self._create_tools()  # Search, analyze, save
        self.agent = self._create_agent()  # Framework-specific

    def research(self, topic: str) -> dict:
        # 1. Search web for information
        # 2. Analyze data (sentiment, statistics)
        # 3. Generate AI-powered report
        # 4. Save to memory/file
```

## Development Commands

### Setup and Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Install specific framework
pip install langchain==1.0.0 langchain-openai==1.0.0  # LangChain
pip install crewai==0.203.1 crewai-tools==0.14.0      # CrewAI
pip install smolagents==1.22.0                         # SmolAgents
```

### Running Examples
```bash
# Quick start (auto-detects API key)
bash quick_start.sh

# Test all agents
python3 test_agents.py

# Run individual agents
python3 examples/01_langchain_v1.py    # LangChain research agent
python3 examples/02_crewai_agent.py    # CrewAI multi-agent team
python3 examples/02_crewai_simple.py   # Simple CrewAI example
python3 examples/03_smolagents_agent.py # SmolAgents code agent
```

### Environment Configuration
```bash
# Create .env file with API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# All agents work in demo mode without API key
# They will use fallback data for search and simulated AI responses
```

## Key Technical Details

### API Key Handling
- All agents check for `OPENAI_API_KEY` via `os.getenv()`
- Graceful degradation: agents run in demo mode without API keys
- Demo mode uses hardcoded search results and sentiment analysis
- Real mode requires OpenAI GPT-4 access

### Tool System
Each framework implements these core tools differently:
- **Web Search**: Uses `ddgs` (DuckDuckGo Search) or demo fallback
- **Data Analysis**: Sentiment analysis via keyword matching
- **Memory/Save**: Stores results to JSON files
- **Date/Time**: Gets current timestamp

### Output Files
Agents generate these files:
- `langchain1_report.json`, `langchain1_memory.json` - LangChain outputs
- `crewai_report_*.json`, `crewai_final_*.json` - CrewAI outputs
- `smolagents_memory.json` - SmolAgents memory storage

### Error Handling Pattern
All agents implement try/except with fallbacks:
```python
try:
    # Use real API/search
except Exception as e:
    # Fall back to demo mode
    # Return simulated results
```

## Important Package Versions

Working versions verified for Python 3.10+:
- `langchain==1.0.0` (not 0.x - breaking changes)
- `langchain-community==0.3.31` (latest available, not 1.0)
- `crewai==0.203.1` (no LangChain dependencies in simple example)
- `smolagents==1.22.0`
- `openai==1.109.1`
- `ddgs>=1.0.0` (replaces deprecated duckduckgo-search)

Note: The course explicitly avoids `duckduckgo-search==6.3.0` (deprecated) in favor of `ddgs`.

## Code Modifications Best Practices

When modifying agent code:
1. Preserve demo mode fallbacks for educational use
2. Maintain Ukrainian language in prompts and outputs
3. Keep verbose=True for educational visibility
4. Use structured JSON output for results
5. Implement graceful error handling with helpful messages

### Adding New Tools
For LangChain:
```python
def new_tool(input: str) -> str:
    """Tool description"""
    return result

tools["new_tool"] = new_tool
```

For CrewAI:
```python
@tool("Tool Name")
def new_tool(input: str) -> str:
    """Tool description"""
    return result
```

For SmolAgents:
```python
@tool
def new_tool(input: str) -> str:
    """
    Tool description with proper docstring.

    Args:
        input: Description

    Returns:
        Description
    """
    return result
```

## Testing
- `test_agents.py` provides automated testing without manual intervention
- Checks environment, package versions, and API key availability
- Runs all agents in appropriate mode (demo vs API)
- Interactive mode selection when API key is present

## Course Learning Path
The repository follows a progressive learning structure (per README):
1. Level 1 (Beginner): Simple agents, ~100 lines
2. Level 2 (Intermediate): LangChain v1, ~300 lines
3. Level 3 (Advanced): Production patterns, ~700 lines (not included in current files)

Focus is on educational clarity over production optimization.
