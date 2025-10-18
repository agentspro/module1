# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an educational AI agents course (Module 1) teaching how to build AI agents using different frameworks. The course is structured in Ukrainian and includes examples using LangChain, CrewAI, and SmolAgents. The repository demonstrates three different approaches to building AI research agents for educational purposes.

## Core Architecture

### Four AI Agent Frameworks Compared

1. **LangChain 1.0** (`examples/01_langchain_v1.py`)
   - Uses LCEL (LangChain Expression Language) chains
   - Structure: LLM + Tools + Chains pipeline
   - Pattern: `prompt | llm | output_parser`
   - Creates research/conclusion chains for sequential processing
   - Single-agent approach

2. **LangChain + LangGraph** (`examples/02_langchain_langgraph.py`)
   - Multi-agent system using StateGraph for orchestration
   - Structure: StateGraph + Nodes (agents) + Edges (flow)
   - State machine pattern with typed state sharing
   - Three agents: Researcher → Analyst → Reporter
   - Each agent is a node function that receives and returns state
   - Graph compiled and executed with `app.invoke(initial_state)`

3. **CrewAI** (`examples/04_crewai_agents.py`, `examples/03_crewai_simple.py`)
   - Multi-agent system with roles, goals, and backstories
   - Structure: Agents + Tasks + Crew with Process.sequential
   - Uses `@tool` decorator for custom tools
   - Supports agent memory and task context dependencies

4. **SmolAgents** (`examples/05_smolagents_agent.py`, `examples/06_smolagents_multiagent.py`)
   - Minimalist code-first approach using CodeAgent
   - Structure: Model + Tools + CodeAgent
   - Generates Python code to solve tasks
   - Supports OpenAI, HuggingFace, and local models
   - Multi-agent version demonstrates two approaches:
     - Sequential: One CodeAgent with three-stage task
     - Multi-Agent: Three CodeAgent instances with different system prompts

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
python3 examples/01_langchain_v1.py         # LangChain LCEL chains
python3 examples/02_langchain_langgraph.py  # LangGraph multi-agent
python3 examples/03_crewai_simple.py        # Simple CrewAI agent
python3 examples/04_crewai_agents.py        # CrewAI multi-agent team
python3 examples/05_smolagents_agent.py     # SmolAgents single agent
python3 examples/06_smolagents_multiagent.py # SmolAgents multi-agent
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
- `langgraph_report_*.json` - LangGraph multi-agent outputs
- `crewai_report_*.json`, `crewai_final_*.json` - CrewAI outputs
- `smolagents_memory.json` - SmolAgents single agent memory
- `smolagents_multiagent_memory.json`, `smolagents_multiagent_*.json` - SmolAgents multi-agent outputs

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
- `langgraph>=0.2.0` (for multi-agent orchestration)
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
For LangChain (single agent):
```python
def new_tool(input: str) -> str:
    """Tool description"""
    return result

tools["new_tool"] = new_tool
```

For LangGraph (as node function):
```python
def new_agent_node(state: AgentState) -> AgentState:
    """Node function for new agent"""
    result = process_data(state["input_data"])
    return {
        "output_data": result,
        "messages": ["✅ New agent completed"]
    }

# Add to graph
workflow.add_node("new_agent", new_agent_node)
workflow.add_edge("previous_agent", "new_agent")
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

## Framework Comparison
The repository demonstrates four approaches to AI agents:
1. **LangChain LCEL** (~350 lines): Chains and pipelines, single agent
2. **LangGraph** (~400 lines): State graph multi-agent, explicit orchestration
3. **CrewAI** (~300 lines): Role-based multi-agent, task dependencies
4. **SmolAgents** (~270 lines): Code generation, minimal abstraction

Key Differences:
- **State Management**: LangGraph uses TypedDict state, CrewAI uses context, LangChain uses chains
- **Orchestration**: LangGraph = explicit graph, CrewAI = Process.sequential, LangChain = chain operators
- **Agent Definition**: LangGraph = node functions, CrewAI = Agent class with role/goal/backstory
- **Visualization**: Only LangGraph supports graph visualization

Focus is on comparing different architectural approaches to the same problem (AI research task).
