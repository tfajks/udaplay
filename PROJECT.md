# UdaPlay — Gaming Research AI Agent

## Context

Build a two-notebook project (UdaPlay) that answers natural-language questions about video games using a RAG pipeline backed by ChromaDB and an OpenAI-powered agent with Tavily web-search fallback.

---

## Libraries (requirements.txt)

```
chromadb>=1.0.4
openai>=1.73.0
pydantic>=2.11.3
python-dotenv>=1.1.0
tavily-python>=0.5.4
```

---

## Environment (.env)

```
OPENAI_API_KEY="sk-**********"        # Vocareum key
TAVILY_API_KEY="tvly-********"
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv('config.env')
assert os.getenv('OPENAI_API_KEY') is not None
assert os.getenv('TAVILY_API_KEY') is not None
```

OpenAI client:
```python
from openai import OpenAI
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

---

## Deliverables

| File | Purpose |
|---|---|
| `Udaplay_01_solution_project.ipynb` | RAG pipeline — ChromaDB setup, embed game data, semantic search |
| `Udaplay_02_solution_project.ipynb` | Agent — tools, state machine, stateful queries, reporting |
| `data/games.json` | Local game dataset (20+ games, 10+ companies) |
| `requirements.txt` | Package list above |
| `.env` | API keys (not committed) |

---

## Part 1 — RAG Pipeline (Notebook 1)

### Goal
Load JSON game data → embed → store in ChromaDB → query semantically.

### Steps

1. **Load data** — read `data/games.json` containing games and companies
2. **Prepare documents** — convert each game/company record into a text chunk:
   ```
   Title: Elden Ring. Developer: FromSoftware. Publisher: Bandai Namco. Release: 2022-02-25. Platforms: PC, PS5, Xbox. Genre: Action RPG. Description: ...
   ```
3. **Embed + store** — use `chromadb.Client()` with `OpenAIEmbeddingFunction` (model `text-embedding-3-small`):
   ```python
   import chromadb
   from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

   client_db = chromadb.PersistentClient(path="./chroma_db")
   ef = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")
   collection = client_db.get_or_create_collection("games", embedding_function=ef)
   collection.upsert(documents=[...], ids=[...], metadatas=[...])
   ```
4. **Semantic search demo** — show 3 example queries with results

### Rubric criteria satisfied
- Loads and processes game JSON files ✓
- Adds to persistent ChromaDB with embeddings ✓
- Demonstrates semantic search ✓

---

## Part 2 — Agent (Notebook 2)

### Goal
Stateful agent that: retrieves from ChromaDB → evaluates quality → falls back to Tavily if needed → returns cited answer.

### Architecture

```
User query
    │
    ▼
[retrieve_game]  ──── ChromaDB semantic search
    │
    ▼
[evaluate_retrieval]  ──── LLM rates confidence 0-1
    │
    ├─ confidence >= 0.7 ──► [report_agent]  →  Final answer
    │
    └─ confidence < 0.7  ──► [game_web_search]  →  Tavily
                                    │
                                    ▼
                              [report_agent]  →  Final answer with web citation
```

### State machine

```python
from enum import Enum

class AgentState(Enum):
    RETRIEVE   = "retrieve"
    EVALUATE   = "evaluate"
    WEB_SEARCH = "web_search"
    REPORT     = "report"
    DONE       = "done"
```

### Tool 1 — retrieve_game

```python
def retrieve_game(query: str) -> dict:
    results = collection.query(query_texts=[query], n_results=3)
    return {"documents": results["documents"][0], "metadatas": results["metadatas"][0]}
```

### Tool 2 — evaluate_retrieval

```python
def evaluate_retrieval(query: str, retrieved_docs: list[str]) -> dict:
    # LLM rates confidence 0.0–1.0
    # Returns {"confidence": float, "reasoning": str}
```

### Tool 3 — game_web_search

```python
from tavily import TavilyClient
def game_web_search(query: str) -> dict:
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    results = tavily.search(query=query, max_results=3)
    return results
```

### Agent class

```python
class UdaPlayAgent:
    def __init__(self):
        self.conversation_history = []   # stateful across queries
        self.llm = OpenAI(base_url=..., api_key=...)
        self.model = "gpt-4o-mini"

    def invoke(self, query: str) -> dict:
        # runs state machine: RETRIEVE → EVALUATE → (WEB_SEARCH?) → REPORT
        # appends to self.conversation_history
        # returns {"answer": str, "source": "rag"|"web", "citations": [...], "confidence": float}
```

### Rubric criteria satisfied
- 3 tools: retrieve / evaluate / web_search ✓
- Agent tries internal first, evaluates, falls back to web ✓
- Stateful conversation history across queries ✓
- State machine workflow ✓
- Structured output with citations ✓
- 3 example queries with reasoning + tool usage shown ✓

---

## Data file — data/games.json

~20 games, ~10 companies. Fields per game:
```json
{
  "id": "elden-ring",
  "title": "Elden Ring",
  "developer": "FromSoftware",
  "publisher": "Bandai Namco Entertainment",
  "release_date": "2022-02-25",
  "platforms": ["PC", "PS4", "PS5", "Xbox One", "Xbox Series X/S"],
  "genre": ["Action RPG", "Soulslike"],
  "description": "Open-world action RPG...",
  "metacritic_score": 96
}
```

---

## Example queries for Notebook 2

1. `"Who developed Elden Ring and when was it released?"`
2. `"What platforms is Cyberpunk 2077 available on?"`
3. `"What is CD Projekt Red currently working on?"` ← likely triggers web fallback

---

## Verification

1. Run Notebook 1 end-to-end — confirm ChromaDB populated, 3 semantic search demos work
2. Run Notebook 2 — confirm all 3 example queries produce answers with source/citation fields
3. Confirm query 3 triggers web search path (confidence < 0.7)
4. Check `conversation_history` grows across multiple invocations

---

## Files to create

- `data/games.json`
- `Udaplay_01_solution_project.ipynb`
- `Udaplay_02_solution_project.ipynb`
- `.env` (user fills in keys)

## Files already present

- `requirements.txt` ✓
- `.env.example` ✓
