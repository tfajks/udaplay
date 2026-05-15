# UdaPlay — Gaming Research AI Agent

A two-notebook project that answers natural-language questions about video games using a RAG pipeline (ChromaDB + OpenAI embeddings) and a stateful agent with Tavily web-search fallback.

---

## Project Structure

```
udaplay/
├── data/
│   └── games.json                        # 22 games + 10 companies
├── Udaplay_01_solution_project.ipynb     # Part 1: RAG pipeline
├── Udaplay_02_solution_project.ipynb     # Part 2: Agent + tool traces
├── chat.py                               # Interactive CLI (python chat.py)
├── requirements.txt
└── .env                                  # API keys (not committed)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env`
```
OPENAI_API_KEY=your-vocareum-or-openai-key
OPENAI_BASE_URL=https://openai.vocareum.com/v1
OPENAI_MODEL=gpt-4o-mini
TAVILY_API_KEY=your-tavily-key
CONFIDENCE_THRESHOLD=0.7
```

### 3. Run Notebook 1 — build the vector store
Open `Udaplay_01_solution_project.ipynb` and run all cells.

### 4. Run Notebook 2 — run the agent
Open `Udaplay_02_solution_project.ipynb` and run all cells.

### 5. Or use the CLI
```bash
python chat.py
```

---

## Part 1 — RAG Pipeline (`Udaplay_01_solution_project.ipynb`)

**Goal:** Load game JSON data → embed → store in ChromaDB → demonstrate semantic search.

| Step | What happens |
|---|---|
| Load | `data/games.json` — 22 games, 10 companies |
| Format | Each record converted to a rich text chunk |
| Embed | `text-embedding-3-small` via OpenAI API |
| Store | `chromadb.PersistentClient` — persists to `./chroma_db` |
| Demo | 3 semantic search queries with results |

---

## Part 2 — Agent (`Udaplay_02_solution_project.ipynb`)

### Agent workflow

```
User query
    │
    ▼
[retrieve_game]       ChromaDB semantic search — top 3 docs
    │
    ▼
[evaluate_retrieval]  LLM rates confidence 0.0–1.0
    │
    ├─ >= 0.7 ──► [report_agent]   answer  (source=rag)
    │
    └─ < 0.7  ──► [game_web_search]  Tavily search
                        │
                        ├── save results to ChromaDB  ← Advanced Memory
                        ▼
                  [report_agent]   answer  (source=web, citations=[url])
```

### State machine

```python
class AgentState(Enum):
    RETRIEVE   = "retrieve"
    EVALUATE   = "evaluate"
    WEB_SEARCH = "web_search"
    REPORT     = "report"
    DONE       = "done"
```

### Tools

| Tool | Description |
|---|---|
| `retrieve_game(query)` | Semantic search in ChromaDB — returns top 3 docs |
| `evaluate_retrieval(query, docs)` | LLM rates confidence 0.0–1.0 + reasoning |
| `game_web_search(query)` | Tavily search + saves results to ChromaDB (Advanced Memory) |
| `get_game_stats(title)` | Returns structured stats (Metacritic score, platforms) from local DB |

### Session isolation

Each conversation gets a unique `session_id` (UUID). Conversation history is stored per session — users cannot access each other's context:

```python
SESSION_ID = str(uuid.uuid4())
agent = UdaPlayAgent()

result = agent.invoke("Who made Elden Ring?", session_id=SESSION_ID)
```

### Structured output

Every response returns both natural language and a JSON dict:

```json
{
  "answer": "Elden Ring was developed by FromSoftware...",
  "source": "rag",
  "confidence": 1.0,
  "citations": ["Elden Ring", "FromSoftware", "Dark Souls III"]
}
```

### Tool trace (visible in notebook output)

Each query shows the full step-by-step workflow:

```
QUERY: Who developed Elden Ring and when was it released?
────────────────────────────────────────────────────────────
  Step 1: [retrieve_game]
           top results: ['Elden Ring', 'FromSoftware', ...]
  Step 2: [evaluate_retrieval]
           confidence=1.00: Document clearly answers the query.
  Step 3: [report_agent]
           answer generated (312 chars)
────────────────────────────────────────────────────────────
  SOURCE:     RAG
  CONFIDENCE: 1.00
  CITATIONS:  ['Elden Ring', 'FromSoftware', 'Dark Souls III']
```

### Example queries

| Query | Expected path |
|---|---|
| `"Who developed Elden Ring and when was it released?"` | RAG (high confidence) |
| `"What platforms is Cyberpunk 2077 available on?"` | RAG (high confidence) |
| `"What is CD Projekt Red currently working on in 2025?"` | Web fallback (low confidence) |
| `"Is any of those games on Nintendo Switch?"` | RAG + conversation history |

---

## CLI (`chat.py`)

```
python chat.py
```

| Command | Action |
|---|---|
| Any question | Agent answers (RAG or web) |
| `/stats <title>` | Structured JSON stats for a game |
| `/history` | Show this session's conversation history |
| `/quit` | Exit |

---

## Bonus features

| Feature | Implementation |
|---|---|
| Advanced Memory | Web search results saved back to ChromaDB |
| Structured Output | Every answer returns natural language + JSON |
| Custom Tool | `get_game_stats` — Metacritic score, platforms, awards |
| Session isolation | `session_id` prevents context leakage between users |
| Executed notebooks | Both notebooks include saved cell outputs |
