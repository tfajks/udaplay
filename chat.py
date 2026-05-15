"""
UdaPlay CLI — interactive gaming research agent.
Usage: python chat.py
"""
import os
import json
from enum import Enum
from dataclasses import dataclass, field
import hashlib
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY  = os.getenv('TAVILY_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://openai.vocareum.com/v1')
MODEL           = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

assert OPENAI_API_KEY,  'Missing OPENAI_API_KEY in .env'
assert TAVILY_API_KEY,  'Missing TAVILY_API_KEY in .env'


# ------------------------------------------------------------------ #
# State machine
# ------------------------------------------------------------------ #

class AgentState(Enum):
    RETRIEVE   = 'retrieve'
    EVALUATE   = 'evaluate'
    WEB_SEARCH = 'web_search'
    REPORT     = 'report'
    DONE       = 'done'


@dataclass
class AgentContext:
    query: str
    state: AgentState = AgentState.RETRIEVE
    retrieved_docs: list = field(default_factory=list)
    retrieved_meta: list = field(default_factory=list)
    confidence: float = 0.0
    confidence_reasoning: str = ''
    web_results: list = field(default_factory=list)
    final_answer: str = ''
    source: str = ''
    citations: list = field(default_factory=list)


# ------------------------------------------------------------------ #
# Lazy-loaded clients
# ------------------------------------------------------------------ #

_llm = None
_collection = None
_tavily = None
_games_index: dict = {}


def _get_llm():
    global _llm
    if _llm is None:
        from openai import OpenAI
        _llm = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    return _llm


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        db = chromadb.PersistentClient(path='./chroma_db')
        ef = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name='text-embedding-3-small')
        _collection = db.get_or_create_collection(
            name='games', embedding_function=ef, metadata={'hnsw:space': 'cosine'}
        )
        if _collection.count() == 0:
            print('[WARN] ChromaDB is empty — run Notebook 1 first to populate the vector store.')
    return _collection


def _get_tavily():
    global _tavily
    if _tavily is None:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily


def _get_games_index() -> dict:
    global _games_index
    if not _games_index:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'games.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _games_index = {g['title'].lower(): g for g in data['games']}
    return _games_index


# ------------------------------------------------------------------ #
# Tools
# ------------------------------------------------------------------ #

def retrieve_game(query: str, n_results: int = 3) -> dict:
    results = _get_collection().query(query_texts=[query], n_results=n_results)
    return {'documents': results['documents'][0], 'metadatas': results['metadatas'][0]}


def evaluate_retrieval(query: str, documents: list) -> dict:
    system = (
        'You are a quality evaluator for a gaming information system. '
        'Assess whether the retrieved documents answer the user query. '
        'Respond ONLY with valid JSON: {"confidence": <0.0-1.0>, "reasoning": "<one sentence>"}'
    )
    user = f'Query: {query}\n\nRetrieved documents:\n' + '\n---\n'.join(documents)
    response = _get_llm().chat.completions.create(
        model=MODEL,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        max_tokens=200,
    )
    text = response.choices[0].message.content or '{}'
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{[\s\S]*\}', text)
        return json.loads(m.group(0)) if m else {'confidence': 0.0, 'reasoning': 'parse error'}


def game_web_search(query: str) -> dict:
    results = _get_tavily().search(query=f'video game {query}', max_results=3, include_answer=True)
    col = _get_collection()
    for r in results.get('results', []):
        doc_id = 'web-' + hashlib.md5(r['url'].encode()).hexdigest()[:12]
        doc_text = (
            f"Web source: {r['title']}. URL: {r['url']}. "
            f"Content: {r.get('content', '')[:500]}"
        )
        try:
            col.upsert(
                documents=[doc_text], ids=[doc_id],
                metadatas=[{'type': 'web', 'title': r['title'], 'url': r['url']}],
            )
        except Exception:
            pass
    return results


def get_game_stats(title: str) -> dict:
    idx = _get_games_index()
    key = title.lower().strip()
    g = idx.get(key) or next((v for k, v in idx.items() if key in k or k in key), None)
    if not g:
        return {'found': False, 'title': title}
    return {
        'found': True, 'title': g['title'], 'developer': g['developer'],
        'publisher': g['publisher'], 'release_date': g['release_date'],
        'platforms': g['platforms'], 'metacritic_score': g['metacritic_score'],
    }


# ------------------------------------------------------------------ #
# Agent
# ------------------------------------------------------------------ #

class UdaPlayAgent:
    def __init__(self):
        # Keyed by session_id so each user's conversation is isolated.
        self._sessions: dict[str, list[dict]] = {}

    def _get_history(self, session_id: str) -> list[dict]:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def invoke(self, query: str, session_id: str = 'default') -> dict:
        ctx = AgentContext(query=query)

        while ctx.state != AgentState.DONE:
            if ctx.state == AgentState.RETRIEVE:
                self._retrieve(ctx)
            elif ctx.state == AgentState.EVALUATE:
                self._evaluate(ctx)
            elif ctx.state == AgentState.WEB_SEARCH:
                self._web_search(ctx)
            elif ctx.state == AgentState.REPORT:
                self._report(ctx, session_id)

        history = self._get_history(session_id)
        history.append({'role': 'user',      'content': query})
        history.append({'role': 'assistant', 'content': ctx.final_answer})
        return {'answer': ctx.final_answer, 'source': ctx.source,
                'confidence': ctx.confidence, 'citations': ctx.citations}

    def _retrieve(self, ctx: AgentContext):
        print('  Searching local knowledge base...', flush=True)
        r = retrieve_game(ctx.query)
        ctx.retrieved_docs = r['documents']
        ctx.retrieved_meta = r['metadatas']
        ctx.state = AgentState.EVALUATE

    def _evaluate(self, ctx: AgentContext):
        print('  Evaluating result quality...', flush=True)
        ev = evaluate_retrieval(ctx.query, ctx.retrieved_docs)
        ctx.confidence = float(ev.get('confidence', 0.0))
        ctx.confidence_reasoning = ev.get('reasoning', '')
        print(f'  Confidence: {ctx.confidence:.0%} — {ctx.confidence_reasoning}', flush=True)
        if ctx.confidence >= CONFIDENCE_THRESHOLD:
            ctx.source = 'rag'
            ctx.citations = [m.get('title') or m.get('name', 'Local DB') for m in ctx.retrieved_meta]
            ctx.state = AgentState.REPORT
        else:
            print('  Falling back to web search...', flush=True)
            ctx.state = AgentState.WEB_SEARCH

    def _web_search(self, ctx: AgentContext):
        web = game_web_search(ctx.query)
        ctx.web_results = web.get('results', [])
        ctx.source = 'web'
        ctx.citations = [r['url'] for r in ctx.web_results]
        print(f'  Web search: {len(ctx.web_results)} results found (saved to memory)', flush=True)
        ctx.state = AgentState.REPORT

    def _report(self, ctx: AgentContext, session_id: str = 'default'):
        if ctx.source == 'rag':
            context_text = '\n---\n'.join(ctx.retrieved_docs)
        else:
            context_text = '\n---\n'.join(
                f"{r['title']}: {r.get('content', '')[:400]}" for r in ctx.web_results
            )

        history_text = ''
        history = self._get_history(session_id)
        if history:
            recent = history[-6:]  # last 3 turns for this session only
            parts = []
            for i in range(0, len(recent) - 1, 2):
                parts.append(f"Q: {recent[i]['content']}\nA: {recent[i+1]['content']}")
            if parts:
                history_text = 'Previous conversation:\n' + '\n'.join(parts) + '\n\n'

        system = (
            'You are UdaPlay, an expert gaming research assistant. '
            'Answer the user query using only the provided context. '
            'Be factual and concise. Always mention your source.'
        )
        user = f'{history_text}Context:\n{context_text}\n\nQuestion: {ctx.query}'

        response = _get_llm().chat.completions.create(
            model=MODEL,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            max_tokens=600,
        )
        ctx.final_answer = response.choices[0].message.content or ''
        ctx.state = AgentState.DONE


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    import uuid
    session_id = str(uuid.uuid4())

    print('=' * 60)
    print('  UdaPlay — Gaming Research Agent')
    print(f'  Session: {session_id[:8]}...')
    print('  Type your question, or:')
    print('    /stats <game title>  — show structured stats')
    print('    /history             — show this session\'s history')
    print('    /quit                — exit')
    print('=' * 60)

    agent = UdaPlayAgent()

    while True:
        try:
            user_input = input('\nYou: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break

        if not user_input:
            continue

        if user_input.lower() in ('/quit', 'quit', 'exit'):
            print('Goodbye!')
            break

        if user_input.lower() == '/history':
            history = agent._get_history(session_id)
            if not history:
                print('No conversation history yet.')
            else:
                for msg in history:
                    prefix = 'You' if msg['role'] == 'user' else 'Agent'
                    print(f'\n{prefix}: {msg["content"]}')
            continue

        if user_input.lower().startswith('/stats '):
            title = user_input[7:].strip()
            stats = get_game_stats(title)
            print(json.dumps(stats, indent=2))
            continue

        print()
        result = agent.invoke(user_input, session_id=session_id)
        print(f'\nAgent ({result["source"].upper()}, confidence {result["confidence"]:.0%}):')
        print(result['answer'])
        if result['citations']:
            print(f'\nSources: {", ".join(result["citations"][:3])}')


if __name__ == '__main__':
    main()
