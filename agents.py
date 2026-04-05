"""
agents.py — NexusAI Multi-Agent System v4.0 (Ultra-Fast)
=========================================================
  ✅ llama-3.1-8b-instant — fast Groq model for low-latency responses
  ✅ Smart router        — simple queries (hi, hello, thanks) answered in
                           1 direct LLM call, skip Planner+Worker+Reviewer
  ✅ max_tokens 500      — fast output for all queries
  ✅ max_revisions 0     — no looping ever
  ✅ Tight prompts       — less tokens in = less time waiting
  ✅ httpx proxies patch — no crash on newer httpx
  ✅ No output bug fixed — 3-level fallback always shows something
  ✅ llama3-70b-8192 removed — it is DECOMMISSIONED by Groq
"""

import os
import re
import json
import logging
import time
from typing import TypedDict, Optional, List

# ── httpx / groq compatibility patch ─────────────────────────────────────────
try:
    import httpx as _httpx
    _orig_client_init = _httpx.Client.__init__
    def _patched_client_init(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        _orig_client_init(self, *args, **kwargs)
    _httpx.Client.__init__ = _patched_client_init

    _orig_async_init = _httpx.AsyncClient.__init__
    def _patched_async_init(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        _orig_async_init(self, *args, **kwargs)
    _httpx.AsyncClient.__init__ = _patched_async_init
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langgraph.graph import StateGraph, END

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — STRUCTURED LOGGING
# ══════════════════════════════════════════════════════════════════════════════
class StructuredLogger:
    def __init__(self, name: str):
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(h)
        self._log.setLevel(logging.INFO)

    def _emit(self, level: str, agent: str, event: str, **kw):
        entry = {"ts": time.strftime("%H:%M:%S"), "level": level,
                 "agent": agent, "event": event, **kw}
        getattr(self._log, level.lower(), self._log.info)(json.dumps(entry))

    def info(self,  agent, event, **kw): self._emit("INFO",    agent, event, **kw)
    def warn(self,  agent, event, **kw): self._emit("WARNING", agent, event, **kw)
    def error(self, agent, event, **kw): self._emit("ERROR",   agent, event, **kw)

logger = StructuredLogger("multi_agent")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE GROQ MODELS (current):
#   llama-3.1-8b-instant    ~2-5s   FASTEST  ← we use this
#   llama-3.3-70b-versatile ~15-30s smarter but slower
#   openai/gpt-oss-20b      ~3-8s    fast + strong
#
# DEAD — DO NOT USE:
#   llama3-70b-8192  ← DECOMMISSIONED

CONFIG = {
    # ── Model — 8B is the sweet spot: fast + smart ──
    "model":              os.getenv("AGENT_MODEL",         "llama-3.1-8b-instant"),
    "temperature":        float(os.getenv("AGENT_TEMP",    "0.4")),
    "max_tokens":         int(os.getenv("AGENT_TOKENS",    "500")),

    "groq_api_key":       os.getenv("GROQ_API_KEY",        ""),

    # ── Keep retries minimal for speed ──
    "max_retries":        int(os.getenv("AGENT_RETRIES",   "1")),
    "retry_delay":        float(os.getenv("RETRY_DELAY",   "0.3")),

    # ── Disable revision loops completely ──
    "max_revisions":      int(os.getenv("MAX_REVISIONS",   "0")),
    "revision_threshold": int(os.getenv("REVISION_THRESH", "10")),

    # ── Memory ──
    "memory_k":           int(os.getenv("MEMORY_K",        "2")),
    "embed_model":        os.getenv("EMBED_MODEL",          "all-MiniLM-L6-v2"),

    # ── Tools ──
    "tavily_key":         os.getenv("TAVILY_API_KEY",       ""),
    "max_search_results": int(os.getenv("MAX_SEARCH",       "2")),
}

# Simple queries that should bypass the full pipeline and get instant replies
SIMPLE_PATTERNS = [
    r"^(hi|hello|hey|hiya|howdy|yo)[\s!?.]*$",
    r"^(thanks|thank you|thx|ty|ok|okay|cool|great|nice|sure|got it)[\s!?.]*$",
    r"^(bye|goodbye|see you|cya|later)[\s!?.]*$",
    r"^(how are you|how r u|whats up|what's up|wassup)[\s!?.]*$",
    r"^(good morning|good night|good afternoon|gm|gn)[\s!?.]*$",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LLM SINGLETON
# ══════════════════════════════════════════════════════════════════════════════
_llm_instance: Optional[ChatGroq] = None

def get_llm() -> ChatGroq:
    global _llm_instance
    if _llm_instance is None:
        key = CONFIG["groq_api_key"]
        if not key:
            raise ValueError("GROQ_API_KEY is not set.")
        _llm_instance = ChatGroq(
            model=CONFIG["model"],
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
            groq_api_key=key,
        )
        logger.info("LLM", "ready", model=CONFIG["model"])
    return _llm_instance

def reset_llm():
    global _llm_instance
    _llm_instance = None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VECTOR MEMORY
# ══════════════════════════════════════════════════════════════════════════════
_embed_model = None
_vector_store = None
_memory_docs: List[str] = []

def _get_embeddings():
    global _embed_model
    if _embed_model is None and FAISS_AVAILABLE:
        _embed_model = HuggingFaceEmbeddings(
            model_name=CONFIG["embed_model"],
            model_kwargs={"device": "cpu"},
        )
    return _embed_model

def memory_save(user_input: str, final_output: str):
    global _vector_store, _memory_docs
    if not FAISS_AVAILABLE:
        return
    doc = f"Q: {user_input}\nA: {final_output}"
    _memory_docs.append(doc)
    try:
        emb = _get_embeddings()
        if _vector_store is None:
            _vector_store = FAISS.from_texts([doc], emb)
        else:
            _vector_store.add_texts([doc])
    except Exception as e:
        logger.warn("Memory", "save_failed", error=str(e))

def memory_retrieve(query: str) -> str:
    if not FAISS_AVAILABLE or _vector_store is None:
        return ""
    try:
        docs = _vector_store.similarity_search(query, k=CONFIG["memory_k"])
        if not docs:
            return ""
        return "\n".join(f"[Past]\n{d.page_content}" for d in docs)
    except Exception:
        return ""

def memory_clear():
    global _vector_store, _memory_docs
    _vector_store = None
    _memory_docs = []
    logger.info("Memory", "cleared")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TOOLS
# ══════════════════════════════════════════════════════════════════════════════
def web_search_tool(query: str) -> str:
    if not TAVILY_AVAILABLE or not CONFIG["tavily_key"]:
        return "[Web Search] Set TAVILY_API_KEY to enable."
    try:
        searcher = TavilySearchResults(
            max_results=CONFIG["max_search_results"],
            tavily_api_key=CONFIG["tavily_key"],
        )
        results = searcher.invoke(query)
        if not results:
            return "No results found."
        return "\n".join(
            f"• {r.get('title','')}: {r.get('content','')[:200]}" for r in results
        )
    except Exception as e:
        return f"[Web Search Error] {e}"

def calculator_tool(expression: str) -> str:
    allowed = set("0123456789+-*/.() ,")
    cleaned = expression.strip()
    if not all(c in allowed for c in cleaned):
        return "[Calculator] Only numeric expressions allowed."
    try:
        return f"Result: {eval(cleaned, {'__builtins__': {}})}"
    except Exception as e:
        return f"[Calculator Error] {e}"

def file_reader_tool(filepath: str) -> str:
    path = filepath.strip().strip('"\'')
    if not os.path.exists(path):
        return f"[File Error] Not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(2000)
        return content + ("\n[...truncated]" if len(content) == 2000 else "")
    except Exception as e:
        return f"[File Error] {e}"

TOOLS: List[Tool] = [
    Tool(name="web_search",  func=web_search_tool,
         description="Search the web for current information."),
    Tool(name="calculator",  func=calculator_tool,
         description="Evaluate a math expression."),
    Tool(name="file_reader", func=file_reader_tool,
         description="Read a local text file."),
]
TOOL_MAP = {t.name: t.func for t in TOOLS}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — JSON PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_json(text: str, default: dict) -> dict:
    if not text:
        return default
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return default


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LLM INVOKE WITH RETRY
# ══════════════════════════════════════════════════════════════════════════════
def invoke_with_retry(messages: list, agent_name: str = "Agent") -> str:
    llm = get_llm()
    max_retries = CONFIG["max_retries"]
    delay = CONFIG["retry_delay"]
    for attempt in range(1, max_retries + 2):
        try:
            logger.info(agent_name, "invoke", attempt=attempt)
            return llm.invoke(messages).content.strip()
        except Exception as e:
            logger.warn(agent_name, "error", attempt=attempt, error=str(e))
            if attempt <= max_retries:
                time.sleep(delay * attempt)
            else:
                logger.error(agent_name, "failed", error=str(e))
                return json.dumps({"error": f"{agent_name} failed: {e}"})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SMART ROUTER  (NEW — key to speed)
# ══════════════════════════════════════════════════════════════════════════════
def is_simple_query(text: str) -> bool:
    """
    Returns True for greetings and very short chitchat.
    These bypass the full Planner→Worker→Reviewer pipeline.
    """
    t = text.strip().lower()
    # very short = likely simple
    if len(t.split()) <= 3:
        for pattern in SIMPLE_PATTERNS:
            if re.match(pattern, t, re.IGNORECASE):
                return True
    return False

def direct_reply(user_input: str) -> str:
    """
    Single LLM call for simple queries. No pipeline overhead.
    Target: ~2-3 seconds.
    """
    system = (
        "You are NexusAI, a helpful AI assistant. "
        "Reply naturally and concisely. Keep it short and friendly."
    )
    raw = invoke_with_retry(
        [SystemMessage(content=system), HumanMessage(content=user_input)],
        agent_name="DirectReply",
    )
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    user_input:      str
    plan_json:       dict
    worker_json:     dict
    reviewer_json:   dict
    revision_count:  int
    quality_score:   int
    tools_used:      List[str]
    error:           Optional[str]
    trace:           List[str]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — AGENTS  (tight prompts for speed)
# ══════════════════════════════════════════════════════════════════════════════
def planner_agent(state: AgentState) -> AgentState:
    logger.info("Planner", "start")
    context   = memory_retrieve(state["user_input"])
    ctx_block = f"Context: {context}\n" if context else ""

    system = (
        'You are a Planner. Reply ONLY with this JSON, nothing else:\n'
        '{"summary":"one sentence","steps":[{"id":1,"title":"...","description":"..."}],'
        '"needs_web_search":false,"complexity":"low"}\n'
        'Use 1-3 steps maximum. No markdown fences.'
    )
    user = f"{ctx_block}Request: {state['user_input']}"

    raw  = invoke_with_retry([SystemMessage(content=system), HumanMessage(content=user)],
                             agent_name="Planner")
    default = {
        "summary": state["user_input"],
        "steps": [{"id": 1, "title": "Respond", "description": state["user_input"]}],
        "needs_web_search": False,
        "complexity": "low",
    }
    plan = parse_json(raw, default)
    plan["steps"] = plan.get("steps", [])[:3]
    logger.info("Planner", "done", steps=len(plan["steps"]))
    return {**state, "plan_json": plan,
            "trace": state.get("trace", []) + ["planner"], "error": None}


def worker_agent(state: AgentState) -> AgentState:
    logger.info("Worker", "start")
    plan  = state.get("plan_json", {})
    steps = plan.get("steps", [])

    search_ctx = ""
    if plan.get("needs_web_search") and CONFIG["tavily_key"]:
        search_ctx = web_search_tool(state["user_input"])

    steps_text     = "\n".join(f"{s['id']}. {s['title']}: {s['description']}" for s in steps)
    search_section = f"\nSearch results:\n{search_ctx}\n" if search_ctx else ""
    context        = memory_retrieve(state["user_input"])
    ctx_block      = f"Context: {context}\n" if context else ""

    system = (
        'You are a Worker Agent. Reply ONLY with this JSON, nothing else:\n'
        '{"output":"complete response here","tools_used":[],"step_results":[]}\n'
        'Write a full, helpful answer in "output". For code, include the complete code. '
        'No markdown fences around the JSON itself.'
    )
    user = (
        f"Request: {state['user_input']}\n"
        f"Steps:\n{steps_text}"
        f"{search_section}"
        f"{ctx_block}"
    )

    raw    = invoke_with_retry([SystemMessage(content=system), HumanMessage(content=user)],
                               agent_name="Worker")
    default = {
        "output": raw,
        "tools_used": ["web_search"] if search_ctx else [],
        "step_results": [],
    }
    result = parse_json(raw, default)

    # Always ensure output is populated
    if not result.get("output", "").strip():
        result["output"] = raw

    tools_used = list(set(result.get("tools_used", []) + (["web_search"] if search_ctx else [])))
    logger.info("Worker", "done", chars=len(result.get("output", "")))
    return {**state, "worker_json": result, "tools_used": tools_used,
            "trace": state.get("trace", []) + ["worker"], "error": None}


def reviewer_agent(state: AgentState) -> AgentState:
    logger.info("Reviewer", "start")
    worker        = state.get("worker_json", {})
    worker_output = worker.get("output", "").strip()

    # If worker produced nothing, skip LLM call
    if not worker_output:
        result = {
            "score": 5, "issues": [], "improvements": "",
            "final_output": "Sorry, could not generate a response. Please try again.",
        }
        return {**state, "reviewer_json": result, "quality_score": 5,
                "trace": state.get("trace", []) + ["reviewer"], "error": None}

    system = (
        'You are a Reviewer. Reply ONLY with this JSON, nothing else:\n'
        '{"score":8,"issues":[],"improvements":"","final_output":"polished response"}\n'
        'Copy the worker output into final_output, fix any issues, keep code intact. '
        'No markdown fences around the JSON.'
    )
    user = (
        f"Request: {state['user_input']}\n\n"
        f"Worker output:\n{worker_output[:1200]}\n\n"
        "Return reviewer JSON."
    )

    raw    = invoke_with_retry([SystemMessage(content=system), HumanMessage(content=user)],
                               agent_name="Reviewer")
    default = {
        "score": 7, "issues": [], "improvements": "",
        "final_output": worker_output,
    }
    result = parse_json(raw, default)

    # Always fall back to worker_output if final_output is empty
    if not result.get("final_output", "").strip():
        result["final_output"] = worker_output

    score = max(1, min(10, int(result.get("score", 7))))
    result["score"] = score

    if score >= CONFIG["revision_threshold"]:
        memory_save(state["user_input"], result["final_output"])

    logger.info("Reviewer", "done", score=score)
    return {**state, "reviewer_json": result, "quality_score": score,
            "trace": state.get("trace", []) + ["reviewer"], "error": None}


# ── Routing ──────────────────────────────────────────────────────────────────
def should_revise(state: AgentState) -> str:
    score = state.get("quality_score", 7)
    revs  = state.get("revision_count", 0)
    if score < CONFIG["revision_threshold"] and revs < CONFIG["max_revisions"]:
        return "revise"
    return "done"

def increment_revision(state: AgentState) -> AgentState:
    return {**state,
            "revision_count": state.get("revision_count", 0) + 1,
            "trace": state.get("trace", []) + ["revision_loop"]}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — LANGGRAPH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner",            planner_agent)
    g.add_node("worker",             worker_agent)
    g.add_node("reviewer",           reviewer_agent)
    g.add_node("increment_revision", increment_revision)

    g.set_entry_point("planner")
    g.add_edge("planner",            "worker")
    g.add_edge("worker",             "reviewer")
    g.add_conditional_edges("reviewer", should_revise,
                            {"revise": "increment_revision", "done": END})
    g.add_edge("increment_revision", "worker")
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(user_input: str) -> dict:
    """
    Entry point called by app.py.

    FAST PATH: simple greetings → 1 LLM call → ~2-3s
    FULL PATH: coding/analysis  → Planner+Worker+Reviewer → ~8-15s
    """
    logger.info("Pipeline", "start", preview=user_input[:60])
    t0 = time.time()

    # ── Fast path for simple queries ──────────────────────────────────────────
    if is_simple_query(user_input):
        logger.info("Pipeline", "fast_path")
        try:
            answer  = direct_reply(user_input)
            elapsed = round(time.time() - t0, 2)
            logger.info("Pipeline", "fast_done", elapsed_s=elapsed)
            return {
                "plan":           {"summary": user_input, "steps": [],
                                   "complexity": "low", "needs_web_search": False},
                "worker_output":  answer,
                "final_output":   answer,
                "reviewer_json":  {"score": 9, "issues": [],
                                   "improvements": "", "final_output": answer},
                "quality_score":  9,
                "revision_count": 0,
                "tools_used":     [],
                "trace":          ["direct"],
                "error":          None,
            }
        except Exception as e:
            logger.error("Pipeline", "fast_path_error", error=str(e))
            # fall through to full pipeline

    # ── Full multi-agent pipeline ─────────────────────────────────────────────
    graph = build_graph()
    initial: AgentState = {
        "user_input":     user_input,
        "plan_json":      {},
        "worker_json":    {},
        "reviewer_json":  {},
        "revision_count": 0,
        "quality_score":  0,
        "tools_used":     [],
        "error":          None,
        "trace":          [],
    }

    try:
        state   = graph.invoke(initial)
        elapsed = round(time.time() - t0, 2)

        reviewer_json = state.get("reviewer_json", {})
        worker_output = state.get("worker_json", {}).get("output", "").strip()

        # 3-level fallback — final_output is NEVER empty
        final_output = (
            reviewer_json.get("final_output", "").strip()
            or worker_output
            or "No output generated."
        )

        logger.info("Pipeline", "done", elapsed_s=elapsed,
                    score=state.get("quality_score"),
                    chars=len(final_output))

        return {
            "plan":           state.get("plan_json", {}),
            "worker_output":  worker_output,
            "final_output":   final_output,
            "reviewer_json":  reviewer_json,
            "quality_score":  state.get("quality_score", 0),
            "revision_count": state.get("revision_count", 0),
            "tools_used":     state.get("tools_used", []),
            "trace":          state.get("trace", []),
            "error":          state.get("error"),
        }

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        logger.error("Pipeline", "fatal", error=str(e), elapsed_s=elapsed)
        return {
            "plan": {}, "worker_output": "",
            "final_output": f"Error: {e}",
            "reviewer_json": {}, "quality_score": 0,
            "revision_count": 0, "tools_used": [],
            "trace": [], "error": str(e),
        }
