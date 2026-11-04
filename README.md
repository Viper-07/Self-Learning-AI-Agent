<div align="center">

<!-- Hero Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1e1b4b,100:0284c7&height=220&section=header&text=S%20E%20N%20T%20I%20A&fontSize=62&fontAlignY=38&desc=Autonomous%20Cognitive%20AI%20Architecture%20%E2%80%A2%20Continuous%20Learning&descAlignY=60&descAlign=50&stroke=38bdf8&strokeWidth=1" width="100%" alt="Sentia Header Banner"/>

<!-- Typing Animation Subtitle -->
<p align="center">
  <a href="https://github.com/Viper-07/Self-Learning-AI-Agent">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=17&duration=2500&pause=1000&color=38BDF8&center=true&vCenter=true&width=620&lines=Plan.+Reason.+Act.+Observe.+Reflect.+Learn.;Autonomous+ReAct+Loop+with+Self-Correction;Multi-Tier+Memory+backed+by+ChromaDB+%2B+SQLite;Reinforcement+Learning+from+Human+Feedback+%26+Experience+Replay" alt="Typing SVG" />
  </a>
</p>

<!-- Tech Stack & Status Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/SQLite-Async-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge&logo=chromadb&logoColor=white" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/LiteLLM-Multi_Provider-6366F1?style=for-the-badge" alt="LiteLLM" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-cognitive-cycle">Cognitive Cycle</a> •
  <a href="#-tiered-memory-system">Memory</a> •
  <a href="#-rest-api-endpoints">API</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-security--sandbox">Security</a>
</p>

---

</div>

## 🌟 Overview

**Sentia** is not another simple LLM wrapper. It is a full-fledged **Autonomous Cognitive Agent** engineered with an introspective reasoning loop, strategic goal decomposition, multi-tiered contextual memory, and a continuous learning pipeline.

Rather than providing static single-turn responses, Sentia **plans**, **reasons**, **invokes tools**, **observes outcomes**, **self-reflects on errors**, and **learns from its own experiences** to continuously improve its future decision-making.

<br>

<div align="center">

### 🧩 Core Architecture Matrix

| Subsystem | Functional Role | Tech Stack | Indicator |
| :--- | :--- | :--- | :---: |
| **Strategic Planner** | Decomposes high-level ambiguous goals into ordered execution plans | LLM + Pydantic JSON Schema | 🗺️ |
| **Reasoning Engine** | Drives ReAct (*Think $\rightarrow$ Act $\rightarrow$ Observe*) with failure reflection | LiteLLM + Structured Output | ⚙️ |
| **Memory Manager** | Manages Short-Term, Semantic, Long-Term, and Episodic context | ChromaDB + SQLAlchemy Async | 💾 |
| **Learning Pipeline** | Experience replay buffer & reinforcement from user feedback | Experience Replay + Reward Signal | 📈 |
| **Tool Sandbox** | Secure execution of Python code, search, and file manipulation | Subprocesses + Sandboxed I/O | 🛡️ |

</div>

---

## 🧠 Cognitive Cycle

Sentia thinks and acts through a closed-loop reasoning pipeline:

```mermaid
flowchart TD
    subgraph Input [" 📥 User Input "]
        Q[User Query]
    end

    subgraph MemoryRetrieval [" 💾 Context Hydration "]
        ST[Short-Term Conversation Buffer]
        SEM[(ChromaDB: Semantic Vectors)]
        EPI[(SQLite: Episodic Interaction History)]
        ST --> Ctx[Augmented Prompt Context]
        SEM --> Ctx
        EPI --> Ctx
    end

    subgraph ReActLoop [" ⚙️ Cognitive ReAct Loop "]
        Plan[Strategic Planner]
        Think[Reasoning Engine: Think]
        Act[Tool Dispatcher: Act]
        Observe[Observation & Reflection]
        
        Plan --> Think
        Think -->|Selects Tool| Act
        Act -->|Tool Output| Observe
        Observe -->|Failure / Retry| Think
        Observe -->|Goal Met| FinalAns[Synthesize Final Response]
    end

    subgraph Learning [" 📈 Continuous Learning Loop "]
        Eval[Response Evaluator]
        Replay[(Experience Replay Buffer)]
        Reward[Derived Reward Signal -1.0 to +1.0]
        Feedback[User Rating 1-5 Stars]

        FinalAns --> Eval
        Feedback --> Reward
        Eval --> Reward
        Reward --> Replay
        Replay -.->|Few-Shot Guidance| Plan
    end

    Q --> MemoryRetrieval
    Ctx --> Plan
    FinalAns --> Output[Deliver Output]

    style Q fill:#0284c7,stroke:#38bdf8,stroke-width:2px,color:#fff
    style FinalAns fill:#10b981,stroke:#34d399,stroke-width:2px,color:#fff
    style Replay fill:#6366f1,stroke:#818cf8,stroke-width:2px,color:#fff
    style SEM fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#fff
    style EPI fill:#ec4899,stroke:#f472b6,stroke-width:2px,color:#fff
```

1. **Plan 🗺️**: Decomposes complex multi-step queries into atomic, measurable milestones.
2. **Reason ⚙️**: Selects the optimal tool and inputs based on previous steps and retrieved knowledge.
3. **Act 🛠️**: Dispatches sandboxed tools (Python code execution, DuckDuckGo search, safe calculator, file system).
4. **Observe 👁️**: Parses execution outputs, handles edge cases, and logs observations.
5. **Reflect 🔄**: If an error or HTTP failure occurs, reflects on the cause and tries an alternate strategy.
6. **Learn 📈**: Evaluates success, captures the interaction in the Experience Replay buffer, and updates policies based on user feedback.

---

## 💾 Tiered Memory System

Sentia mirrors biological memory consolidation with four distinct tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SENTIA MEMORY HIERARCHY                         │
├──────────────────┬─────────────────┬─────────────────┬─────────────────┤
│    Short-Term    │    Semantic     │    Long-Term    │    Episodic     │
│   (Context Bed)  │ (Concept Space) │ (Factual Store) │ (Episode Rec)   │
├──────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ • Sliding window │ • Dense vectors │ • Persistent DB │ • Full traces   │
│ • In-memory FIFO │ • all-MiniLM-L6 │ • High-priority │ • Past failures │
│ • Token capped   │ • ChromaDB      │ • SQLite Async  │ • Strategy bank │
└──────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

* ⚡ **Short-Term Memory**: In-process conversational buffer preserving immediate context.
* 🌐 **Semantic Memory**: ChromaDB vector index encoding facts, documentation, and conceptual definitions.
* 📚 **Long-Term Memory**: Relational persistence of curated, high-importance knowledge facts.
* 🎞️ **Episodic Memory**: Structured library of past completed episodes (user query, reasoning trace, tools used, reward outcome).

---

## 🔌 REST API Endpoints

Sentia ships with a production-ready **FastAPI** server:

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/health` | Health & readiness check returning agent status and LLM provider |
| `POST` | `/chat` | Standard chat interface returning formatted answer & latency |
| `POST` | `/chat/detailed` | Deep diagnostic response including full reasoning trace & tool outputs |
| `POST` | `/feedback` | Submits 1–5 star ratings to update the RL experience buffer |
| `GET` | `/conversations` | Lists past conversational sessions with metadata |
| `GET` | `/conversations/{id}` | Retrieves full message history for a conversation |
| `DELETE` | `/conversations/{id}` | Cascades deletion of conversation and all child messages |
| `GET` | `/memories` | Inspects stored persistent memories with optional type filtering |
| `GET` | `/skills` | Lists registered and learned dynamic skills |
| `GET` | `/tools` | Returns JSON Schema specifications for all active tools |

---

## 🛡️ Security & Sandbox

Sentia features an **Immune System** designed to eliminate Remote Code Execution (RCE) hazards:

* 🔒 **Restricted Python Built-ins**: The Python executor blocks dangerous modules (`os`, `sys`, `shutil`, `subprocess`, `socket`) and runs code with restricted globals.
* ⏱️ **Subprocess Isolation & Timeouts**: Script execution runs in sandboxed subprocesses enforced by strict execution timeouts.
* 📁 **Workspace Path Confinement**: The `file_manager` tool resolves paths strictly against `./data/workspace/` and raises `PermissionError` on path-traversal attacks (`../`).
* 🔐 **Transactional ACID Safety**: Relational database updates run with automatic rollback on error and enforced SQLite foreign-key cascading.

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/Viper-07/Self-Learning-AI-Agent.git
cd Self-Learning-AI-Agent

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here        # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional

DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-1.5-pro
DATABASE_URL=sqlite+aiosqlite:///./data/agent.db
CHROMA_DB_PATH=./data/embeddings
```

### 3. Launch Sentia!

#### Mode A: Interactive Terminal CLI 💻
Experience Sentia directly in your console with rich live status animations, step breakdowns, and inline feedback:
```bash
python main.py --cli
```

#### Mode B: FastAPI REST Server 🌐
```bash
python main.py
```
* **Swagger UI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Root Redirect**: [http://localhost:8000](http://localhost:8000) automatically forwards to Swagger Docs.
* **Alternative URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📂 Repository Structure

```text
├── agent/
│   ├── agent.py          # Primary Agent orchestrator & coordinator
│   ├── evaluator.py      # Automated response evaluator & reward scorer
│   ├── executor.py       # ReAct tool dispatcher & execution handler
│   ├── planner.py        # Strategic plan decomposer
│   └── reasoning.py      # ReAct reasoning loop (Think-Act-Observe)
├── database/
│   ├── database.py       # Async SQLAlchemy engine & session factory
│   ├── models.py         # Declarative ORM schemas (7 entities)
│   └── crud.py           # Decoupled domain CRUD operations
├── learning/
│   ├── experience.py     # Experience capture & replay buffer
│   └── feedback.py       # Human feedback processor & policy updater
├── memory/
│   ├── episodic_memory.py# Structured interaction episode persistence
│   ├── long_term.py      # Relational + vector fact memory bridge
│   ├── memory_manager.py # Unified memory facade
│   ├── semantic_memory.py# ChromaDB dense vector store
│   └── short_term.py     # Sliding window conversation memory
├── models/
│   ├── embeddings.py     # Async SentenceTransformer service
│   ├── llm.py            # LiteLLM unified multi-provider client
│   └── schemas.py        # Canonical Pydantic v2 DTO schemas
├── tools/
│   ├── calculator.py     # Safe math evaluation tool
│   ├── file_manager.py   # Sandboxed workspace file manager
│   ├── python_executor.py# Isolated Python execution sandbox
│   ├── tool_manager.py   # Dynamic tool registry & discovery
│   └── web_search.py     # Async DuckDuckGo search & scraper
├── main.py               # Unified application entry point (FastAPI + CLI)
├── config.py             # Type-safe pydantic-settings configuration
└── requirements.txt      # Pinned dependency manifest
```

---

<div align="center">

<!-- Footer Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0284c7,50:1e1b4b,100:0f172a&height=120&section=footer" width="100%" alt="Footer Banner"/>

<p align="center">
  <sub>Built with ❤️, ⚡, and Python for the next generation of Autonomous AI Agents.</sub>
</p>

</div>
