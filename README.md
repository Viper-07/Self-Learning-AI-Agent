# 🧠 Sentia

<p align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Srijon/Animated-Emojis/main/brain.gif" width="100" alt="Brain Animation">
</p>

<p align="center">
  <strong>A Cognitive Architecture for Autonomous Research, Reasoning, and Continuous Learning</strong>
</p>

---

## 🌟 Overview

This isn't just another LLM wrapper. This is a **Cognitive Agent** designed with a multi-tiered memory system, a strategic planning brain, and a self-correcting execution loop. It doesn't just answer questions—it **plans**, **executes**, **fails**, **reflects**, and **learns** from its own experiences. 🚀

### 🛠️ The Core Architecture

| Component | Role | Tech Stack | Emoji |
| :--- | :--- | :--- | :---: |
| **Strategic Planner** | Decomposes complex goals into actionable steps | LLM + JSON Schema | 🗺️ |
| **Reasoning Engine** | ReAct (Think $\rightarrow$ Act $\rightarrow$ Observe) loop | LLM + Structured Output | ⚙️ |
| **Memory Manager** | Multi-tier context (Short $\rightarrow$ Semantic $\rightarrow$ Episodic) | ChromaDB + SQLite | 💾 |
| **Learning Loop** | Experience Replay & Reward-based Adaptation | Feedback $\rightarrow$ Reward $\rightarrow$ Prompt | 📈 |
| **Tool Sandbox** | Secure execution of Python and Web Search | Subprocesses + Restricted Globals | 🛡️ |

---

## 🧠 How It "Thinks"

The agent operates on a sophisticated cognitive cycle:

1.  **Plan** 🗺️: "I need to research NVIDIA's stock. Step 1: Find current price. Step 2: Find 52-week high. Step 3: Compare."
2.  **Reason** ⚙️: "I'll use the `web_search` tool to find the current price."
3.  **Act** 🛠️: Calls `web_search(query="NVDA current price")`.
4.  **Observe** 👁️: "The price is \$120."
5.  **Reflect** 🔄: *(If the tool failed)* "I got a 403 error. I'll try a different search engine or refine my query."
6.  **Learn** 📈: "The user rated this answer 5 stars. I'll store this successful strategy in my long-term memory."

---

## 💾 Tiered Memory System

The agent manages information like a human brain:

*   **Short-Term Memory** ⚡: A sliding window of the current conversation.
*   **Semantic Memory** 🌐: A vector space of facts (ChromaDB) for conceptual retrieval.
*   **Long-Term Memory** 📚: Persistent, high-importance facts stored in SQLite.
*   **Episodic Memory** 🎞️: A library of past "episodes" (What worked? What failed?).

---

## 🛡️ Security First

The agent is equipped with an **Immune System** to prevent RCE (Remote Code Execution):
- **Restricted Built-ins**: The Python executor only allows a safe subset of Python functions.
- **Subprocess Isolation**: All code runs in isolated processes with strict timeouts.
- **Atomic Writes**: Memory is stored using transactional guards to prevent data corruption.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file:
```env
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
DEFAULT_LLM_PROVIDER=gemini
```

### 3. Run it!
```bash
python main.py
```

---

## 📈 Learning Roadmap

The agent continuously evolves:
- **Human-in-the-loop**: Learns from 1-5 star ratings.
- **Self-Correction**: Uses a Reflection Layer to fix its own bugs.
- **Experience Replay**: Samples past successes to perform few-shot prompting.

<p align="center">
  Built with ❤️ and ⚡ for the future of Autonomous AI.
</p>
# Self-Learning-AI-Agent
