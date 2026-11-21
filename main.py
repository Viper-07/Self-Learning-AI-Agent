"""
main.py — Main application entry point for Sentia Cognitive Agent.

Features:
  1. Interactive Web Playground UI served directly at http://localhost:8000
  2. Production FastAPI REST Server with endpoints:
     - POST /chat (supports standard JSON & SSE token streaming)
     - POST /chat/stream (dedicated Server-Sent Events endpoint)
     - POST /chat/detailed (full reasoning trace & tool outputs)
     - POST /feedback (RL feedback loop with 1-5 star ratings)
     - GET  /health (system status & readiness probe)
     - GET  /stats (live memory, buffer & tools telemetry)
     - GET  /conversations & GET /conversations/{id}
     - DELETE /conversations/{id}
     - GET  /memories & POST /memories (inject factual knowledge)
     - GET  /skills & POST /skills (dynamic skill registry)
     - GET  /tools & POST /tools/execute (test tools directly)
  3. Interactive Terminal CLI Shell via: python main.py --cli
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import UUID

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from agent.agent import Agent
from config import Settings, get_settings
from database.crud import (
    delete_conversation,
    get_all_skills,
    get_conversation,
    get_messages,
    get_recent_conversations,
    get_recent_memories,
    save_skill,
)
from database.database import close_db, get_db, get_db_session, init_db
from learning.experience import ExperienceCollector
from learning.feedback import FeedbackProcessor
from memory.memory_manager import MemoryManager
from models.embeddings import EmbeddingService
from models.llm import LLMClient
from models.schemas import (
    AgentResponse,
    ChatRequest,
    ChatResponse,
    Conversation,
    CreateMemoryRequest,
    ExecuteToolRequest,
    FeedbackRequest,
    HealthResponse,
    MemoryItem,
    MemoryType,
    Message,
    Skill,
    ToolDefinition,
    ToolInput,
    ToolOutput,
)
from tools.tool_manager import ToolManager

logger = structlog.get_logger(__name__)
_SERVER_START_TIME = time.time()


def setup_agent_services(settings: Settings):
    """Wire and initialize all agent components and dependencies."""
    llm_client = LLMClient(settings)
    embedding_service = EmbeddingService(settings.embedding_model)

    tool_manager = ToolManager()
    tool_manager.register_builtin_tools()

    memory_manager = MemoryManager.create(
        embedding_service=embedding_service,
        db_session_factory=get_db_session,
    )

    experience_collector = ExperienceCollector(buffer_size=1000)
    feedback_processor = FeedbackProcessor(
        experience_collector=experience_collector,
        memory_manager=memory_manager,
    )

    agent = Agent(
        llm_client=llm_client,
        tool_manager=tool_manager,
        memory_manager=memory_manager,
        experience_collector=experience_collector,
        name=settings.agent_name,
        max_iterations=settings.max_iterations,
    )

    return {
        "llm_client": llm_client,
        "embedding_service": embedding_service,
        "tool_manager": tool_manager,
        "memory_manager": memory_manager,
        "experience_collector": experience_collector,
        "feedback_processor": feedback_processor,
        "agent": agent,
    }


# ─── FastAPI Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle manager."""
    global _SERVER_START_TIME
    _SERVER_START_TIME = time.time()
    settings = get_settings()
    logger.info("app.starting", agent_name=settings.agent_name)

    # 1. Initialize DB and create tables
    await init_db(settings)

    # 2. Build agent services and attach to app state
    services = setup_agent_services(settings)
    app.state.agent = services["agent"]
    app.state.tool_manager = services["tool_manager"]
    app.state.memory_manager = services["memory_manager"]
    app.state.experience_collector = services["experience_collector"]
    app.state.feedback_processor = services["feedback_processor"]

    logger.info(
        "app.ready",
        agent=settings.agent_name,
        tools=services["tool_manager"].list_names(),
    )
    yield

    # 3. Graceful shutdown
    logger.info("app.shutting_down")
    await close_db()
    logger.info("app.stopped")


# ─── Web Playground UI HTML ────────────────────────────────────────────────────

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sentia — Cognitive AI Agent</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #000000;
      --card-bg: #111111;
      --sidebar: #0d0d0d;
      --border: #222222;
      --accent: #ef4444;
      --accent-glow: rgba(239, 68, 68, 0.25);
      --accent-hover: #dc2626;
      --text: #ececec;
      --text-dim: #8e8e8e;
      --user-msg: #212121;
      --agent-msg: #111111;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-track {
      background: #000000;
    }
    ::-webkit-scrollbar-thumb {
      background: #262626;
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #3f3f46;
    }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    /* Sidebar */
    #sidebar {
      width: 280px;
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 18px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 24px;
    }
    .brand-icon {
      font-size: 24px;
      background: rgba(239, 68, 68, 0.12);
      padding: 8px;
      border-radius: 12px;
      border: 1px solid rgba(239, 68, 68, 0.35);
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .brand-title {
      font-weight: 700;
      font-size: 18px;
      color: #ffffff;
      letter-spacing: 0.5px;
    }
    .brand-subtitle {
      font-size: 11px;
      color: var(--accent);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.8px;
    }
    .new-chat-btn {
      background: #171717;
      color: #ffffff;
      border: 1px solid #2e2e2e;
      padding: 12px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      margin-bottom: 20px;
      transition: all 0.2s;
    }
    .new-chat-btn:hover {
      background: #222222;
      border-color: var(--accent);
      color: #ffffff;
      transform: translateY(-1px);
    }
    .history-title {
      font-size: 12px;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-bottom: 10px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }
    #conv-list {
      flex: 1;
      overflow-y: auto;
      list-style: none;
    }
    .conv-item {
      padding: 10px 12px;
      border-radius: 8px;
      cursor: pointer;
      color: var(--text-dim);
      font-size: 13px;
      margin-bottom: 6px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      transition: all 0.15s;
      border-left: 3px solid transparent;
    }
    .conv-item:hover {
      background: #171717;
      color: var(--text);
    }
    .conv-item.active {
      background: #1c1c1c;
      color: #ffffff;
      border-left-color: var(--accent);
    }
    .sidebar-links {
      border-top: 1px solid var(--border);
      padding-top: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 12px;
    }
    .sidebar-links a {
      color: var(--text-dim);
      text-decoration: none;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: color 0.15s;
    }
    .sidebar-links a:hover { color: var(--accent); }
    /* Main Chat Container */
    #main-chat {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: var(--bg);
    }
    header {
      height: 60px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      background: rgba(0, 0, 0, 0.85);
      backdrop-filter: blur(8px);
    }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      background: #180808;
      border: 1px solid #4a1515;
      color: #fca5a5;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
    }
    .dot {
      width: 7px;
      height: 7px;
      background: var(--accent);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent);
    }
    #chat-box {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .msg {
      max-width: 82%;
      padding: 16px 20px;
      border-radius: 16px;
      line-height: 1.6;
      font-size: 14.5px;
      animation: fadeIn 0.25s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .msg.user {
      align-self: flex-end;
      background: var(--user-msg);
      border: 1px solid #2e2e2e;
      color: #ffffff;
    }
    .msg.agent {
      align-self: flex-start;
      background: var(--agent-msg);
      border: 1px solid var(--border);
      box-shadow: 0 4px 24px rgba(0,0,0,0.45);
      color: var(--text);
    }
    .msg-header {
      font-size: 11px;
      color: var(--accent);
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      letter-spacing: 0.5px;
    }
    /* Red bullets for lists */
    .msg ul, ul.red-bullets {
      list-style-type: none;
      padding-left: 18px;
      margin: 8px 0;
    }
    .msg ul li, ul.red-bullets li {
      position: relative;
      margin-bottom: 6px;
      padding-left: 4px;
    }
    .msg ul li::before, ul.red-bullets li::before {
      content: "•";
      color: var(--accent);
      font-size: 18px;
      font-weight: 800;
      position: absolute;
      left: -14px;
      top: -2px;
    }
    .msg ol {
      padding-left: 20px;
      margin: 8px 0;
    }
    .msg ol li::marker {
      color: var(--accent);
      font-weight: 600;
    }
    .msg pre {
      background: #080808;
      border: 1px solid #222222;
      border-radius: 8px;
      padding: 12px 14px;
      margin: 10px 0;
      overflow-x: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: #f3f4f6;
    }
    .msg .inline-code {
      background: #181818;
      border: 1px solid #2a2a2a;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px;
      color: #fca5a5;
    }
    .typing-dots {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 6px 0;
      min-height: 20px;
    }
    .typing-dots span {
      width: 7px;
      height: 7px;
      background: var(--accent);
      border-radius: 50%;
      display: inline-block;
      animation: dot-pulse 1.4s infinite ease-in-out both;
      box-shadow: 0 0 6px var(--accent-glow);
    }
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    .typing-dots span:nth-child(3) { animation-delay: 0s; }
    @keyframes dot-pulse {
      0%, 80%, 100% {
        transform: scale(0.35);
        opacity: 0.35;
      }
      40% {
        transform: scale(1);
        opacity: 1;
      }
    }
    .tools-accordion {
      margin-top: 12px;
      background: #080808;
      border-radius: 8px;
      border: 1px solid #222222;
      overflow: hidden;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
    }
    .tools-summary {
      padding: 8px 12px;
      cursor: pointer;
      color: var(--text-dim);
      user-select: none;
      transition: color 0.15s;
    }
    .tools-summary:hover {
      color: var(--accent);
    }
    .tools-content {
      padding: 10px 14px;
      border-top: 1px solid #222222;
      color: #a1a1aa;
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 200px;
      overflow-y: auto;
    }
    .rating-row {
      margin-top: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--text-dim);
    }
    .star-btn {
      background: transparent;
      border: none;
      color: #444444;
      font-size: 16px;
      cursor: pointer;
      transition: color 0.15s;
    }
    .star-btn:hover, .star-btn.active { color: #facc15; }
    /* Input Container */
    #input-container {
      padding: 20px 24px;
      border-top: 1px solid var(--border);
      background: rgba(0, 0, 0, 0.95);
    }
    .suggestion-chips {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      overflow-x: auto;
    }
    .chip {
      background: #111111;
      border: 1px solid #262626;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 12px;
      color: var(--text-dim);
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s;
    }
    .chip:hover {
      border-color: var(--accent);
      color: #ffffff;
      background: #1a1a1a;
    }
    .input-box {
      display: flex;
      gap: 12px;
      background: #111111;
      border: 1px solid #2a2a2a;
      border-radius: 12px;
      padding: 6px 8px 6px 16px;
    }
    .input-box:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-glow);
    }
    #user-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text);
      font-family: inherit;
      font-size: 14px;
    }
    #send-btn {
      background: var(--accent);
      color: #ffffff;
      border: none;
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }
    #send-btn:hover {
      background: var(--accent-hover);
      transform: translateY(-1px);
    }
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="brand">
      <div class="brand-icon">🧠</div>
      <div>
        <div class="brand-title">Sentia</div>
        <div class="brand-subtitle">Cognitive Agent</div>
      </div>
    </div>
    <button class="new-chat-btn" onclick="startNewChat()">+ New Conversation</button>
    <div class="history-title">Past Sessions</div>
    <ul id="conv-list"></ul>
    <div class="sidebar-links">
      <a href="/docs" target="_blank">📖 Swagger API Docs</a>
      <a href="/health" target="_blank">🩺 Health & Readiness</a>
      <a href="/stats" target="_blank">📊 System Stats</a>
    </div>
  </div>

  <div id="main-chat">
    <header>
      <div class="status-pill">
        <div class="dot"></div>
        <span>Online • gemini-3-flash-preview</span>
      </div>
      <div style="font-size: 12px; color: var(--text-dim);">
        SQLite + ChromaDB Ready
      </div>
    </header>

    <div id="chat-box">
      <div class="msg agent">
        <div class="msg-header">Sentia Assistant</div>
        Hello! I am <strong>Sentia</strong>, your autonomous cognitive research assistant.<br><br>
        <ul class="red-bullets">
          <li>Autonomous planning, tool execution & calculation</li>
          <li>Live web research & Python code execution</li>
          <li>Self-learning cognitive memory & evaluation</li>
        </ul><br>
        How can I assist your research today?
      </div>
    </div>

    <div id="input-container">
      <div class="suggestion-chips">
        <div class="chip" onclick="useChip('What are the latest breakthroughs in quantum computing?')">🔬 Quantum Computing</div>
        <div class="chip" onclick="useChip('Calculate 15 * 420 + sqrt(144) using your calculator tool')">🧮 Safe Math</div>
        <div class="chip" onclick="useChip('Write a Python function to test if 104729 is a prime number')">🐍 Python Test</div>
      </div>
      <div class="input-box">
        <input type="text" id="user-input" placeholder="Ask Sentia anything or give an autonomous task..." autocomplete="off">
        <button id="send-btn" onclick="sendMessage()">Send</button>
      </div>
    </div>
  </div>

  <script>
    let currentConversationId = null;

    async function loadConversations() {
      try {
        const res = await fetch('/conversations?limit=10');
        if (!res.ok) return;
        const convs = await res.json();
        const list = document.getElementById('conv-list');
        list.innerHTML = '';
        convs.forEach(c => {
          const li = document.createElement('li');
          li.className = 'conv-item' + (c.id === currentConversationId ? ' active' : '');
          li.textContent = 'Session ' + c.id.slice(0, 8);
          li.onclick = () => switchConversation(c.id);
          list.appendChild(li);
        });
      } catch (e) {
        console.error(e);
      }
    }

    function startNewChat() {
      currentConversationId = null;
      document.getElementById('chat-box').innerHTML = `
        <div class="msg agent">
          <div class="msg-header">Sentia Assistant</div>
          New conversation started! Ask me any question or assign a research goal.
        </div>
      `;
      loadConversations();
    }

    async function switchConversation(id) {
      currentConversationId = id;
      try {
        const res = await fetch('/conversations/' + id);
        if (!res.ok) return;
        const data = await res.json();
        const chatBox = document.getElementById('chat-box');
        chatBox.innerHTML = '';
        data.messages.forEach(m => {
          const div = document.createElement('div');
          div.className = 'msg ' + (m.role === 'user' ? 'user' : 'agent');
          div.innerHTML = `<div class="msg-header">${m.role === 'user' ? 'You' : 'Sentia'}</div>` + (m.role === 'user' ? escapeHtml(m.content) : formatMessage(m.content));
          chatBox.appendChild(div);
        });
        chatBox.scrollTop = chatBox.scrollHeight;
        loadConversations();
      } catch (e) {
        console.error(e);
      }
    }

    function useChip(text) {
      document.getElementById('user-input').value = text;
      sendMessage();
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    function formatMessage(text) {
      if (!text) return '';
      let safe = escapeHtml(text);
      safe = safe.replace(/```([\s\S]*?)```/g, (match, code) => `<pre><code>${code.trim()}</code></pre>`);
      safe = safe.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
      safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      const lines = safe.split('\n');
      let inList = false;
      let html = '';
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const bulletMatch = line.match(/^(\s*)[*-]\s+(.+)/);
        if (bulletMatch) {
          if (!inList) {
            html += '<ul class="red-bullets">';
            inList = true;
          }
          html += `<li>${bulletMatch[2]}</li>`;
        } else {
          if (inList) {
            html += '</ul>';
            inList = false;
          }
          html += (i > 0 && html !== '' && !html.endsWith('</ul>') && !html.endsWith('</pre>') ? '<br>' : '') + line;
        }
      }
      if (inList) html += '</ul>';
      return html;
    }

    async function sendMessage() {
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) return;

      input.value = '';
      const chatBox = document.getElementById('chat-box');

      // User Message
      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.innerHTML = `<div class="msg-header">You</div>` + escapeHtml(text);
      chatBox.appendChild(userDiv);

      // Loading Placeholder (3 Animated Red Dots)
      const agentDiv = document.createElement('div');
      agentDiv.className = 'msg agent';
      agentDiv.innerHTML = `<div class="msg-header">Sentia</div><div class="typing-dots"><span></span><span></span><span></span></div>`;
      chatBox.appendChild(agentDiv);
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const res = await fetch('/chat/detailed', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            conversation_id: currentConversationId
          })
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || 'Server error');
        }

        const data = await res.json();
        if (data.metadata && data.metadata.conversation_id) {
          currentConversationId = data.metadata.conversation_id;
          loadConversations();
        }

        let toolsHtml = '';
        if (data.steps && data.steps.length > 0) {
          const toolSteps = data.steps.filter(s => s.thought && s.thought.action);
          if (toolSteps.length > 0) {
            toolsHtml = `<div class="tools-accordion">
              <div class="tools-summary" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'block' ? 'none' : 'block'">
                ⚙️ Executed ${toolSteps.length} reasoning step(s) & tools [Click to expand]
              </div>
              <div class="tools-content" style="display:none;">` +
              toolSteps.map(s => `Step ${s.step_number}: ${s.thought.action}\nInput: ${JSON.stringify(s.thought.action_input)}\nOutput: ${s.tool_output ? (s.tool_output.result || s.tool_output.error) : 'None'}`).join('\n\n') +
              `</div></div>`;
          }
        }

        const starsHtml = `
          <div class="rating-row">
            <span>Rate response:</span>
            <button class="star-btn" onclick="rate('${data.session_id}', '${data.metadata.message_id || data.session_id}', 1, this)">★</button>
            <button class="star-btn" onclick="rate('${data.session_id}', '${data.metadata.message_id || data.session_id}', 2, this)">★</button>
            <button class="star-btn" onclick="rate('${data.session_id}', '${data.metadata.message_id || data.session_id}', 3, this)">★</button>
            <button class="star-btn" onclick="rate('${data.session_id}', '${data.metadata.message_id || data.session_id}', 4, this)">★</button>
            <button class="star-btn" onclick="rate('${data.session_id}', '${data.metadata.message_id || data.session_id}', 5, this)">★</button>
          </div>
        `;

        agentDiv.innerHTML = `<div class="msg-header"><span>Sentia</span><span>${Math.round(data.latency_ms)}ms</span></div>` +
          formatMessage(data.answer) + toolsHtml + starsHtml;

      } catch (err) {
        agentDiv.innerHTML = `<div class="msg-header">Sentia Error</div><span style="color:#ef4444;">Error: ${escapeHtml(err.message)}</span>`;
      }
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function rate(sessionId, messageId, rating, btn) {
      try {
        const res = await fetch('/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            message_id: messageId,
            rating: rating
          })
        });
        if (res.ok) {
          const row = btn.parentElement;
          row.innerHTML = `<span style="color:#ef4444;">✓ Rated ${rating}★ — recorded for continuous learning!</span>`;
        }
      } catch (e) {
        console.error(e);
      }
    }

    document.getElementById('user-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') sendMessage();
    });

    loadConversations();
  </script>
</body>
</html>
"""


# ─── FastAPI Application ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=f"{settings.agent_name} Cognitive Agent REST API",
        description=(
            "Autonomous Cognitive Architecture for Research, Reasoning, and Continuous Learning. "
            "Supports real-time chat, tool execution, knowledge injection, and reinforcement learning."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    cors_origins = (
        settings.cors_origins
        if isinstance(settings.cors_origins, list)
        else [settings.cors_origins]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Interactive Web UI ────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def web_ui():
        """Interactive Web Playground UI."""
        return HTMLResponse(content=_UI_HTML)

    # ── Core System Endpoints ─────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        """Health and readiness probe."""
        return HealthResponse(
            status="ok",
            version="1.0.0",
            agent_name=settings.agent_name,
            llm_provider=settings.default_llm_provider,
        )

    @app.get("/stats", tags=["System"])
    async def system_stats() -> Dict[str, Any]:
        """Live system metrics: memory counts, token estimates, tool registry, and uptime."""
        agent: Agent = app.state.agent
        tool_mgr: ToolManager = app.state.tool_manager
        exp_collector: ExperienceCollector = app.state.experience_collector

        uptime_seconds = round(time.time() - _SERVER_START_TIME, 1)

        return {
            "status": "healthy",
            "agent_name": settings.agent_name,
            "llm_provider": settings.default_llm_provider,
            "llm_model": settings.default_llm_model,
            "uptime_seconds": uptime_seconds,
            "memory": agent.memory_stats,
            "experience_replay_buffer_size": len(exp_collector._buffer),
            "tools_registered": tool_mgr.list_names(),
        }

    # ── Agent Chat Endpoints ──────────────────────────────────────────────────

    @app.post("/chat", response_model=ChatResponse, tags=["Agent"])
    async def chat(request: ChatRequest) -> ChatResponse:
        """Send a message to the agent and receive its response. Supports multi-turn conversation tracking."""
        agent: Agent = app.state.agent
        try:
            resp: AgentResponse = await agent.chat(
                user_message=request.message,
                session_id=request.session_id,
                conversation_id=request.conversation_id,
                llm_config=request.llm_config,
            )
            conv_id = resp.metadata.get("conversation_id")
            msg_id = (
                UUID(resp.metadata["message_id"])
                if "message_id" in resp.metadata
                else None
            )

            return ChatResponse(
                session_id=resp.session_id,
                conversation_id=conv_id,
                message_id=msg_id,
                answer=resp.answer,
                steps_taken=len(resp.steps),
                latency_ms=resp.latency_ms,
                status=resp.status,
            )
        except Exception as exc:
            logger.error("api.chat_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent error: {exc}",
            )

    @app.post("/chat/stream", tags=["Agent"])
    async def chat_stream(request: ChatRequest):
        """Server-Sent Events (SSE) streaming endpoint: streams answer chunks token-by-token."""
        agent: Agent = app.state.agent

        async def event_generator():
            try:
                async for chunk in agent.stream_chat(
                    user_message=request.message,
                    session_id=request.session_id,
                    llm_config=request.llm_config,
                ):
                    yield f"data: {json.dumps({'chunk': chunk})}\\n\\n"
                yield "data: [DONE]\\n\\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\\n\\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.post("/chat/detailed", response_model=AgentResponse, tags=["Agent"])
    async def chat_detailed(request: ChatRequest) -> AgentResponse:
        """Send a message and receive full execution trace, reasoning thoughts, and tool inputs/outputs."""
        agent: Agent = app.state.agent
        try:
            return await agent.chat(
                user_message=request.message,
                session_id=request.session_id,
                conversation_id=request.conversation_id,
                llm_config=request.llm_config,
            )
        except Exception as exc:
            logger.error("api.chat_detailed_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent error: {exc}",
            )

    # ── Continuous Learning & Feedback ────────────────────────────────────────

    @app.post("/feedback", tags=["Learning"])
    async def submit_feedback(request: FeedbackRequest) -> Dict[str, Any]:
        """Submit 1-5 star rating feedback on an agent response to update the RL policy and experience replay."""
        processor: FeedbackProcessor = app.state.feedback_processor
        try:
            reward = await processor.process(request)
            return {
                "status": "ok",
                "derived_reward": round(reward, 3),
                "message": "Feedback recorded for continuous learning.",
            }
        except Exception as exc:
            logger.error("api.feedback_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feedback processing error: {exc}",
            )

    # ── Conversations ─────────────────────────────────────────────────────────

    @app.get("/conversations", response_model=List[Conversation], tags=["Conversations"])
    async def list_conversations(
        limit: int = Query(20, ge=1, le=100),
        db=Depends(get_db),
    ) -> List[Conversation]:
        """List recent conversation sessions."""
        return await get_recent_conversations(db, limit=limit)

    @app.get("/conversations/{conversation_id}", response_model=Conversation, tags=["Conversations"])
    async def get_conversation_by_id(
        conversation_id: str,
        db=Depends(get_db),
    ) -> Conversation:
        """Fetch a specific conversation session and all its messages."""
        conv = await get_conversation(db, conversation_id)
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        return conv

    @app.delete("/conversations/{conversation_id}", tags=["Conversations"])
    async def delete_conversation_by_id(
        conversation_id: str,
        db=Depends(get_db),
    ) -> Dict[str, Any]:
        """Delete a conversation session and cascade delete all child messages."""
        deleted = await delete_conversation(db, conversation_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        return {"status": "ok", "deleted": conversation_id}

    # ── Memory ────────────────────────────────────────────────────────────────

    @app.get("/memories", response_model=List[MemoryItem], tags=["Memory"])
    async def list_memories(
        memory_type: Optional[MemoryType] = None,
        limit: int = Query(20, ge=1, le=100),
        db=Depends(get_db),
    ) -> List[MemoryItem]:
        """List persistent memories stored across ChromaDB and SQLite."""
        return await get_recent_memories(db, memory_type=memory_type, limit=limit)

    @app.post("/memories", tags=["Memory"])
    async def inject_memory(request: CreateMemoryRequest) -> Dict[str, Any]:
        """Inject a factual memory directly into Sentia's long-term vector & relational memory."""
        agent: Agent = app.state.agent
        try:
            mem_id = await agent.memorise(request.content, importance=request.importance)
            return {
                "status": "ok",
                "memory_id": mem_id,
                "message": "Fact successfully injected into memory.",
            }
        except Exception as exc:
            logger.error("api.inject_memory_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to inject memory: {exc}",
            )

    # ── Skills ────────────────────────────────────────────────────────────────

    @app.get("/skills", response_model=List[Skill], tags=["Skills"])
    async def list_skills(db=Depends(get_db)) -> List[Skill]:
        """List registered and active skills."""
        return await get_all_skills(db)

    @app.post("/skills", response_model=Skill, tags=["Skills"])
    async def create_skill(skill: Skill, db=Depends(get_db)) -> Skill:
        """Register a new active skill in the database repository."""
        try:
            await save_skill(db, skill)
            return skill
        except Exception as exc:
            logger.error("api.create_skill_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save skill: {exc}",
            )

    # ── Tools ─────────────────────────────────────────────────────────────────

    @app.get("/tools", response_model=List[ToolDefinition], tags=["Tools"])
    async def list_tools() -> List[ToolDefinition]:
        """List all available tools and their JSON schemas."""
        tool_mgr: ToolManager = app.state.tool_manager
        return tool_mgr.list_definitions()

    @app.post("/tools/execute", response_model=ToolOutput, tags=["Tools"])
    async def execute_tool(request: ExecuteToolRequest) -> ToolOutput:
        """Test and execute a tool directly."""
        tool_mgr: ToolManager = app.state.tool_manager
        output = await tool_mgr.execute(
            ToolInput(tool_name=request.tool_name, parameters=request.parameters)
        )
        return output

    return app


app = create_app()


# ─── Interactive CLI Mode ──────────────────────────────────────────────────────


async def run_cli():
    """Run an interactive console session with the agent."""
    console = Console()
    settings = get_settings()

    console.print(
        Panel(
            f"[bold red]🧠 {settings.agent_name}[/bold red] — Autonomous Cognitive Agent\n"
            f"[dim]Provider: {settings.default_llm_provider} | Model: {settings.default_llm_model}[/dim]\n"
            "[dim]Commands: [bold]:quit[/bold] to exit | [bold]:clear[/bold] to clear session[/dim]",
            title="[bold red]Sentia Interactive Shell[/bold red]",
            border_style="red",
        )
    )

    console.print("[dim]Initializing database and cognitive systems...[/dim]")
    await init_db(settings)
    services = setup_agent_services(settings)
    agent: Agent = services["agent"]
    feedback_proc: FeedbackProcessor = services["feedback_processor"]

    session_id = uuid.uuid4()
    console.print(f"[dim]Ready! Active Session ID: {session_id}[/dim]\n")

    try:
        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            if user_input.lower() in (":quit", ":exit", "quit", "exit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == ":clear":
                session_id = uuid.uuid4()
                console.print(f"[dim]Cleared session. New session ID: {session_id}[/dim]")
                continue

            with console.status("", spinner="point", spinner_style="bold red"):
                response = await agent.chat(user_message=user_input, session_id=session_id)

            # Display reasoning steps if any
            if response.steps:
                for step in response.steps:
                    if step.thought.action:
                        console.print(
                            f"[dim]  ⚙️ Step {step.step_number}: Action [bold]{step.thought.action}[/bold] "
                            f"(Input: {step.thought.action_input})[/dim]"
                        )
                        if step.tool_output:
                            status_emoji = "✅" if step.tool_output.status == "success" else "❌"
                            preview = str(step.tool_output.result or step.tool_output.error)[:120]
                            console.print(f"[dim]     {status_emoji} Result: {preview}...[/dim]")

            # Display answer
            console.print(
                Panel(
                    Markdown(response.answer),
                    title=f"[bold red]{settings.agent_name}[/bold red] [dim]({response.latency_ms:.0f}ms)[/dim]",
                    border_style="red",
                )
            )

            # Optional feedback prompt for learning
            rating_input = Prompt.ask(
                "[dim]Rate this answer (1-5 stars, or Enter to skip)[/dim]",
                default="",
            ).strip()

            if rating_input in ("1", "2", "3", "4", "5"):
                rating = int(rating_input)
                comment = Prompt.ask("[dim]Optional feedback comment[/dim]", default="").strip() or None
                fb_req = FeedbackRequest(
                    session_id=session_id,
                    message_id=uuid.uuid4(),
                    rating=rating,
                    comment=comment,
                )
                reward = await feedback_proc.process(
                    fb_req, original_query=user_input, original_answer=response.answer
                )
                console.print(f"[dim green]✓ Feedback recorded (derived reward: {reward:+.2f}). Agent updated![/dim green]")

    finally:
        console.print("[dim]Shutting down services...[/dim]")
        await close_db()


# ─── Entry Point Dispatcher ────────────────────────────────────────────────────


def main():
    """Main CLI / Server entry point."""
    parser = argparse.ArgumentParser(description="Sentia Self-Learning AI Agent")
    parser.add_argument(
        "--cli",
        "-c",
        action="store_true",
        help="Run interactive terminal CLI shell",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="API host (default from config: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="API port (default from config: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload",
    )

    args = parser.parse_args()
    settings = get_settings()

    if args.cli:
        asyncio.run(run_cli())
    else:
        host = args.host or settings.api_host
        port = args.port or settings.api_port
        console = Console()
        console.print(
            Panel(
                f"[bold red]🚀 {settings.agent_name} Cognitive Agent Server[/bold red]\n\n"
                f"• Web UI Playground:     [bold red]http://localhost:{port}[/bold red]\n"
                f"• Swagger API Docs:      [bold red]http://localhost:{port}/docs[/bold red]\n"
                f"• System Health:         [red]http://localhost:{port}/health[/red]\n"
                f"• Telemetry Stats:       [red]http://localhost:{port}/stats[/red]\n\n"
                f"[dim]Bound to {host}:{port} — Press CTRL+C to stop[/dim]",
                border_style="red",
            )
        )
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
