import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from scripts import rag_ask


def load_user_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


class ChatRequest(BaseModel):
    message: str
    no_vision: bool = False


def create_app(cfg: dict, index_dir: Path):
    app = FastAPI()
    session_state = {"history": []}

    @app.get("/", response_class=HTMLResponse)
    def root():
        return HTMLResponse(
            """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DWAI Chat</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0b0f14; color: #e6edf3; }
      .wrap { max-width: 900px; margin: 0 auto; padding: 16px; }
      .card { background: #0f1620; border: 1px solid #223041; border-radius: 12px; padding: 14px; }
      #log { height: 70vh; overflow: auto; padding: 10px; border-radius: 10px; background: #0b111a; border: 1px solid #223041; }
      .msg { margin: 10px 0; line-height: 1.35; white-space: pre-wrap; }
      .me { color: #c9d1d9; }
      .bot { color: #e6edf3; }
      .meta { color: #8b949e; font-size: 12px; margin-top: 4px; }
      .row { display: flex; gap: 10px; margin-top: 12px; }
      input[type=text] { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #223041; background: #0b111a; color: #e6edf3; }
      button { padding: 12px 14px; border-radius: 10px; border: 1px solid #223041; background: #1f6feb; color: white; cursor: pointer; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      label { display: inline-flex; align-items: center; gap: 8px; color: #8b949e; font-size: 13px; }
      .top { display:flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="top">
        <div><strong>DWAI Chat</strong> <span class="meta">local RAG + auto vision</span></div>
        <label><input id="noVision" type="checkbox" /> disable vision</label>
      </div>
      <div class="card">
        <div id="log"></div>
        <div class="row">
          <input id="msg" type="text" placeholder="Ask about RT45 parts, oil capacity, etc." />
          <button id="send">Send</button>
        </div>
      </div>
    </div>
    <script>
      const log = document.getElementById('log');
      const input = document.getElementById('msg');
      const btn = document.getElementById('send');
      const noVision = document.getElementById('noVision');

      function add(role, text) {
        const d = document.createElement('div');
        d.className = 'msg ' + (role === 'me' ? 'me' : 'bot');
        d.textContent = (role === 'me' ? 'You: ' : 'DWAI: ') + text;
        log.appendChild(d);
        log.scrollTop = log.scrollHeight;
      }

      async function send() {
        const text = input.value.trim();
        if (!text) return;
        input.value = '';
        btn.disabled = true;
        add('me', text);
        try {
          const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text, no_vision: noVision.checked})
          });
          const data = await res.json();
          add('bot', data.answer || '(no answer)');
          if (data.sources && data.sources.length) {
            add('bot', 'Sources:\\n' + data.sources.map((s, i) => `[${i+1}] ${s}`).join('\\n'));
          }
        } finally {
          btn.disabled = false;
          input.focus();
        }
      }

      btn.addEventListener('click', send);
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
      input.focus();
    </script>
  </body>
</html>
            """.strip()
        )

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        answer, sources = rag_ask.answer_question(
            req.message,
            config=cfg,
            index_dir=index_dir,
            force_vision=False,
            disable_vision=req.no_vision,
        )
        session_state["history"].append({"q": req.message, "a": answer})
        return {"answer": answer, "sources": sources}

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Local web chat UI for DWAI RAG.")
    ap.add_argument("--config", default="scripts/rag_config.json", help="Path to rag_config.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    cfg = load_user_config(args.config)
    index_dir = Path(cfg.get("index_dir", Path("data/rag").resolve()))
    app = create_app(cfg, index_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

