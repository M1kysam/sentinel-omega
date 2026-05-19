"""
SENTINEL-Ω — FastAPI server with WebSocket live event streaming.

Run:  uvicorn app:app --reload
Open: http://localhost:8000
"""

import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from core.agent import SentinelOmegaAgent
from core.memory import EpisodicMemory

app = FastAPI(title="SENTINEL-Ω")
BASE = Path(__file__).parent

# A shared memory instance keeps state across conversations during the demo
MEMORY = EpisodicMemory()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE / "dashboard.html").read_text()


@app.get("/scenarios")
async def list_scenarios():
    """A curated subset of CHIMERA scenarios for the live demo."""
    # Build a small demo deck if not already present
    demo_file = BASE / "demo_scenarios.json"
    if not demo_file.exists():
        from eval.chimera import generate
        scenarios = generate(n=12, seed=7)
        demo_file.write_text(json.dumps([{
            "id": s.id, "archetype": s.archetype, "alerts": s.alerts,
            "ground_truth": s.ground_truth,
        } for s in scenarios], indent=2))
    return JSONResponse(json.loads(demo_file.read_text()))


@app.websocket("/ws/triage")
async def ws_triage(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)
            alert = payload["alerts"][0] if "alerts" in payload else payload

            async def push(evt):
                await ws.send_text(json.dumps(evt))

            agent = SentinelOmegaAgent(memory=MEMORY, stream_cb=push)
            try:
                await agent.triage(alert)
            except Exception as e:
                await ws.send_text(json.dumps({"type":"error","message":str(e)}))
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
