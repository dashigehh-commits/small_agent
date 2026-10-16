from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import Agent

app = FastAPI(title="Minimal Agent")
agent = Agent()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    return await agent.run(request.message, request.session_id)

@app.get("/api/sessions/{session_id}")
async def session_history(session_id: str):
    session = agent.sessions.get(session_id)
    return {"session_id": session_id, "messages": session.messages if session else []}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    agent.sessions.pop(session_id, None)
    return {"deleted": session_id}
