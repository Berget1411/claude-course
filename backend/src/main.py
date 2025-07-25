from fastapi import FastAPI, HTTPException
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            messages=[{"role": "user", "content": chat_message.message}],
            max_tokens=1000
        )
        return {"response": response.content[0].text}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

@app.post("/chat-stream")
async def chat_stream(chat_message: ChatMessage):
    try:
        def event_stream():
            with client.messages.stream(
                model="claude-3-7-sonnet-latest",
                messages=[{"role": "user", "content": chat_message.message}],
                max_tokens=1000
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        # Properly format as Server-Sent Events
                        yield f"data: {json.dumps({'content': text})}\n\n"
                
                # Send done signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")