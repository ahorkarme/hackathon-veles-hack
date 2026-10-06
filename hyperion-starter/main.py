import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag import buscar_documentacion
from tools import TOOLS, execute_tool


load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"

MAX_ROUNDS = 4
MAX_HISTORY = 10

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
)
llm_tools = llm.bind_tools(TOOLS)

app = FastAPI(title="Hyperion Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    text: str


# Memoria de las conversaciones
sessions = {}

SYSTEM_PROMPT = """You are Hyperion, the assistant inside the HyperAI IDE.

You answer questions about HYPER-AI using the documentation below, and you can act
in the IDE with the tools you have (create/delete folders, create/edit/delete files,
read and validate files).

Rules:
- Paths are relative to the workspace root. Never absolute, never with '..'.
- To change an existing file, first read it with read_tool_file, then call edit_file
  with the COMPLETE new content (edit_file replaces the whole file).
- When the user asks for a file, create it with create_file and put the full content in it.
- After using a tool, tell the user briefly what you did.
- If the documentation does not contain the answer, say so. Do not invent information.
- Reply in the same language the user writes in.

DOCUMENTATION:

{contexto}
"""


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def a_mensajes(history):
    mensajes = []
    for m in history:
        if m["role"] == "user":
            mensajes.append(HumanMessage(content=m["content"]))
        else:
            mensajes.append(AIMessage(content=m["content"]))
    return mensajes


async def generate_reply(request: ChatRequest):
    history = sessions.setdefault(request.user_id, [])

    resultados = buscar_documentacion(request.text, k=3)
    contexto = "\n\n".join(
        f"DOCUMENT: {r['filename']}\n{r['text']}" for r in resultados
    )

    history.append({"role": "user", "content": request.text})

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(contexto=contexto))]
    messages += a_mensajes(history[-MAX_HISTORY:])

    full_response = ""
    acciones = []

    try:
        for _ in range(MAX_ROUNDS):
            final = None
            async for chunk in llm_tools.astream(messages):
                if chunk.text:
                    full_response += chunk.text
                    yield sse({"response": chunk.text})
                final = chunk if final is None else final + chunk

            if final is None or not final.tool_calls:
                break

            messages.append(final)
            for call in final.tool_calls:
                evento, resultado = await execute_tool(call["name"], call["args"])
                if evento:
                    yield sse(evento)
                    acciones.append(f"{evento['action']} {evento['path']}")
                messages.append(
                    ToolMessage(
                        content=resultado,
                        tool_call_id=call.get("id") or call["name"],
                    )
                )
    except Exception as exc:
        yield sse({"response": f"\n\n[Error: {exc}]"})

    # Guardar respuesta (y las acciones hechas) en memoria
    guardado = full_response
    if acciones:
        guardado += "\n[Actions done: " + ", ".join(acciones) + "]"
    history.append({"role": "assistant", "content": guardado})

    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_reply(request),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)