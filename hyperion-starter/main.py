import json
import os
import re

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
if not API_KEY:
    print("WARNING: API_KEY is not set; LLM calls will fail with 401.", flush=True)
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"

MAX_ROUNDS = 4
MAX_HISTORY = 10
TOOL_NAMES = {t.name for t in TOOLS}

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
    temperature=0,
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

ABOUT HYPER-AI: HYPER-AI is a research project that aims to revolutionise distributed
computing by integrating IoT, Edge and Cloud (the computing continuum). It creates
smart virtual computing nodes and optimises data-processing applications across a
distributed network. The HyperAI IDE is the workspace where users build, validate
and deploy YAML app profiles.

You answer questions about HYPER-AI using the documentation below and the text above,
and you can act in the IDE with your tools (create/delete folders, create/edit/delete
files, read and validate files).

HyperAI, Hyper-AI and HYPER-AI are the same project. Never say you have no information
about HyperAI: the text above and the documentation below are your information.

Rules:
- To act, you MUST call a tool. Never write JSON or tool calls in your answer.
- Never say you did something unless you actually called the tool for it.
- Paths are relative to the workspace root. Never absolute, never with '..'.
- To change an existing file, first read it with read_tool_file, then call edit_file
  with the COMPLETE new content (edit_file replaces the whole file).
- To validate a file, call validate_tool_file and explain the errors and warnings.
- When the user asks for a file, create it with create_file with the full content.
  Follow the YAML examples in the documentation. If there is no example, use
  'apiVersion: hyper.ai/v1' and tell the user to validate the file.
- After using a tool, tell the user briefly what you did.
- If the question is about HYPER-AI and the documentation does not contain the answer,
  say so. Do not invent information.
- Reply in the same language the user writes in.
- Never say a file is valid unless validate_tool_file returned "valid": true in this turn.

DOCUMENTATION:

{contexto}
"""


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"

def normalizar(texto):
    # Los documentos escriben siempre "HYPER-AI"
    return re.sub(r"hyper[\s-]?ai", "HYPER-AI", texto, flags=re.IGNORECASE)

def a_mensajes(history):
    mensajes = []
    for m in history:
        if m["role"] == "user":
            mensajes.append(HumanMessage(content=m["content"]))
        else:
            mensajes.append(AIMessage(content=m["content"]))
    return mensajes


def extraer_llamadas(texto):
    """Find tool calls that the model wrote as plain JSON text."""
    llamadas = []
    decoder = json.JSONDecoder()
    i = 0
    while True:
        i = texto.find("{", i)
        if i == -1:
            break
        try:
            obj, fin = decoder.raw_decode(texto[i:])
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
            args = obj.get("parameters") or obj.get("arguments") or {}
            if isinstance(args, dict):
                llamadas.append(
                    {"name": obj["name"], "args": args, "id": f"text-{len(llamadas)}"}
                )
        i += fin
    return llamadas


async def generate_reply(request: ChatRequest):
    # --- GUARDRAIL SIMPLE ---
    # Convertimos la pregunta a minúsculas para buscar palabras clave
    pregunta = request.text.lower()
    
    # Lista de palabras permitidas (conceptos clave del proyecto y herramientas)
    palabras_clave = ["hyper-ai", "hyperai", "ide", "connector", "connectors", 
                      "deploy", "deployment", "yaml", "archivo", "carpeta", 
                      "explicar", "explicado", "qué", "crea", "borra", "valida"]
    
    # Si la pregunta no contiene NINGUNA de las palabras clave, la rechazamos
    if not any(palabra in pregunta for palabra in palabras_clave):
        yield sse({"response": "Lo siento, como asistente de HYPER-AI solo puedo responder preguntas relacionadas con el proyecto, el IDE o realizar acciones sobre los archivos."})
        yield "data: [DONE]\n\n"
        return
    # ------------------------

    history = sessions.setdefault(request.user_id, [])

    # Los documentos escriben siempre "HYPER-AI"; unificamos la grafía de la pregunta
    pregunta_norm = normalizar(request.text)

    try:
        resultados = buscar_documentacion(pregunta_norm, k=3)
    except Exception as exc:
        # Si falla el RAG, seguimos sin contexto en vez de cortar el stream
        print(f"RAG error: {exc}", flush=True)
        resultados = []
    print([(r["filename"], round(r["score"], 3)) for r in resultados], flush=True)
    contexto = "\n\n".join(
        f"DOCUMENT: {r['filename']}\n{r['text']}" for r in resultados
    )

    # El contexto del RAG va ahora en el mensaje del usuario, no en el system prompt
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    # Construimos la lista de mensajes: System + Historial (sin el actual) + User actual
    messages = [system_msg]
    if history:
         messages += a_mensajes(history[-MAX_HISTORY:])
    
    # Añadimos la pregunta actual, junto con la documentación recuperada
    messages.append(
        HumanMessage(content=f"Documentation:\n{contexto}\n\nQuestion: {pregunta_norm}")
    )

    full_response = ""

    try:
        for _ in range(MAX_ROUNDS):
            final = None
            ronda = ""
            retenido = ""
            reteniendo = False

            async for chunk in llm_tools.astream(messages):
                final = chunk if final is None else final + chunk
                t = chunk.text
                if not t:
                    continue
                ronda += t
                if reteniendo:
                    retenido += t
                elif "{" in t:
                    antes, _, despues = t.partition("{")
                    if antes:
                        full_response += antes
                        yield sse({"response": antes})
                    reteniendo = True
                    retenido = "{" + despues
                else:
                    full_response += t
                    yield sse({"response": t})

            nativas = bool(final is not None and final.tool_calls)
            if nativas:
                llamadas = list(final.tool_calls)
            elif retenido:
                llamadas = extraer_llamadas(ronda)
            else:
                llamadas = []

            if not llamadas:
                if retenido:
                    full_response += retenido
                    yield sse({"response": retenido})
                break

            if nativas:
                messages.append(final)
            else:
                messages.append(AIMessage(content=ronda))

            for call in llamadas:
                evento, resultado = await execute_tool(call["name"], call["args"])
                if evento:
                    yield sse(evento)
                if nativas:
                    messages.append(
                        ToolMessage(
                            content=resultado,
                            tool_call_id=call.get("id") or call["name"],
                        )
                    )
                else:
                    messages.append(
                        HumanMessage(
                            content=f"Result of {call['name']}: {resultado}"
                        )
                    )
    except Exception as exc:
        yield sse({"response": f"\n\n[Error: {exc}]"})

    history.append({"role": "user", "content": request.text})
    history.append({"role": "assistant", "content": full_response})

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