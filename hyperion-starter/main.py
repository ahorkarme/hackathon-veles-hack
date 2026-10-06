import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag import buscar_documentacion


load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
)

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


async def generate_reply(request: ChatRequest):

    # Crear sesión si no existe
    if request.user_id not in sessions:
        sessions[request.user_id] = []

    history = sessions[request.user_id]

    # Buscar información relevante en la documentación
    resultados = buscar_documentacion(request.text, k=3)

    contexto = "\n\n".join(
        [
            f"DOCUMENTO: {resultado['filename']}\n"
            f"{resultado['text']}"
            for resultado in resultados
        ]
    )

    # Añadir la pregunta a la memoria
    history.append({
        "role": "user",
        "content": request.text
    })

    # Instrucciones para el modelo
    system_message = {
        "role": "system",
        "content": f"""
Eres Hyperion, un asistente para el entorno HYPER-AI.

Debes responder utilizando la documentación proporcionada cuando
la pregunta esté relacionada con HYPER-AI.

DOCUMENTACIÓN RECUPERADA:

{contexto}

Si la información necesaria está en la documentación, basa tu respuesta
en ella.

Si la documentación no contiene información suficiente, dilo claramente
y no inventes información.

Mantén el contexto de la conversación utilizando el historial.
"""
    }

    messages = [system_message] + [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in history
    ]

    full_response = ""

    async for chunk in llm.astream(messages):

        if chunk.text:
            full_response += chunk.text

            yield f"data: {json.dumps({'response': chunk.text})}\n\n"

    # Guardar respuesta en memoria
    history.append({
        "role": "assistant",
        "content": full_response
    })

    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_reply(request),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )