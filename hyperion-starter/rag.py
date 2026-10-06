import os

import numpy as np
from openai import OpenAI

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
BASE_URL = "https://legion1.di.uoa.gr/v1"
EMBED_MODEL = "nomic-embed-text"

_client = None
_fragmentos = None
_embeddings = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("API_KEY", ""))
    return _client


def _embed(textos):
    vectores = []
    for i in range(0, len(textos), 16):
        lote = textos[i:i + 16]
        respuesta = _get_client().embeddings.create(model=EMBED_MODEL, input=lote)
        vectores.extend(d.embedding for d in respuesta.data)
    matriz = np.array(vectores, dtype=np.float32)
    matriz /= np.linalg.norm(matriz, axis=1, keepdims=True)
    return matriz


def cargar_documentos():
    documentos = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                documentos.append({"filename": filename, "text": f.read()})
    return documentos


def dividir_texto(texto, tamano=200, solape=40):
    palabras = texto.split()
    fragmentos = []
    paso = tamano - solape
    for i in range(0, len(palabras), paso):
        trozo = palabras[i:i + tamano]
        if trozo:
            fragmentos.append(" ".join(trozo))
        if i + tamano >= len(palabras):
            break
    return fragmentos


def _construir_indice():
    global _fragmentos, _embeddings
    fragmentos = []
    for documento in cargar_documentos():
        for parte in dividir_texto(documento["text"]):
            fragmentos.append({"filename": documento["filename"], "text": parte})
    _embeddings = _embed([f["text"] for f in fragmentos])
    _fragmentos = fragmentos


def buscar_documentacion(pregunta, k=3):
    if _fragmentos is None:
        _construir_indice()

    consulta = _embed([pregunta])[0]
    similitudes = np.dot(_embeddings, consulta)
    indices = np.argsort(similitudes)[::-1][:k]

    return [
        {
            "filename": _fragmentos[i]["filename"],
            "text": _fragmentos[i]["text"],
            "score": float(similitudes[i]),
        }
        for i in indices
    ]