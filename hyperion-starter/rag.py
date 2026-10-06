import os
import numpy as np
from sentence_transformers import SentenceTransformer


DOCS_DIR = "docs"

model = SentenceTransformer("all-MiniLM-L6-v2")


def cargar_documentos():
    documentos = []

    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DOCS_DIR, filename)

            with open(path, "r", encoding="utf-8") as f:
                texto = f.read()

            documentos.append({
                "filename": filename,
                "text": texto
            })

    return documentos


def dividir_texto(texto, tamano=800):
    palabras = texto.split()
    fragmentos = []

    for i in range(0, len(palabras), tamano):
        fragmentos.append(" ".join(palabras[i:i + tamano]))

    return fragmentos


def construir_indice():
    documentos = cargar_documentos()

    fragmentos = []

    for documento in documentos:
        partes = dividir_texto(documento["text"])

        for parte in partes:
            fragmentos.append({
                "filename": documento["filename"],
                "text": parte
            })

    textos = [fragmento["text"] for fragmento in fragmentos]

    embeddings = model.encode(
        textos,
        normalize_embeddings=True
    )

    return fragmentos, embeddings


fragmentos, embeddings = construir_indice()


def buscar_documentacion(pregunta, k=3):
    query_embedding = model.encode(
        [pregunta],
        normalize_embeddings=True
    )[0]

    similitudes = np.dot(embeddings, query_embedding)

    indices = np.argsort(similitudes)[::-1][:k]

    resultados = []

    for indice in indices:
        resultados.append({
            "filename": fragmentos[indice]["filename"],
            "text": fragmentos[indice]["text"],
            "score": float(similitudes[indice])
        })

    return resultados