from rag import buscar_documentacion


pregunta = "¿Qué son los Open Connectors?"

resultados = buscar_documentacion(pregunta)

for resultado in resultados:
    print("\n---")
    print("Documento:", resultado["filename"])
    print("Similitud:", resultado["score"])
    print(resultado["text"][:1000])