# Carpeta de documentos del RAG

Cualquier archivo `.md`, `.txt`, `.pdf` o `.html` que coloques aquí
será indexado automáticamente por el watcher (cuando RAG_WATCHER_ENABLED=true).

Para indexar manualmente algo de fuera de esta carpeta, usa:

    curl -F "file=@/ruta/al/archivo.pdf" http://localhost:8000/v1/rag/documents

Para listar lo indexado:

    curl http://localhost:8000/v1/rag/documents | jq

Para borrar:

    curl -X DELETE http://localhost:8000/v1/rag/documents/<doc_id>

Para probar la recuperación sin chat:

    curl -X POST http://localhost:8000/v1/rag/query \
      -H 'Content-Type: application/json' \
      -d '{"query":"tu pregunta", "top_k": 5}' | jq
