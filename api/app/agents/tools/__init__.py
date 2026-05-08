"""Tools registradas. Importar este módulo registra todas en el registry."""

# El simple acto de importar registra las tools (vía decorador @tool)
from app.agents.tools import rag_search  # noqa: F401
