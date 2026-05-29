"""Tools registradas. Importar este módulo registra todas en el registry."""

# El simple acto de importar registra las tools (vía decorador @tool)
from app.agents.tools import rag_search  # noqa: F401
from app.agents.tools import web  # noqa: F401
from app.agents.tools import compute  # noqa: F401  (calculator, datetime_now)
from app.agents.tools import filesystem  # noqa: F401  (fs_list, fs_read, fs_write)
from app.agents.tools import code  # noqa: F401  (python_exec)
