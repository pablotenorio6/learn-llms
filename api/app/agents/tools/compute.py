"""Tools puras (sin superficie de ataque): calculadora y fecha/hora.

- calculator: evalúa expresiones aritméticas parseando el AST. NUNCA usa eval():
  solo se permiten nodos y nombres de una allow-list, así que no hay forma de
  llamar a funciones arbitrarias, acceder a atributos ni importar nada.
- datetime_now: fecha/hora actual (UTC + opcionalmente otra timezone IANA).
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timezone

from app.agents.registry import tool

# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

# Operadores binarios y unarios permitidos.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Nombres (constantes y funciones) permitidos. Todo lo demás se rechaza.
_ALLOWED_NAMES: dict[str, object] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "exp": math.exp, "log": math.log, "log2": math.log2,
    "log10": math.log10, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "degrees": math.degrees, "radians": math.radians, "factorial": math.factorial,
    "gcd": math.gcd, "pow": pow, "min": min, "max": max, "sum": sum,
}

# Límite al exponente para no colgar el proceso con 10**10**10.
_MAX_POW = 1_000_000


class _CalcError(Exception):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _CalcError(f"constante no permitida: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise _CalcError(f"operador no permitido: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_POW:
            raise _CalcError("exponente demasiado grande")
        return _BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise _CalcError(f"operador unario no permitido: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise _CalcError(f"nombre no permitido: {node.id!r}")
        return _ALLOWED_NAMES[node.id]  # type: ignore[return-value]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise _CalcError("solo se permiten funciones matemáticas de la allow-list")
        if node.keywords:
            raise _CalcError("argumentos con nombre no permitidos")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise _CalcError(f"expresión no permitida: {type(node).__name__}")


@tool
async def calculator(expression: str) -> dict:
    """Evalúa una expresión matemática y devuelve el resultado numérico exacto. ÚSALA siempre que necesites aritmética no trivial, potencias, raíces, logaritmos, trigonometría o cualquier cálculo donde un error sería grave — no te fíes del cálculo mental para operaciones con varios dígitos. Soporta + - * / // % **, paréntesis, y funciones: sqrt, exp, log, log2, log10, sin, cos, tan, asin, acos, atan, floor, ceil, abs, round, factorial, gcd, min, max, sum, y las constantes pi, e, tau. NO ejecuta código Python arbitrario: solo expresiones matemáticas.

    Args:
        expression: La expresión a evaluar, p.ej. "2**10 + sqrt(144)" o "sin(pi/2)".
    """
    expr = (expression or "").strip()
    if not expr:
        return {"error": "expresión vacía"}
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
    except _CalcError as e:
        return {"error": str(e), "expression": expr}
    except SyntaxError as e:
        return {"error": f"sintaxis inválida: {e.msg}", "expression": expr}
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        return {"error": f"{type(e).__name__}: {e}", "expression": expr}
    return {"expression": expr, "result": result}


# ---------------------------------------------------------------------------
# datetime_now
# ---------------------------------------------------------------------------

@tool
async def datetime_now(tz: str = "UTC") -> dict:
    """Devuelve la fecha y hora ACTUALES. ÚSALA siempre que la pregunta dependa del momento presente ("qué día es hoy", "qué hora es", "cuántos días faltan para X", edades, vencimientos) — tú no sabes la fecha actual por tu cuenta. Devuelve ISO 8601, epoch unix y día de la semana. Por defecto en UTC; puedes pedir otra zona horaria IANA.

    Args:
        tz: Zona horaria IANA, p.ej. "Europe/Madrid" o "America/New_York". Por defecto "UTC".
    """
    now_utc = datetime.now(timezone.utc)
    tzname = (tz or "UTC").strip()
    if tzname.upper() == "UTC":
        dt = now_utc
        resolved = "UTC"
    else:
        try:
            from zoneinfo import ZoneInfo

            dt = now_utc.astimezone(ZoneInfo(tzname))
            resolved = tzname
        except Exception:
            return {"error": f"zona horaria desconocida: {tzname!r}", "hint": "usa un nombre IANA como 'Europe/Madrid'"}
    return {
        "timezone": resolved,
        "iso": dt.isoformat(),
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S"),
        "weekday": dt.strftime("%A"),
        "epoch": int(now_utc.timestamp()),
    }
