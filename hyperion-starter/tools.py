import json

from langchain_core.tools import tool

from helpers import ReadFileError, ValidateFileError, read_file, validate_file


@tool
def create_folder(path: str) -> str:
    """Create a folder in the IDE workspace. path is relative to the workspace root, e.g. 'demo/sub'."""
    return "ok"


@tool
def delete_folder(path: str) -> str:
    """Delete a folder from the IDE workspace. path is the folder path or just its name."""
    return "ok"


@tool
def create_file(path: str, content: str) -> str:
    """Create a new file in the IDE workspace and open it in the editor.
    path is relative to the workspace root (e.g. 'demo/nginx.yaml'). content is the full text of the file."""
    return "ok"


@tool
def edit_file(path: str, content: str) -> str:
    """Replace the WHOLE content of an existing file and open it in the editor.
    path can be just the file name. content is the complete new text of the file."""
    return "ok"


@tool
def delete_file(path: str) -> str:
    """Delete a file from the IDE workspace. path can be just the file name."""
    return "ok"


@tool
def read_tool_file(path: str) -> str:
    """Read the content of a file in the IDE workspace. path can be just the file name."""
    return "ok"


@tool
def validate_tool_file(path: str) -> str:
    """Validate a YAML file of the IDE workspace and return its errors and warnings. path can be just the file name."""
    return "ok"


TOOLS = [
    create_folder,
    delete_folder,
    create_file,
    edit_file,
    delete_file,
    read_tool_file,
    validate_tool_file,
]

ACTIONS = {
    "create_folder": ["path"],
    "delete_folder": ["path"],
    "create_file": ["path", "content"],
    "edit_file": ["path", "content"],
    "delete_file": ["path"],
}

MAX_RESULT_CHARS = 6000


def _path_ok(path):
    if not isinstance(path, str) or not path.strip():
        return False
    ruta = path.replace("\\", "/")
    return not ruta.startswith("/") and ":" not in ruta and ".." not in ruta.split("/")


async def execute_tool(name, args):
    """Returns (event for the IDE or None, text result for the model)."""
    args = args or {}

    if name in ACTIONS:
        faltan = [k for k in ACTIONS[name] if not isinstance(args.get(k), str)]
        if faltan:
            return None, f"Error: missing arguments {faltan}."
        if not _path_ok(args["path"]):
            return None, "Error: path must be relative to the workspace, never absolute and never contain '..'."
        evento = {"action": name, **{k: args[k] for k in ACTIONS[name]}}
        return evento, f"OK: {name} done for {args['path']}."

    if name == "read_tool_file":
        try:
            contenido = await read_file(args.get("path", ""))
        except ReadFileError as exc:
            return None, f"Error: {exc}"
        return None, contenido[:MAX_RESULT_CHARS]

    if name == "validate_tool_file":
        try:
            informe = await validate_file(args.get("path", ""))
        except ValidateFileError as exc:
            return None, f"Error: {exc}"
        return None, json.dumps(informe)[:MAX_RESULT_CHARS]

    return None, f"Error: unknown tool {name}."