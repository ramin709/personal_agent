from langchain_core.tools import tool
from pathlib import Path


WORKSPACE = Path("./workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

def safe_path(path: str) -> Path:
    """
    Resolve a user-provided path inside the workspace.
    Prevents path traversal outside the workspace.
    """
    target = (WORKSPACE / path).resolve()

    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path is outside the allowed workspace.")

    return target

@tool
def list_directory(path: str = ".") -> str:
    "List files and directories inside the specified path"

    directory = safe_path(path)

    if not directory.exists():
        return f"Error: Directory does not exist: {path}"

    if not directory.is_dir():
        return f"Error: Not a directory: {path}"
    
    entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

    if not entries:
        return f"Directory '{path}' is empty."

    output = [f"Directory: {path}"]

    for entry in entries:
        if entry.is_dir():
            output.append(f"[Dir]: {entry.name}")
        else:
            output.append(f"[File]: {entry.name}")

    return "\n".join(output)

@tool
def create_directory(path: str) -> str:
    """ Creates a directory inside the specified path
        Parent directories are created automatically if necessary.
    """

    directory = safe_path(path)

    if directory.exists():
        if directory.is_dir():
            return f"Directory already exists: {path}"
        return f"Error: A file already exists at: {path}"
    
    directory.mkdir(parents=True)

    return f"Directory has been created successfully: {path}"

@tool
def read_file(path: str) -> str:
    """
    Read and return the contents of a text file inside the workspace.
    """
    file_path = safe_path(path)

    if not file_path.exists():
        return f"Error: File does not exist: {path}"

    if not file_path.is_file():
        return f"Error: Not a file: {path}"

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: File is not a valid UTF-8 text file: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

    return content

@tool
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """
        Creates a file within the path specified with the content defined as the argument.
        By default, an existing file will not be overwritten.
        Set overwrite=True to intentionally replace an existing file.
    """

    file_path = safe_path(path)

    if file_path.exists():
        if not file_path.is_file():
            return f"Error: A directory already exists at: {path}"

        if not overwrite:
            return (
                f"Error: File already exists: {path}. "
                f"Set overwrite=True to replace it."
            )
        
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_path.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"Error writing file: {e}"

    if overwrite:
        return f"File overwritten successfully: {path}"

    return f"File created successfully: {path}"