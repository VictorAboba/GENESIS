"""
Various utility tools for memory management, note-taking, and dynamic tool creation.
"""

import re
from pathlib import Path
import json
from pydantic import BaseModel
from typing import Literal
import importlib.util
import os

from qdrant_client import QdrantClient, models
from langchain_core.tools import tool, BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

__all__ = [
    "add_to_memory",
    "retrieve_from_memory",
    "create_note",
    "read_notes",
    "delete_note",
    "update_note",
    "create_tool",
    "delete_tools",
    "all_available_tools",
    "update_tools_in_context",
    "basetools_to_jsons",
]


def basetools_to_jsons(tools: list[BaseTool]):
    return [
        {"type": "function", "function": convert_to_openai_function(tool_)}
        for tool_ in tools
    ]


#######################
# Working with memory #
#######################
@tool(parse_docstring=True)
def add_to_memory(text: str) -> str:
    """Add a text entry to the vector memory database for semantic search.

    This function stores text using a hybrid search approach. It generates
    dense embeddings via Jina Embeddings v3 and sparse vectors via BM25.
    The system combines semantic similarity with keyword matching for
    accurate retrieval and automatically creates the collection if needed.

    Args:
        text: The text content to be added to memory. This can be any string
            data you want to store for later search and retrieval.

    Returns:
        A confirmation message that includes the unique ID of the stored entry.
    """
    try:
        if not text.strip():
            raise ValueError("Text cannot be empty")

        client = QdrantClient(url="http://qdrant:6333")

        if not client.collection_exists("memory"):
            client.set_model("jinaai/jina-embeddings-v3")
            client.set_sparse_model("Qdrant/bm25")
            client.create_collection(
                collection_name="memory",
                vectors_config={
                    "dense": list(client.get_fastembed_vector_params().values())[0],
                },
                sparse_vectors_config={
                    "sparse": list(
                        client.get_fastembed_sparse_vector_params().values()
                    )[0],
                },
            )
        point_id = client.count("memory").count + 1
        point = models.PointStruct(
            id=point_id,
            vector={
                "dense": models.Document(text=text, model="jinaai/jina-embeddings-v3"),
                "sparse": models.Document(text=text, model="Qdrant/bm25"),
            },
            payload={"text": text},
        )
        client.upsert(collection_name="memory", points=[point], wait=True)
        return f"Text added to memory with ID: {point_id}"
    except Exception as e:
        raise IOError(f"Error adding text to memory: {str(e)}")


@tool(parse_docstring=True)
def retrieve_from_memory(query: str, top_k: int = 5) -> str:
    """Retrieve relevant text entries from the vector memory database using hybrid search.

    Args:
        query: The search query text to find relevant entries in memory. Can be a question,
            keyword, or any text for semantic matching.
        top_k: The maximum number of top relevant entries to retrieve.
            Must be a positive integer.

    Returns:
        A formatted string containing the retrieved entries with relevance scores
        separated by dashes.
    """
    try:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        client = QdrantClient(url="http://qdrant:6333")

        if not client.collection_exists("memory"):
            return "No entries found in memory."

        search_result = client.query_points(
            collection_name="memory",
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query, model="jinaai/jina-embeddings-v3"
                    ),
                    using="dense",
                    limit=top_k * 5,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model="Qdrant/bm25"),
                    using="sparse",
                    limit=top_k * 5,
                ),
            ],
            limit=top_k,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        ).points

        if not search_result:
            return "No relevant entries found in memory."

        return "\n---\n".join(
            [
                f"RAG score: {point.score:.4f}\nText: {point.payload.get('text', 'N/A')}"
                for point in search_result
            ]
        )

    except Exception as e:
        raise IOError(f"Error retrieving text from memory: {str(e)}")


######################
# Working with notes #
######################
STATUSES = Literal[
    "Need to be done", "Completed", "High priority", "Low priority", "Interesting"
]


class Note(BaseModel):
    text: str
    status: STATUSES


def maybe_clear_notes(notes: dict) -> dict:
    if len(notes) >= 100:
        return dict(filter(lambda x: x[1]["status"] != "Completed", notes.items()))

    return notes


def create_unique_id(ids: list[str]) -> str:
    """Generate a unique note ID from a list of existing note IDs.

    This function analyzes existing note IDs in the format 'note_<int>' and finds
    the smallest available integer that hasn't been used. It handles gaps in the
    sequence by finding missing numbers between the minimum and maximum existing IDs.
    If no gaps are found, it returns the next consecutive number after the maximum.

    Args:
        ids: A list of existing note IDs in the format ['note_1', 'note_2', ...]

    Returns:
        A unique note ID string in the format 'note_<number>' where <number> is
        the smallest available positive integer not present in the input list

    Example:
        create_unique_id(['note_1', 'note_2', 'note_4']) returns 'note_3'
        create_unique_id(['note_1', 'note_2', 'note_3']) returns 'note_4'
    """
    if not ids:
        return "note_1"

    integer_parts = list(map(lambda x: int(x.removeprefix("note_")), ids))
    max_ = max(integer_parts)

    used_ids = set(integer_parts)

    for i in range(1, max_ + 2):
        if i not in used_ids:
            return f"note_{i}"

    # This line should never be reached, but just in case
    return f"note_{max_ + 1}"


@tool(parse_docstring=True)
def create_note(text: str, status: STATUSES = "Need to be done") -> str:
    """Create a new note with the given text and status.

    Args:
        text: The content of the note.
        status: The status of the note. Default is 'Need to be done'.

    Returns:
        A confirmation message with the unique ID of the created note.
    """
    try:
        # Validate inputs
        if not text.strip():
            raise ValueError("Note text cannot be empty")
        if status not in STATUSES.__args__:
            raise ValueError(
                f"Invalid status. Must be one of: {', '.join(STATUSES.__args__)}"
            )

        note = Note(text=text, status=status)

        path_to_notes = Path("src/notes.json")
        try:
            with open(path_to_notes, "r", encoding="utf-8") as file:
                notes = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            notes = {}

        notes = maybe_clear_notes(notes)

        note_id = create_unique_id(list(notes.keys()))
        notes[note_id] = {
            "text": note.text,
            "status": note.status,
        }

        path_to_notes.parent.mkdir(parents=True, exist_ok=True)
        with open(path_to_notes, "w", encoding="utf-8") as file:
            json.dump(notes, file, indent=2)

        return f"Note created successfully with ID: {note_id}"

    except (IOError, OSError) as e:
        raise IOError(f"File system error while creating note: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in notes file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while creating note: {str(e)}")


@tool(parse_docstring=True)
def read_notes(status_filtering: STATUSES | None = None) -> str:
    """Read and retrieve notes from the notes.json file.

    Args:
        status_filtering: Optional status to filter notes by. If None, all notes are returned.

    Returns:
        A JSON string containing the requested notes.
    """
    try:
        # Validate status filter
        if status_filtering is not None and status_filtering not in STATUSES.__args__:
            raise ValueError(
                f"Invalid status filter. Must be one of: {', '.join(STATUSES.__args__)}"
            )

        notes_path = Path("src/notes.json")

        try:
            with open(notes_path, "r", encoding="utf-8") as file:
                notes = json.load(file)
        except FileNotFoundError:
            return json.dumps({})
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in notes file")

        if status_filtering:
            filtered_notes = dict(
                filter(lambda x: x[1]["status"] == status_filtering, notes.items())
            )
            return json.dumps(filtered_notes, indent=2)

        return json.dumps(notes, indent=2)

    except (IOError, OSError) as e:
        raise IOError(f"File system error while reading notes: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in notes file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while reading notes: {str(e)}")


@tool(parse_docstring=True)
def delete_note(note_id: str) -> str:
    """Delete a note by its ID.

    Args:
        note_id: The ID of the note to delete.

    Returns:
        A message confirming the note was successfully deleted.
    """
    try:
        if not note_id.strip():
            raise ValueError("Note ID cannot be empty")

        notes_path = Path("src/notes.json")

        try:
            with open(notes_path, "r", encoding="utf-8") as file:
                notes = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            raise ValueError("Notes file not found or contains invalid JSON")

        if note_id not in notes:
            raise ValueError(f"Note with ID '{note_id}' not found")

        deleted_note = notes[note_id]
        deleted_text = deleted_note["text"]
        deleted_status = deleted_note["status"]

        del notes[note_id]

        with open(notes_path, "w", encoding="utf-8") as file:
            json.dump(notes, file, indent=2)

        return f"Note '{note_id}' (text: '{deleted_text}', status: '{deleted_status}') was successfully deleted"

    except (IOError, OSError) as e:
        raise IOError(f"File system error while deleting note: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in notes file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while deleting note: {str(e)}")


@tool(parse_docstring=True)
def update_note(
    note_id: str, new_text: str | None = None, new_status: STATUSES | None = None
) -> str:
    """Update an existing note's text and/or status in the notes file.

    It is recommended to use read_notes first to find the correct note_id.
    You can change only the text, only the status, or both at once.
    At least one update field must be provided.

    Args:
        note_id: The unique ID of the note to update (example - note_1).
        new_text: Optional new text content. Leave None to keep current text.
        new_status: Optional new status. Valid values - Need to be done,
            Completed, High priority, Low priority, Interesting.

    Returns:
        A message showing the successfully updated fields and their new values.

    Raises:
        ValueError: If the ID is not found or the status is not recognized.
    """
    try:
        # Validate inputs
        if not note_id.strip():
            raise ValueError("Note ID cannot be empty")
        if new_text is None and new_status is None:
            raise ValueError("At least one of new_text or new_status must be provided")
        if new_status is not None and new_status not in STATUSES.__args__:
            raise ValueError(
                f"Invalid status. Must be one of: {', '.join(STATUSES.__args__)}"
            )

        notes_path = Path("src/notes.json")

        try:
            with open(notes_path, "r", encoding="utf-8") as file:
                notes = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            raise ValueError("Notes file not found or contains invalid JSON")

        if note_id not in notes:
            raise ValueError(f"Note with ID '{note_id}' not found")

        current_note = notes[note_id]
        old_text = current_note["text"]
        old_status = current_note["status"]

        updated_fields = []
        if new_text is not None:
            if not new_text.strip():
                raise ValueError("New text cannot be empty")
            current_note["text"] = new_text
            updated_fields.append(f"text: '{old_text}' -> '{new_text}'")

        if new_status is not None:
            current_note["status"] = new_status
            updated_fields.append(f"status: '{old_status}' -> '{new_status}'")

        with open(notes_path, "w", encoding="utf-8") as file:
            json.dump(notes, file, indent=2)

        return f"Note '{note_id}' was successfully updated. Changes: {', '.join(updated_fields)}"

    except (IOError, OSError) as e:
        raise IOError(f"File system error while updating note: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in notes file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while updating note: {str(e)}")


######################
# Working with tools #
######################
@tool(parse_docstring=True)
def create_tool(source: str, tool_name: str) -> str:
    """Create a new tool from source code and register it in the system.

    The source code must be a valid Python script including:
    1. Necessary imports (e.g., from langchain_core.tools import tool).
    2. A function decorated with @tool(parse_docstring=True).
    3. A clear Google-style docstring inside the function.
    4. Crucially, the function name in the code must be exactly the same as the tool_name.

    Important for Docstrings:
    Descriptions of variables in the 'Args' section must be extremely simple.
    Avoid using colons (:) or complex formatting inside the description text.
    Use dashes (-) or plain sentences instead to prevent parser errors.

    The tool will be saved as a .py file and tracked in the registry.

    Args:
        source: The complete Python source code as a string.
        tool_name: A unique identifier for the tool. Must match the function name.

    Returns:
        A confirmation message indicating the tool was successfully created.
    """
    try:
        # Validate inputs
        if not source.strip():
            raise ValueError("Source code cannot be empty")
        if not tool_name.strip() or not tool_name.replace("_", "").isalnum():
            raise ValueError(
                "Tool name must be non-empty and contain only alphanumeric characters and underscores"
            )

        path = Path("src/created_tools") / f"{tool_name}.py"

        if path.exists():
            raise AssertionError(f"The tool with name {tool_name} already exists.")

        description = None
        description_pattern = re.compile(r"\"\"\"(.*?)\"\"\"", re.DOTALL)
        match = re.search(description_pattern, source)
        if match:
            description = match.group(1).strip()

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(source)

        path_to_tool_registry = Path("src/tool_registry.json")
        try:
            with open(path_to_tool_registry, "r", encoding="utf-8") as file:
                tools = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            tools = {}

        tools[tool_name] = {
            "source_path": str(path.absolute().resolve()),
            "description": description,
        }

        with open(path_to_tool_registry, "w", encoding="utf-8") as file:
            json.dump(tools, file, indent=2)

        return f"The tool '{tool_name}' was successfully added to your environment!"

    except (IOError, OSError) as e:
        raise IOError(f"File system error while creating tool '{tool_name}': {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool registry: {str(e)}")
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error while creating tool '{tool_name}': {str(e)}"
        )


@tool(parse_docstring=True)
def delete_tools(tool_names: list[str], dry_run: bool = False) -> str:
    """Securely delete tools from the registry and filesystem with atomic operations.

    This function safely removes tools from both the tool registry and their corresponding
    source files. It implements security measures to prevent path traversal attacks,
    validates input, and provides atomic operations with proper rollback mechanisms.

    Args:
        tool_names: List of tool names to delete (must be alphanumeric + underscores)
        dry_run: If True, only validates and reports what would be deleted without actually performing the deletion (default: False)

    Returns:
        A detailed status report showing successful deletions and any errors encountered
    """
    if not tool_names:
        raise ValueError("Tool names list cannot be empty")

    if len(tool_names) > 50:
        raise ValueError("Cannot delete more than 50 tools at once")

    valid_tool_name_pattern = re.compile(r"^[a-zA-Z0-9_]+$")
    for name in tool_names:
        if not isinstance(name, str):
            raise ValueError(f"Tool name must be a string, got {type(name)}")
        if not name.strip():
            raise ValueError("Tool name cannot be empty")
        if not valid_tool_name_pattern.match(name):
            raise ValueError(
                f"Invalid tool name '{name}'. Must contain only alphanumeric characters and underscores"
            )

    path_to_tool_registry = Path("src/tool_registry.json")
    path_to_tool_sources = Path("src/created_tools").resolve()

    try:
        with open(path_to_tool_registry, "r", encoding="utf-8") as file:
            tools = json.load(file)
    except FileNotFoundError:
        tools = {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in tool registry: {str(e)}")
    except (IOError, OSError) as e:
        raise IOError(f"File system error reading tool registry: {str(e)}")

    operations = []
    successful_deletions = []
    failed_operations = []

    try:
        for name in tool_names:
            try:
                if name not in tools:
                    raise ValueError(f"Tool '{name}' not found in registry")

                checkpoint = tools[name].copy()

                source_file_name = f"{name}.py"
                path_to_source_file = (
                    path_to_tool_sources / source_file_name
                ).resolve()

                if not str(path_to_source_file).startswith(
                    str(path_to_tool_sources) + os.sep
                ):
                    raise ValueError(
                        f"Invalid file path for tool '{name}' - potential security violation"
                    )

                file_exists = path_to_source_file.exists()

                if dry_run:
                    operations.append(
                        {
                            "name": name,
                            "status": "dry_run",
                            "message": f"Would delete tool '{name}' and file '{source_file_name}'",
                        }
                    )
                    continue

                del tools[name]

                if file_exists:
                    try:
                        os.remove(path_to_source_file)
                    except (IOError, OSError) as e:
                        tools[name] = checkpoint
                        raise IOError(
                            f"Failed to delete file '{source_file_name}': {str(e)}"
                        )

                operations.append(
                    {
                        "name": name,
                        "status": "success",
                        "message": f"Successfully deleted tool '{name}' and file '{source_file_name}'",
                    }
                )
                successful_deletions.append(name)

            except ValueError as e:
                operations.append(
                    {
                        "name": name,
                        "status": "validation_error",
                        "message": f"Validation error for tool '{name}': {str(e)}",
                    }
                )
                failed_operations.append(name)

            except (IOError, OSError) as e:
                operations.append(
                    {
                        "name": name,
                        "status": "file_error",
                        "message": f"File system error for tool '{name}': {str(e)}",
                    }
                )
                failed_operations.append(name)

            except Exception as e:
                operations.append(
                    {
                        "name": name,
                        "status": "unexpected_error",
                        "message": f"Unexpected error for tool '{name}': {str(e)}",
                    }
                )
                failed_operations.append(name)

        if not dry_run and successful_deletions:
            try:
                with open(path_to_tool_registry, "w", encoding="utf-8") as file:
                    json.dump(tools, file, indent=2)
            except (IOError, OSError) as e:
                raise IOError(f"Failed to update tool registry: {str(e)}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to serialize tool registry: {str(e)}")

        report_lines = []

        if dry_run:
            report_lines.append(f"DRY RUN MODE - Would delete {len(operations)} tools:")
        else:
            report_lines.append(f"Deletion operation completed:")
            report_lines.append(
                f"- Successfully deleted: {len(successful_deletions)} tools"
            )
            report_lines.append(f"- Failed operations: {len(failed_operations)} tools")

        for op in operations:
            status_icon = (
                "✓"
                if op["status"] == "success"
                else "✗" if op["status"] != "dry_run" else "~"
            )
            report_lines.append(f"{status_icon} {op['name']}: {op['message']}")

        return "\n".join(report_lines)

    except Exception as e:
        raise RuntimeError(f"Critical error during tool deletion process: {str(e)}")


@tool(parse_docstring=True)
def all_available_tools_in_registry():
    """List all custom tools currently registered in the system (**not in your context**).

    Returns:
        A formatted string listing all tool names and their descriptions.
    """
    path_to_tool_registry = Path("src/tool_registry.json")
    try:
        with open(path_to_tool_registry, "r", encoding="utf-8") as file:
            tools = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        tools = {}

    if len(tools) == 0:
        return "No tools are currently registered."
    return "\n".join(
        [f"- {name}: {info['description']}" for name, info in tools.items()]
    )


@tool(parse_docstring=True)
def update_tools_in_context(tool_names: list[str]) -> list[BaseTool]:
    """Load tool instances from the registry to expand your current capabilities.

    This function finds the source code for the requested tools, imports them
    dynamically, and returns a combined list containing all core base tools
    and the specific new tools you selected. The resulting list of objects
    is used to update your active context with both your essential functions
    and new specialized abilities.

    Args:
        tool_names: A list of strings with tool names to load. Each name
            must match an entry in the tool registry. Keep descriptions
            simple without colons or complex formatting.

    Returns:
        A list of BaseTool objects including both core base tools and the
        newly selected tools from the registry.
    """
    ALL_BASE_TOOLS = [
        create_tool,
        all_available_tools_in_registry,
        update_tools_in_context,
        add_to_memory,
        retrieve_from_memory,
        create_note,
        read_notes,
        delete_note,
        update_note,
    ]
    path_to_tool_registry = Path("src/tool_registry.json")
    try:
        with open(path_to_tool_registry, "r", encoding="utf-8") as file:
            tools_registry = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        tools_registry = {}

    loaded_tools = []
    for tool_name in tool_names:
        if tool_name not in tools_registry:
            raise ValueError(f"Tool '{tool_name}' not found in the registry.")

        tool_info = tools_registry[tool_name]
        source_path = tool_info["source_path"]
        module_name = Path(source_path).stem

        try:
            spec = importlib.util.spec_from_file_location(module_name, source_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tool_instance = getattr(module, tool_name)
            loaded_tools.append(tool_instance)
        except Exception as e:
            raise ImportError(
                f"Error importing tool '{tool_name}' from '{source_path}': {str(e)}"
            )

    return ALL_BASE_TOOLS + loaded_tools


def load_base_tools() -> list[BaseTool]:
    """Load and return all base tool instances.

    This function returns a list of all predefined base tool instances
    available in the system.

    Returns:
        A list of BaseTool instances representing all base tools.
    """
    return [
        add_to_memory,
        update_tools_in_context,
        all_available_tools_in_registry,
        create_note,
        create_tool,
        delete_note,
        read_notes,
        retrieve_from_memory,
        update_note,
    ]


# TODO: Tool to see tool source code by name and tool to change tool source code by name
