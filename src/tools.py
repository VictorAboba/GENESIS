"""
Various utility tools for memory management, note-taking, and dynamic tool creation.
"""

import re
from pathlib import Path
import json
from pydantic import BaseModel
from typing import Literal
import importlib.util

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
    "all_available_tools",
    "add_tools_to_context",
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

    This function takes a text string, generates both dense and sparse embeddings using
    state-of-the-art models (Jina Embeddings v3 for dense vectors and BM25 for sparse vectors),
    and stores it in a Qdrant vector database with hybrid search capabilities.

    The memory database automatically creates the collection if it doesn't exist, using
    a hybrid approach that combines dense semantic similarity with sparse keyword matching
    for more accurate and comprehensive retrieval.

    Args:
        text: The text content to be added to memory. Can be any text content that you
              want to store for later retrieval and semantic search.

    Returns:
        A confirmation message containing the unique ID assigned to the stored text entry,
        in the format: "Text added to memory with ID: {point_id}"

    Raises:
        ValueError: If the input text is empty or contains only whitespace
        IOError: If there are errors connecting to the Qdrant database at localhost:6333,
                 or if there are issues during vector database operations
        RuntimeError: For unexpected errors during the memory storage process

    Example:
        >>> add_to_memory("Python is a high-level programming language")
        "Text added to memory with ID: 1"
    """
    try:
        if not text.strip():
            raise ValueError("Text cannot be empty")

        client = QdrantClient(url="http://localhost:6333")

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

    This function performs advanced semantic search by combining dense vector similarity
    with sparse keyword matching. It uses the Reciprocal Rank Fusion (RRF) algorithm
    to balance results from both Jina Embeddings v3 (dense) and BM25 (sparse) models,
    providing more comprehensive and accurate retrieval than either method alone.

    The search prefetches more candidates (5x top_k) from both dense and sparse vectors,
    then applies RRF fusion to select the final top_k most relevant results.

    Args:
        query: The search query text to find relevant entries in memory.
               Can be a question, keyword, or any text you want to match against
               stored entries based on semantic similarity and keyword relevance.
        top_k: The maximum number of top relevant entries to retrieve (default: 5).
               Must be a positive integer. Higher values return more results but
               may include less relevant entries.

    Returns:
        A formatted string containing the retrieved entries with relevance scores,
        in the format:
        "RAG score: X.XXXX\nText: [stored text]"
        Entries are separated by "---" lines. Returns "No relevant entries found"
        if no matches are found, or "No entries found in memory" if the database
        is empty or doesn't exist.

    Raises:
        ValueError: If the query is empty or contains only whitespace, or if top_k
                   is not a positive integer
        IOError: If there are errors connecting to the Qdrant database at localhost:6333,
                 or if there are issues during the search operation
        RuntimeError: For unexpected errors during the memory retrieval process

    Example:
        >>> retrieve_from_memory("programming languages", top_k=3)
        "RAG score: 0.8756\nText: Python is a high-level programming language\n---\nRAG score: 0.7234\nText: JavaScript is widely used for web development"
    """
    try:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        client = QdrantClient(url="http://localhost:6333")

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

    This function creates a new note and saves it to the notes.json file.
    Notes are automatically cleaned up by removing completed notes when
    creating new ones. The note is assigned a unique ID based on the current
    number of notes.

    Args:
        text: The content of the note as a string
        status: The status of the note from predefined statuses:
                "Need to be done", "Completed", "High priority",
                "Low priority", "Interesting" (default: "Need to be done")

    Returns:
        A confirmation message with the unique ID of the created note

    Raises:
        ValueError: If text is empty or status is invalid
        IOError: If there are file system errors while saving the note
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

    This function loads all notes from the notes.json file and returns them
    as a JSON string. Optionally, you can filter notes by their status to
    retrieve only notes with specific statuses.

    Args:
        status_filtering: Optional status to filter notes by. If provided,
                         only notes with this status will be returned.
                         Available statuses: "Need to be done", "Completed",
                         "High priority", "Low priority", "Interesting".
                         If None, all notes are returned.

    Returns:
        A JSON string containing the requested notes. If no notes match the
        filter, returns an empty JSON object.

    Raises:
        ValueError: If status_filtering is invalid
        IOError: If there are file system errors while reading notes
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

    This function removes a note from the notes.json file based on its ID.
    Once deleted, the note cannot be recovered.

    Args:
        note_id: The ID of the note to delete (format: 'note_<number>')

    Returns:
        A confirmation message indicating the note was successfully deleted

    Raises:
        ValueError: If note_id is invalid or note doesn't exist
        IOError: If there are file system errors while deleting the note
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
    """Update an existing note's text and/or status. Use `read_notes` first to find the needed note_id if you are not sure about it.

    This function allows you to update either the text, status, or both fields
    of an existing note. At least one of new_text or new_status must be provided.

    Args:
        note_id: The ID of the note to update (format: 'note_<number>')
        new_text: Optional new text content for the note. If None, text remains unchanged.
        new_status: Optional new status for the note. If None, status remains unchanged.
                   Available statuses: "Need to be done", "Completed", "High priority",
                   "Low priority", "Interesting"

    Returns:
        A confirmation message showing what was updated

    Raises:
        ValueError: If note_id is invalid, note doesn't exist, or both new_text and new_status are None
        IOError: If there are file system errors while updating the note
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
    """Create a new tool from source code and register it in the tool registry.

    This function takes Python source code and a tool name, creates a new tool file
    in the created_tools directory, and registers it in the tool_registry.json file.
    The function automatically extracts description from the source code's docstring.

    Args:
        source: The Python source code for the tool as a string
        tool_name: The name to assign to the new tool (must be unique)

    Returns:
        A confirmation message indicating the tool was successfully created

    Raises:
        AssertionError: If a tool with the same name already exists
        ValueError: If the source code is empty or tool_name is invalid
        IOError: If there are file system errors during tool creation or registration
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

        # Check if tool already exists
        if path.exists():
            raise AssertionError(f"The tool with name {tool_name} already exists.")

        # Extract description from source code
        description = None
        description_pattern = re.compile(r"\"\"\"(.*?)\"\"\"", re.DOTALL)
        match = re.search(description_pattern, source)
        if match:
            description = match.group(1).strip()

        # Create the tool file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(source)

        # Update tool registry
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
def all_available_tools():
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
def add_tools_to_context(tool_names: list[str]) -> list[BaseTool]:
    """Load and return tool instances based on their names from the tool registry.

    This function reads the tool registry to find the source paths of the specified
    tools, dynamically imports them, and returns a list of tool instances.

    Args:
        tool_names: A list of tool names to load
    Returns:
        A list of BaseTool instances corresponding to the specified tool names
    Raises:
        ValueError: If a tool name is not found in the registry or if there are import errors
    """
    ALL_BASE_TOOLS = [
        create_tool,
        all_available_tools,
        add_tools_to_context,
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
        add_tools_to_context,
        all_available_tools,
        create_note,
        create_tool,
        delete_note,
        read_notes,
        retrieve_from_memory,
        update_note,
    ]
