import os
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from collections import deque
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    message: Dict[str, str] = Field(
        ..., description="A dictionary containing 'human' and 'ai' messages."
    )

    @property
    def human(self) -> str:
        """Return the human message."""
        return self.message.get("human", "")

    @property
    def ai(self) -> str:
        """Return the AI message."""
        return self.message.get("ai", "")

    def cleanup(memory: Any):
        memory.clear()

    def __str__(self) -> str:
        """Custom string representation for ChatMessage."""
        return json.dumps({"human": self.human, "ai": self.ai}, ensure_ascii=False)


class BaseMemory:
    def __init__(self, max_size=100):
        """Initialize the memory with a maximum size."""
        self.memory = deque(maxlen=max_size)

    def add(self, message: ChatMessage):
        """Add an item to the memory."""
        self.memory.append(message)

    def peek(self) -> str:
        if self.memory:
            return str(self.memory[-1])
        return None

    def get(self):
        """Retrieve all items in memory."""
        return list(self.memory)

    def get_as_str(self):
        """Retrieve all items in memory."""
        return ";".join(f"{x}" for x in list(self.memory))

    def get_max(self, limit: int):
        """Retrieve all items in memory."""
        if self.memory:
            return list(self.memory)[:limit]
        return None

    def clear(self):
        """Clear all items from memory."""
        self.memory.clear()

    def save(self, model_name: str = None, file_path: str = None):
        """Dump all memory contents to a local JSON file."""

        if not file_path or not os.path.exists(file_path):
            file_path = Path(
                os.path.abspath(os.path.join(os.path.dirname(__file__)))
            ).as_posix()

        if self.memory:
            timestamp = (
                str(datetime.now().replace(microsecond=0))
                .replace(":", "")
                .replace(" ", "")
                .replace("-", "")
                .replace("_", "")
            )
            name = (
                f"chat_{model_name.replace("-","")}_{timestamp}.json"
                if model_name
                else f"chat_{timestamp}.json"
            )
            with open(
                os.path.join(file_path, name), "w", encoding="utf-8"
            ) as json_file:
                json.dump([msg.message for msg in self.memory], json_file, indent=4)

    def __str__(self):
        """String representation of the memory."""
        if self.memory:
            return f"Memory({list(self.memory)})"
        return None
