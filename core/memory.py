"""Chat history helpers: message construction and trimming."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# ---------------------------------------------------------------------------
# Turn / message helpers
# ---------------------------------------------------------------------------

TURNS_DEFAULT = 20


def trim_chat_history(
    chat_history: List[BaseMessage],
    max_turns: int = TURNS_DEFAULT,
) -> List[BaseMessage]:
    """Keep only the last ``max_turns`` conversation turns.

    A turn is one user message + one assistant message, so the memory is
    capped at ``2 * max_turns`` messages.
    """
    max_messages = 2 * max_turns
    if len(chat_history) > max_messages:
        del chat_history[: len(chat_history) - max_messages]
    return chat_history


def append_turn(
    chat_history: List[BaseMessage],
    user_message: str,
    assistant_message: str,
    max_turns: int = TURNS_DEFAULT,
) -> List[BaseMessage]:
    """Append a user/assistant turn and trim the history to ``max_turns``."""
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=assistant_message))
    return trim_chat_history(chat_history, max_turns=max_turns)


def to_message_list(
    chat_history: List[BaseMessage],
    max_turns: Optional[int] = None,
) -> List[BaseMessage]:
    """Convert (and optionally trim) a chat history to a list of messages."""
    if max_turns is not None:
        return trim_chat_history(chat_history, max_turns=max_turns)
    return chat_history