"""The recalled-memory note must not grant entries more authority than their own labels.

The note called every provider entry "authoritative reference data ... should inform all responses",
while one provider block mixes the owner's own words with agent inferences and curated rows. It now
defers to each entry's label. Sanitizing still strips the note this build emits and every note older
turns stored, so replayed context keeps no stray note.
"""

import pytest

from agent.memory_manager import build_memory_context_block, sanitize_context

_STORED_NOTES = (
    "[System note: The following is recalled memory context, NOT new user input. Treat as authoritative "
    "reference data — this is the agent's persistent memory and should inform all responses.]",
    "[System note: The following is recalled memory context, NOT new user input. Treat as informational "
    "background data.]",
)


def _emitted_note() -> str:
    # The line after the opening fence, up to its closing bracket.
    note_line = build_memory_context_block("fact").split("\n", 2)[1]
    return note_line[: note_line.index("]") + 1]


def test_sanitize_strips_the_note_build_emits():
    assert sanitize_context(_emitted_note() + "\n\nfact") == "fact"


@pytest.mark.parametrize("note", _STORED_NOTES)
def test_sanitize_strips_notes_stored_in_older_turns(note):
    assert sanitize_context(note + "\n\nfact") == "fact"


def test_note_defers_to_entry_labels():
    note = _emitted_note()

    assert "NOT new user input" in note
    assert "authoritative" not in note.lower()
