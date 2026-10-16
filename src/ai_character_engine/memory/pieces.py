"""What two texts must share to speak of the same thing."""

from __future__ import annotations

import re

CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+")
# Stands for "some number": an age against an age. Useless for telling topics
# apart, where any two texts with a number would look alike.
NUMBER_PIECE = "#"


def word_pieces(text: str) -> set[str]:
    """Two characters in a row of Chinese, Japanese or Korean, the first four
    letters of a longer word, or any number at all."""
    text = text.casefold()
    pieces = {
        run[index : index + 2]
        for run in CJK_RUN.findall(text)
        for index in range(len(run) - 1)
    }
    pieces.update(word[:4] for word in re.findall(r"[a-z]{4,}", text))
    if re.search(r"\d", text):
        pieces.add(NUMBER_PIECE)
    return pieces
