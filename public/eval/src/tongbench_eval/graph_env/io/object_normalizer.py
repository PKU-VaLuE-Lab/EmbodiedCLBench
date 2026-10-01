from __future__ import annotations

import re


_ALIAS_MAP = {
    "coffeetable": "coffeetable",
    "coffeetable": "coffeetable",
    "table": "table",
    "remotecontroller": "remotecontrol",
    "remotecontrol": "remotecontrol",
    "remote": "remotecontrol",
    "bathshelf": "bathshelf",
    "tableshelf": "tableshelf",
    "tablelamp": "tablelamp",
    "trashcan": "trashcan",
    "refrigerator": "refrigerator",
    "frigerator": "refrigerator",
    "fridge": "refrigerator",
    "dishwasher": "dishwasher",
    "diswasher": "dishwasher",
    "tv": "tv",
    "television": "tv",
    "bedside": "bedside",
    "cup": "cup",
    "plate": "plate",
    "fruit": "fruit",
    "window": "window",
    "bookshelf": "bookshelf",
    "bookshelves": "bookshelf",
    "ceilinglamp": "ceilinglamp",
    "bookstack": "book",
    "booksstack": "book",
}

_DROP_TOKENS = {
    "bp",
    "kitchen",
    "air",
    "room",
    "living",
    "bedroom",
    "bath",
    "c",
    "mat",
    "scene",
}


def _letters_only(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def normalize_object_type(value: str) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower().replace("-", "_").replace(" ", "_")
    lowered = re.sub(r"^bp_", "", lowered)
    tokens = [token for token in re.split(r"[_\W]+", lowered) if token]
    filtered = [token for token in tokens if token not in _DROP_TOKENS and not token.isdigit()]
    joined = "".join(filtered)
    letter_view = _letters_only(raw)
    candidates = [joined, letter_view, lowered.replace("_", "")]
    for candidate in candidates:
        if not candidate:
            continue
        if "banana" in candidate:
            return "banana"
        if "tomato" in candidate:
            return "tomato"
        if "eggplant" in candidate:
            return "eggplant"
        if "doughnut" in candidate:
            return "doughnut"
        if "bread" in candidate:
            return "bread"
        if "fruit" in candidate or "apple" in candidate:
            return "fruit"
        if "coffeetable" in candidate:
            return "coffeetable"
        if "aircondition" in candidate:
            return "aircondition"
        if "refrigerator" in candidate or "frigerator" in candidate or "fridge" in candidate:
            return "refrigerator"
        if "dishwasher" in candidate or "diswasher" in candidate:
            return "dishwasher"
        if "tvhanged" in candidate or "television" in candidate or candidate == "tv":
            return "tv"
        if "bedside" in candidate:
            return "bedside"
        if "remotecontroller" in candidate or "remotecontrol" in candidate:
            return "remotecontrol"
        if "bookshelves" in candidate or "bookshelf" in candidate:
            return "bookshelf"
        if "bookstack" in candidate or "booksstack" in candidate:
            return "book"
        if "ceilinglamp" in candidate:
            return "ceilinglamp"
        if "window" in candidate:
            return "window"
        if "bathshelf" in candidate:
            return "bathshelf"
        if "tablelamp" in candidate:
            return "tablelamp"
        if "trashcan" in candidate:
            return "trashcan"
        if "cup" in candidate:
            return "cup"
        if "plate" in candidate:
            return "plate"
    for candidate in candidates:
        if candidate in _ALIAS_MAP:
            return _ALIAS_MAP[candidate]
    return joined or letter_view or "unknown"


def infer_object_type_from_id(object_id: str) -> str:
    return normalize_object_type(object_id)
