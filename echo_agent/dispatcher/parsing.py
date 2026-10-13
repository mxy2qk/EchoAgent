from __future__ import annotations

import json
from typing import Literal

from echo_agent.dispatcher.types import ReasonResult, WorkerResult


def parse_worker_result(output: str) -> WorkerResult:
    data = _parse_json_object(output, mode="explore")
    return WorkerResult(
        answer=_optional_answer(data.get("answer")),
        fact_description=_optional_fact_description(data.get("fact_description")),
    )


def parse_reason_result(output: str) -> ReasonResult:
    data = _parse_json_object(output, mode="reason")
    intents = _parse_reason_intents(data)
    return ReasonResult(
        completed=bool(data.get("completed", False)),
        next_intent_descriptions=[str(item["description"]) for item in intents],
        next_intents=intents,
        answer=_optional_answer(data.get("answer")),
        raw_output=output,
    )


def _parse_json_object(output: str, mode: Literal["explore", "reason"]) -> dict[str, object]:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        data = _last_json_object(output)
        if data is None:
            raise ValueError(f"{mode} output does not contain a JSON object")
    if not isinstance(data, dict):
        raise ValueError(f"{mode} output must be a JSON object")
    return data


def _last_json_object(output: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    last_object: dict[str, object] | None = None
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            last_object = value
    return last_object


def _parse_reason_intents(data: dict[str, object]) -> list[dict[str, object]]:
    raw_items = data.get("next_intents")
    if raw_items is None:
        raw_items = data.get("next_intent_descriptions") or []
    if not isinstance(raw_items, list):
        raise ValueError("reason output next_intents must be a list")

    intents: list[dict[str, object]] = []
    for item in raw_items:
        if isinstance(item, str):
            description = item.strip()
            if description:
                intents.append({"description": description, "from": None, "to": None})
            continue
        if not isinstance(item, dict):
            raise ValueError("reason output next_intents items must be strings or objects")
        description = _optional_str(item.get("description"))
        if not description:
            raise ValueError("reason output next_intents object description is required")
        intents.append(
            {
                "description": description,
                "from": _optional_int(item.get("from")),
                "to": _optional_int(item.get("to")),
            }
        )
    return intents


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("integer field cannot be boolean")
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_answer(value: object) -> str | None:
    text = _optional_str(value)
    if text in {"真实 flag 字符串", "实际获得的完整 flag", "实际获得的完整 flag；不要照抄本句", "最终答案", "answer"}:
        return None
    return text


def _optional_fact_description(value: object) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    normalized = " ".join(text.split()).lower()
    placeholder_facts = {
        "初始执行得到的新事实",
        "新的事实描述",
        "新事实描述",
        "事实描述",
        "fact description",
        "new fact",
        "new fact description",
        "写入你实际验证得到的具体事实；不要照抄本句",
        "写入你本次 intent 实际验证得到的具体事实；不要照抄本句",
    }
    if normalized in placeholder_facts:
        return None
    return text
