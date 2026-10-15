from __future__ import annotations

import json
from typing import Literal

from echo_agent.dispatcher.types import ReasonResult, WorkerObservation, WorkerResult


def parse_worker_result(output: str) -> WorkerResult:
    slices = extract_mode_json_outputs(output, "explore")
    if not slices:
        raise ValueError("explore output does not contain a JSON object")
    parsed = [_worker_result_from_data(json.loads(raw)) for raw in slices]
    meaningful = [item for item in parsed if item.fact_description or item.answer or item.outcome]
    if not meaningful:
        return WorkerResult()
    last = meaningful[-1]
    priors = [
        WorkerObservation(
            answer=item.answer,
            fact_description=item.fact_description,
            outcome=item.outcome,
        )
        for item in meaningful[:-1]
        if item.fact_description or item.answer
    ]
    return WorkerResult(
        answer=last.answer,
        fact_description=last.fact_description,
        outcome=last.outcome,
        prior_results=priors,
    )


def _worker_result_from_data(data: dict[str, object]) -> WorkerResult:
    fact_description = _optional_fact_description(data.get("fact_description"))
    answer = _optional_answer(data.get("answer"))
    if answer is None and fact_description is None and "outcome" not in data:
        return WorkerResult()
    return WorkerResult(
        answer=answer,
        fact_description=fact_description,
        outcome=_optional_worker_outcome(data.get("outcome")),
    )


def parse_reason_result(output: str) -> ReasonResult:
    data = _parse_json_object(output, mode="reason")
    completed = data.get("completed", False)
    if not isinstance(completed, bool):
        raise ValueError("reason output completed must be a boolean")
    intents = _parse_reason_intents(data)
    return ReasonResult(
        completed=completed,
        next_intent_descriptions=[str(item["description"]) for item in intents],
        next_intents=intents,
        answer=_optional_answer(data.get("answer")),
        stop_reason=_optional_reason_text(data.get("stop_reason"), "stop_reason"),
        raw_output=output,
    )


def _parse_json_object(output: str, mode: Literal["explore", "reason"]) -> dict[str, object]:
    matches = _matching_json_objects(output, mode)
    if not matches:
        raise ValueError(f"{mode} output does not contain a JSON object")
    data = matches[-1]
    if not isinstance(data, dict):
        raise ValueError(f"{mode} output must be a JSON object")
    return data


def _matching_json_objects(output: str, mode: Literal["explore", "reason"]) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for text in _candidate_texts(output):
        matches.extend(_scan_matching_objects(text, mode))
    return matches


def _candidate_texts(output: str) -> list[str]:
    texts = [output]
    decoder = json.JSONDecoder()
    seen: set[str] = set()
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        for text in _event_texts(value):
            if text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def _event_texts(value: dict[str, object]) -> list[str]:
    texts: list[str] = []
    item = value.get("item")
    containers: list[dict[str, object]] = [value]
    if isinstance(item, dict):
        containers.append(item)
    for container in containers:
        for key in ("text", "message", "content"):
            nested = container.get(key)
            if isinstance(nested, str) and nested.strip():
                texts.append(nested)
    return texts


def _scan_matching_objects(output: str, mode: Literal["explore", "reason"]) -> list[dict[str, object]]:
    decoder = json.JSONDecoder()
    matches: list[dict[str, object]] = []
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and _matches_mode(value, mode):
            matches.append(value)
    return matches


def _matches_mode(value: dict[str, object], mode: Literal["explore", "reason"]) -> bool:
    if mode == "reason":
        return "completed" in value and (
            "next_intents" in value or "next_intent_descriptions" in value
        )
    return "answer" in value or "fact_description" in value


def extract_mode_json_output(output: str, mode: Literal["explore", "reason"]) -> str | None:
    found = extract_mode_json_outputs(output, mode)
    if not found:
        return None
    return found[-1]


def extract_mode_json_outputs(output: str, mode: Literal["explore", "reason"]) -> list[str]:
    found: list[str] = []
    _collect_mode_json_slices(output, mode, seen=set(), found=found)
    return found


def _collect_mode_json_slices(
    output: str,
    mode: Literal["explore", "reason"],
    seen: set[str],
    found: list[str],
) -> None:
    decoder = json.JSONDecoder()
    index = 0
    while index < len(output):
        brace = output.find("{", index)
        if brace < 0:
            return
        try:
            value, end = decoder.raw_decode(output[brace:])
        except json.JSONDecodeError:
            index = brace + 1
            continue
        if not isinstance(value, dict):
            index = brace + 1
            continue
        before = len(found)
        for nested in _event_texts(value):
            if nested in seen:
                continue
            seen.add(nested)
            _collect_mode_json_slices(nested, mode, seen, found)
        if len(found) == before and _matches_mode(value, mode):
            found.append(output[brace : brace + end])
        index = brace + end


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
                intents.append({"description": description, "from": None, "to": None, "action_key": ""})
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
                "action_key": _optional_reason_text(item.get("action_key"), "action_key"),
            }
        )
    return intents


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("integer field must be an integer or null")
    return value


def _optional_reason_text(value: object, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"reason output {field_name} must be a string or null")
    return value.strip()


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


def _optional_worker_outcome(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"verified", "refuted"}:
        return text
    raise ValueError("worker output outcome must be verified or refuted")


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
