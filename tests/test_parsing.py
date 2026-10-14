from __future__ import annotations

import unittest

from echo_agent.dispatcher.parsing import parse_reason_result, parse_worker_result


class DispatcherParsingTests(unittest.TestCase):
    def test_reason_output_uses_last_json_object_when_codex_prints_extra_data(self) -> None:
        output = "\n".join(
            [
                "codex",
                '{"completed": false, "next_intent_descriptions": ["old"]}',
                '{"completed": false, "next_intent_descriptions": ["new"]}',
                "tokens used",
                "3,037",
            ]
        )

        result = parse_reason_result(output)

        self.assertFalse(result.completed)
        self.assertEqual(result.next_intent_descriptions, ["new"])

    def test_worker_output_uses_last_json_object_after_progress_lines(self) -> None:
        output = "\n".join(
            [
                "progress: running",
                '{"answer": null, "fact_description": "first"}',
                "tokens used",
                '{"answer": "done", "fact_description": "final"}',
            ]
        )

        result = parse_worker_result(output)

        self.assertEqual(result.answer, "done")
        self.assertEqual(result.fact_description, "final")

    def test_worker_output_ignores_placeholder_fact_description(self) -> None:
        result = parse_worker_result('{"answer": null, "fact_description": "初始执行得到的新事实"}')

        self.assertIsNone(result.answer)
        self.assertIsNone(result.fact_description)

    def test_worker_output_ignores_prompt_template_fact_description(self) -> None:
        result = parse_worker_result(
            '{"answer": null, "fact_description": "写入你实际验证得到的具体事实；不要照抄本句"}'
        )

        self.assertIsNone(result.answer)
        self.assertIsNone(result.fact_description)

    def test_worker_output_ignores_placeholder_answer(self) -> None:
        result = parse_worker_result('{"answer": "真实 flag 字符串", "fact_description": null}')

        self.assertIsNone(result.answer)
        self.assertIsNone(result.fact_description)

    def test_reason_output_accepts_structured_intents(self) -> None:
        result = parse_reason_result(
            '{"completed": false, "next_intents": ['
            '{"from": 2, "to": null, "description": "枚举目录"},'
            '{"from": 2, "to": 1, "description": "尝试利用上传点"}'
            "]}"
        )

        self.assertEqual(result.next_intent_descriptions, ["枚举目录", "尝试利用上传点"])
        self.assertEqual(
            result.next_intents,
            [
                {"from": 2, "to": None, "description": "枚举目录"},
                {"from": 2, "to": 1, "description": "尝试利用上传点"},
            ],
        )


    def test_reason_output_keeps_root_object_when_prefix_contains_nested_intents(self) -> None:
        output = """progress\n{\"completed\": false, \"answer\": null, \"next_intents\": [{\"from\": 2, \"to\": null, \"description\": \"分析上传\"}, {\"from\": 2, \"to\": null, \"description\": \"枚举目录\"}]}\n"""
        result = parse_reason_result(output)
        self.assertFalse(result.completed)
        self.assertEqual(result.next_intent_descriptions, ["分析上传", "枚举目录"])

    def test_reason_output_rejects_string_completed(self) -> None:
        with self.assertRaises(ValueError):
            parse_reason_result('{"completed": "false", "next_intents": [{"description": "枚举"}]}')

    def test_reason_output_rejects_non_integer_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            parse_reason_result('{"completed": false, "next_intents": [{"from": 2, "to": "goal", "description": "枚举"}]}')

    def test_worker_output_ignores_trailing_codex_event(self) -> None:
        import json
        message = json.dumps({"answer": None, "fact_description": "real fact"}, ensure_ascii=False)
        event = {"type": "item.completed", "item": {"type": "agent_message", "text": message}}
        output = "\n".join([
            json.dumps(event, ensure_ascii=False),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}}),
        ])
        result = parse_worker_result(output)
        self.assertEqual(result.fact_description, "real fact")
        self.assertIsNone(result.answer)

    def test_worker_output_reads_refuted_outcome(self) -> None:
        result = parse_worker_result('{"answer": null, "fact_description": "上传点不可利用", "outcome": "refuted"}')
        self.assertEqual(result.outcome, "refuted")
        self.assertEqual(result.fact_description, "上传点不可利用")


    def test_worker_output_without_json_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_worker_result("progress only, no json object")


if __name__ == "__main__":
    unittest.main()
