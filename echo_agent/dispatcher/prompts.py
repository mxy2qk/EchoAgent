from __future__ import annotations

import json

from echo_agent.models import FactKind, Intent, IntentOutcome, IntentStatus, Project


class PromptBuilder:
    def build_first_run_prompt(self, project: Project, intent: Intent) -> str:
        return "\n".join(
            [
                "你是一个 CTF/渗透测试题目的执行 agent，当前模式是 first_run。",
                "这是项目第一次执行。你的任务是先尝试直接解决题目并得到 flag。",
                "如果能够得到 flag，返回 answer；如果不能得到 flag，只总结一个对后续探索最有价值的新事实 fact_description。",
                "不要规划后续 intent。后续 intent 由 reason 模式在下一轮循环中规划。",
                "你运行在该项目对应的 Kali Docker 容器中，可以使用容器内工具完成快速验证、扫描、利用和取证。",
                "不要长时间等待无输出命令；如果某个命令持续运行或输出为空，应立即停止等待并把当前可确认的信息总结为 fact_description。",
                "如果已经得到 HTTP 状态、源码片段、目录枚举结果或命令超时信息，应立即输出 JSON 总结事实，不要继续扩大字典或扩展名猜测。",
                "除非已经明确拿到 flag，否则不要因为单个命令卡住而继续等待。",
                "最终只输出一个 JSON 对象，不要输出 Markdown 或额外解释。",
                "",
                "JSON 格式：",
                '{"answer": "实际获得的完整 flag；不要照抄本句", "fact_description": null, "outcome": "verified"}',
                '{"answer": null, "fact_description": "写入你实际验证得到的具体事实；不要照抄本句", "outcome": "verified"}',
                '{"answer": null, "fact_description": "该方向已验证不存在可利用点", "outcome": "refuted"}',
                "",
                _format_project_context(project),
                "",
                "初始 intent：",
                json.dumps(intent.to_dict(), ensure_ascii=False, indent=2),
            ]
        )

    def build_explore_prompt(self, project: Project, intent: Intent) -> str:
        return "\n".join(
            [
                "你是一个 CTF/渗透测试题目的执行 agent，当前模式是 explore。",
                "你的任务是只执行给定 intent，不要规划新的 intent。",
                "你运行在该项目对应的 Kali Docker 容器中，可以使用容器内工具完成验证、扫描、利用和取证。",
                "不要长时间等待无输出命令；如果某个命令持续运行或输出为空，应立即停止等待并把当前可确认的信息总结为 fact_description。",
                "如果已经得到 HTTP 状态、源码片段、目录枚举结果或命令超时信息，应立即输出 JSON 总结事实，不要继续扩大字典或扩展名猜测。",
                "除非已经明确拿到 flag，否则不要因为单个命令卡住而继续等待。",
                "如果直接得到 flag 或最终答案，返回 answer；否则总结本次执行得到的新事实 fact_description。",
                "最终只输出一个 JSON 对象，不要输出 Markdown 或额外解释。",
                "",
                "JSON 格式：",
                '{"answer": null, "fact_description": "写入你本次 intent 实际验证得到的具体事实；不要照抄本句", "outcome": "verified"}',
                '{"answer": null, "fact_description": "该方向已验证不存在可利用点", "outcome": "refuted"}',
                '{"answer": "实际获得的完整 flag；不要照抄本句", "fact_description": null, "outcome": "verified"}',
                "",
                _format_project_context(project),
                "",
                "当前要执行的 intent：",
                json.dumps(intent.to_dict(), ensure_ascii=False, indent=2),
            ]
        )

    def build_reason_prompt(self, project: Project) -> str:
        return "\n".join(
            [
                "你是一个 CTF/渗透测试题目的规划 agent，当前模式是 reason。",
                "你的任务是根据 evidence 判断项目是否已经完成；如果未完成，只从 allowed_from 给出下一批 intent。",
                "不要执行命令，不要进行扫描或利用，只做推理和规划。",
                "最终答案的形式不固定，不要假设一定是 flag{...}。你需要根据项目事实图判断是否已经得到题目要求的最终答案。",
                "只有当 evidence 中已经包含可以提交的最终答案时，才返回 completed=true，并将 answer 写为该最终答案。",
                "生成的 intent description 必须描述接下来需要探索的方向或验证动作，例如“分析是否存在文件上传漏洞”，不要写成笼统的“继续推进”。",
                "生成的 intent 应该具体、可执行，并且能由后续 explore 模式在 Kali Docker 容器中完成。",
                "不是每条 evidence 都需要生成 intent；只选择对达成目标有价值、能继续验证或利用的 evidence 生成 intent。",
                "每个新 intent 的 from 必须取 allowed_from 里的 fact id。evidence 为空时 allowed_from 只有起点，通常是 0。",
                "不能使用 findings、do_not_repeat、already_done、证伪事实、错误事实或 goal 作为 from。",
                "to 可以为 null，系统会先临时连接到目标终点 fact。",
                "每条 intent 必须带稳定的 action_key，使用小写英文和连字符，例如 check-upload。",
                "already_done 中 outcome 为 refuted 或 verified 的动作不能再规划，即使更换 action_key 或换一种说法也不行。",
                "outcome 为 error 或 empty 时，不要重复原来的 description；只有描述明显更小的新步骤才可以规划。",
                "error、empty 的 note，以及“未完成执行的输出摘录”，都不是证伪结论，只能改写成更小的新步骤。",
                "findings 里的证伪正文只用于判断，不能作为 from，也不能换一种说法重做。",
                "允许同一个 evidence 产生多个 intent，但 action_key 不能重复。",
                "只有 answer 已经出现在某条 evidence 中时，才返回 completed=true。",
                "如果没有新的可执行方向，返回 completed=false、next_intents=[]，并用 stop_reason 说明原因。不要为了凑数重复旧动作。",
                "最终只输出一个 JSON 对象，不要输出 Markdown 或额外解释。",
                "",
                "JSON 格式：",
                '{"completed": true, "answer": "实际可提交的最终答案", "stop_reason": "", "next_intents": []}',
                '{"completed": false, "answer": null, "stop_reason": "上传和备份面已经验证关闭", "next_intents": []}',
                '{"completed": false, "answer": null, "stop_reason": "", "next_intents": [{"from": 2, "to": null, "action_key": "check-upload", "description": "分析是否存在文件上传漏洞"}, {"from": 2, "to": null, "action_key": "enum-backup", "description": "枚举 Web 目录和敏感备份文件"}]}',
                "",
                _format_project_context(project),
            ]
        )


def _format_project_context(project: Project) -> str:
    evidence = [
        {"id": fact.id, "description": fact.description}
        for fact in sorted(project.facts.values(), key=lambda item: item.id)
        if fact.kind == FactKind.EVIDENCE
    ]
    executed = [
        intent
        for intent in sorted(project.intents.values(), key=lambda item: item.id)
        if intent.status in {IntentStatus.COMPLETED, IntentStatus.FAILED}
    ]
    already_done = []
    for intent in executed:
        result_text, result_kind = _intent_result(project, intent)
        already_done.append(
            {
                "id": intent.id,
                "from": intent.from_fact_id,
                "action_key": intent.action_key,
                "description": intent.description,
                "outcome": intent.outcome.value,
                "note": intent.note,
                "result": result_text,
                "result_kind": result_kind,
            }
        )
    blocked = [
        item
        for item in already_done
        if item["outcome"] in {IntentOutcome.REFUTED.value, IntentOutcome.ERROR.value, IntentOutcome.EMPTY.value}
    ]
    findings = [
        {"id": fact.id, "kind": fact.kind.value, "description": fact.description}
        for fact in sorted(project.facts.values(), key=lambda item: item.id)
        if fact.kind in {FactKind.REFUTED, FactKind.ERROR}
    ]
    evidence_ids = [item["id"] for item in evidence]
    if evidence_ids:
        allowed_from = evidence_ids
    elif 0 in project.facts:
        allowed_from = [0]
    else:
        allowed_from = []
    payload = {
        "id": project.id,
        "name": project.name,
        "target_url": project.target_url,
        "hint": project.hint,
        "objective": project.objective,
        "goal": project.facts[1].description if 1 in project.facts else "",
        "evidence": evidence,
        "allowed_from": allowed_from,
        "findings": findings,
        "already_done": already_done,
        "do_not_repeat": blocked,
    }
    return "\n".join(
        [
            "项目上下文分为已知证据、证伪结论、已经做过的动作、不要重复的失败路径、终点说明。",
            "evidence 才是已验证事实。新 intent 的 from 只能取 allowed_from。",
            "evidence 为空时，allowed_from 只包含起点，避免没有证据时无法继续规划。",
            "findings 是证伪或错误事实的正文。不能作为 from，也不能换一种说法重做。",
            "already_done.result 是该动作实际得到的结论。",
            "refuted 和 verified 不能更换 action_key 或换一种说法再规划。",
            "error 和 empty 不能原样重复 description；只有更小的新步骤才可以继续。",
            "未完成执行的输出摘录只是超时前的片段，必须改成更小的步骤，不能当成证伪。",
            "do_not_repeat 不能作为新的探索起点。",
            "goal 只说明目标，不是证据。",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def _intent_result(project: Project, intent: Intent) -> tuple[str, str]:
    if intent.to_fact_id is None:
        return "", ""
    fact = project.facts.get(intent.to_fact_id)
    if fact is None:
        return "", ""
    return fact.description, fact.kind.value
