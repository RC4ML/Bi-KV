import json
import os
from typing import List, Callable
from inputGenerator.inputGenerator import InputPrompt, PromptItem

def _prompt_item_to_dict(pi: PromptItem) -> dict:
    return {"item_id": pi.item_id, "token_count": pi.token_count}

def _dict_to_prompt_item(d: dict) -> PromptItem:
    return PromptItem(d["item_id"], d["token_count"])

def _input_prompt_to_dict(ip: InputPrompt) -> dict:
    return {
        "user_id": ip.user_id,
        "user_history_tokens": ip.user_history_tokens,
        "items": [_prompt_item_to_dict(i) for i in ip.items],
        "timestamp": ip.timestamp,
        "weight": ip.weight,
    }

def _dict_to_input_prompt(d: dict) -> InputPrompt:
    """
    把单行 JSON 还原成 InputPrompt。
    如果行里缺少某些字段，就动态计算或给默认值 → 自动恢复。
    """
    # 1⃣ 先用构造函数需要的 5 个必填参数创建对象
    ip = InputPrompt(
        user_id=d["user_id"],
        user_history_tokens=d["user_history_tokens"],
        items=[_dict_to_prompt_item(x) for x in d["items"]],
        timestamp=d["timestamp"],
        weight=d["weight"],
    )

    # 2⃣ 再对可选/派生字段做“存在即恢复、缺失则推导或置默认值”
    # —— order：多数情况下生成阶段再决定，这里恢复可能为 None
    ip.order = d.get("order")                    # 默认 None

    # —— miss_*：如果文件没存，就按 0 恢复
    ip.miss_user_history_tokens = d.get("miss_user_history_tokens", 0)
    ip.miss_item_tokens         = d.get("miss_item_tokens", 0)

    # —— task_id：若缺失则保持 0
    ip.task_id = d.get("task_id", 0)

    # —— item_tokens：可以重新计算，保证一致性
    ip.item_tokens = d.get(
        "item_tokens",
        sum(item.token_count for item in ip.items)   # 动态计算
    )

    return ip

# ---------- 改成按行存储的实现 ----------
def save_prompt_list(prompts: List[InputPrompt], file_path: str) -> None:
    """逐行把 List[InputPrompt] 写成 NDJSON"""
    with open(file_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(_input_prompt_to_dict(p), ensure_ascii=False) + "\n")

def load_prompt_list(file_path: str) -> List[InputPrompt]:
    """按行读取 NDJSON → List[InputPrompt]"""
    prompts: List[InputPrompt] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                prompts.append(_dict_to_input_prompt(json.loads(line)))
    return prompts