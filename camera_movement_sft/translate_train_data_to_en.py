#!/usr/bin/env python3
"""
把训练数据里的中文 system/user prompt 换成英文(意思一致),assistant 输出原样保留。

单一数据源: 英文 prompt 直接从 common.py import (SYSTEM_PROMPT_EN / USER_PROMPT_EN),
保证训练数据和评测端 (prepare_test_data.py, CAMERA_PROMPT_LANG=en) 用的是逐字节相同的串。

- system 消息: 内容整体替换为 SYSTEM_PROMPT_EN
- user 消息: 保留前缀的媒体占位符(如 <video>), 中文指令替换为 USER_PROMPT_EN
- assistant 消息: 不动(本来就是英文 JSON)

用法:
  python camera_movement_sft/translate_train_data_to_en.py \
      --input  camera_movement_sft/train_data/camera_movement_train_diverse_50k.jsonl \
      --output camera_movement_sft/train_data/camera_movement_train_diverse_50k_en.jsonl
"""
import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from common import (  # noqa: E402
    SYSTEM_PROMPT_ZH, SYSTEM_PROMPT_EN, USER_PROMPT_ZH, USER_PROMPT_EN,
)

CJK = re.compile(r'[一-鿿]')

# video_type 是元数据(不进模型), 顺手英化让文件无中文
VIDEO_TYPE_MAP = {
    "流媒体": "streaming",
    "真人影视": "live_action",
    "动漫": "anime",
    "游戏": "game",
    "unknow": "unknow",
}


def _content_to_str(content):
    """user/system content 可能是 str 或 OpenAI 分段 list; 这里只处理 str 情况。"""
    return content if isinstance(content, str) else None


def translate_user(content: str) -> str:
    """保留首个中文字符之前的媒体占位符前缀 (如 '<video>'), 其余替换为英文指令。"""
    m = CJK.search(content)
    if m is None:
        return content  # 无中文, 不动
    prefix = content[:m.start()]
    return prefix + USER_PROMPT_EN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n = 0
    n_sys_replaced = n_usr_replaced = 0
    n_sys_unexpected = 0
    assistant_cjk = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # video_type 元数据英化 (不进模型, 仅让文件无中文)
            if isinstance(d.get("video_type"), str) and d["video_type"] in VIDEO_TYPE_MAP:
                d["video_type"] = VIDEO_TYPE_MAP[d["video_type"]]
            for msg in d.get("messages", []):
                role = msg.get("role")
                content = _content_to_str(msg.get("content"))
                if content is None:
                    continue
                if role == "system":
                    if content.strip() != SYSTEM_PROMPT_ZH.strip():
                        n_sys_unexpected += 1  # 和已知中文模板不完全一致, 仍整体替换
                    msg["content"] = SYSTEM_PROMPT_EN
                    n_sys_replaced += 1
                elif role == "user":
                    new = translate_user(content)
                    if new != content:
                        n_usr_replaced += 1
                    msg["content"] = new
                elif role == "assistant":
                    if CJK.search(content):
                        assistant_cjk += 1
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1

    print(f"总条数:            {n}")
    print(f"system 替换:       {n_sys_replaced} (其中与已知中文模板不符: {n_sys_unexpected})")
    print(f"user 替换:         {n_usr_replaced}")
    print(f"assistant 含中文:  {assistant_cjk} (>0 需注意, 输出里有中文自由文本)")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
