#!/usr/bin/env python3
"""
Camera SFT loss_scale: 给"答案值 token"提权, 结构/模板 token 保持 1.0。

问题: 输出 JSON 里约 87% 是模板/结构 token(键名、括号、逗号、常量字段), 真正涉及答案的
(type/direction/speed 值、start_time/end_time 数字、special_movement 值)只占少数, 平均 CE
被模板 token 稀释。本 loss_scale 把这些"值 token"的 loss 权重提到 W(默认 5), 其余保持 1.0。

机制: swift 的 LossScale 把 assistant 文本切成 (片段, 权重) 列表, swift 逐片段 tokenize 并
按权重加权 CE。这里按字符区间标权重、再切成"连续等权"片段(拼接严格 == 原串, 保证 token 对齐)。

用法:
  加载本插件(--external_plugins .../camera_loss_scale.py) 后, 训练加 --loss_scale camera_value
  权重可调: 环境变量 CAMERA_VALUE_LOSS_W (默认 5.0)

注意: 只改 loss 权重, 不改输出格式/target token; 三种方案(纯SFT / camdistill / vggt-direct)通用。
"""
import os
import re
from typing import List, Tuple

from swift.loss_scale.base import LossScale
from swift.loss_scale.mapping import loss_scale_map


def _weight() -> float:
    return float(os.environ.get("CAMERA_VALUE_LOSS_W", "5.0"))


# 捕获组1 = 需要加权的"值"字符区间(值本身, 含引号)
_VALUE_PATTERNS = [
    re.compile(r'"type"\s*:\s*("(?:[^"\\]|\\.)*")'),
    re.compile(r'"direction"\s*:\s*(null|"(?:[^"\\]|\\.)*")'),
    re.compile(r'"speed"\s*:\s*("(?:[^"\\]|\\.)*")'),
    re.compile(r'"(?:start_time|end_time)"\s*:\s*(-?\d+(?:\.\d+)?)'),
]
_SPECIAL_ARRAY = re.compile(r'"special_movement"\s*:\s*\[([^\]]*)\]')
_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')


class CameraValueLossScale(LossScale):
    """对 camera 运镜 JSON 的答案值 token 提权。"""

    is_binary = False  # 非 0/1 权重

    def get_loss_scale(self, context, **kwargs) -> Tuple[List[str], List[float]]:
        if not isinstance(context, str) or not context:
            return super().get_loss_scale(context, **kwargs)

        W = _weight()
        n = len(context)
        w = [1.0] * n

        # type/direction/speed/时间 的值区间
        for pat in _VALUE_PATTERNS:
            for m in pat.finditer(context):
                s, e = m.span(1)
                for i in range(s, e):
                    w[i] = W
        # special_movement 数组内每个字符串值
        for m in _SPECIAL_ARRAY.finditer(context):
            base = m.start(1)
            for q in _QUOTED.finditer(m.group(1)):
                for i in range(base + q.start(), base + q.end()):
                    w[i] = W

        # 切成"连续等权"片段(拼接严格还原原串)
        segs: List[str] = []
        ws: List[float] = []
        i = 0
        while i < n:
            j = i
            while j + 1 < n and w[j + 1] == w[i]:
                j += 1
            segs.append(context[i:j + 1])
            ws.append(w[i])
            i = j + 1
        return segs, ws


# 自注册: import 本模块即生效
loss_scale_map['camera_value'] = CameraValueLossScale
print("[camera_loss_scale] registered loss_scale 'camera_value' (W=%.1f via CAMERA_VALUE_LOSS_W)" % _weight())
