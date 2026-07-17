#!/bin/bash
# ============================================================================
# 上传 CamInject-4B (vggt_direct) checkpoint 到 HuggingFace Hub
# ============================================================================
# 用法:
#   bash upload_caminject_4b_to_hf.sh
#
# 前置条件:
#   - 当前机器可以连通 huggingface.co（能 ping / curl 通 HF 或代理可用）
#   - 已安装 huggingface_hub (pip install -U huggingface_hub)
# ============================================================================
set -euo pipefail

# ================================
# 基本配置（按需修改）
# ================================
HF_TOKEN="***REMOVED***"
CKPT_DIR="/group/40009/dazhaodu/ms-swift/output/camera_sft_qwen3vl_4b_vggt_direct/v34-20260708-232604/checkpoint-1326"

# 目标仓库名（<username>/<repo_name>）
# 如果你想上传到个人账户，用户名可留空，脚本会通过 whoami 自动读取
HF_USERNAME=""          # 留空则自动检测；也可写死，如 "yourname"
REPO_NAME="CamInject-4B"
PRIVATE="true"          # true=私有仓库，false=公开

# 代理（默认开启）
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# ================================
# 前置检查
# ================================
if [ ! -d "${CKPT_DIR}" ]; then
    echo "错误: checkpoint 目录不存在: ${CKPT_DIR}"
    exit 1
fi

command -v python3 >/dev/null 2>&1 || { echo "错误: python3 未安装"; exit 1; }
python3 -c "import huggingface_hub" 2>/dev/null || {
    echo "未安装 huggingface_hub，正在安装..."
    pip install --user -U "huggingface_hub>=0.24" || pip3 install --user -U "huggingface_hub>=0.24"
}

# ================================
# 调用 Python 完成上传
# ================================
export HF_TOKEN CKPT_DIR HF_USERNAME REPO_NAME PRIVATE

python3 - <<'PYEOF'
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, create_repo, whoami

token    = os.environ["HF_TOKEN"]
ckpt_dir = Path(os.environ["CKPT_DIR"])
username = os.environ.get("HF_USERNAME", "").strip()
repo_nm  = os.environ["REPO_NAME"]
private  = os.environ.get("PRIVATE", "true").lower() == "true"

api = HfApi(token=token)

# 1) 拿到用户名
if not username:
    info = api.whoami()
    username = info["name"]
    print(f"[whoami] 检测到用户名: {username}")

repo_id = f"{username}/{repo_nm}"
print(f"[repo]   目标仓库: {repo_id} (private={private})")

# 2) 创建仓库（存在则忽略）
create_repo(
    repo_id=repo_id, repo_type="model",
    private=private, exist_ok=True, token=token,
)

# 3) 生成一个简单 README
readme_path = ckpt_dir / "README_hf.md"
readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
- camera-movement
- video-understanding
- qwen3-vl
- sft
- vggt-injection
---

# {repo_nm}

Camera-movement VGGT-Direct 注入 SFT 微调模型，基于 `Qwen/Qwen3-VL-4B-Instruct`。

- Checkpoint: `checkpoint-1326`
- 训练框架: [ms-swift](https://github.com/modelscope/ms-swift)

## 使用

```python
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="bfloat16", device_map="auto")
processor = AutoProcessor.from_pretrained("{repo_id}")
```
"""
readme_path.write_text(readme_content, encoding="utf-8")

# 4) 组装上传文件清单（推理必需 + README + 训练元信息）
INCLUDE_FILES = [
    # ---- 推理必需 ----
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    # ---- 模型卡片 ----
    "README_hf.md",
    # ---- 训练元信息（推理不需要，但保留供复现/参考） ----
    "args.json",
    "trainer_state.json",
]
# 所有 model-*.safetensors 分片
INCLUDE_FILES += sorted(p.name for p in ckpt_dir.glob("model-*.safetensors"))

operations = []
for name in INCLUDE_FILES:
    fpath = ckpt_dir / name
    if not fpath.exists():
        print(f"[skip]   {name} 不存在，跳过")
        continue
    # README_hf.md 在仓库里改名为 README.md
    path_in_repo = "README.md" if name == "README_hf.md" else name
    operations.append(CommitOperationAdd(
        path_in_repo=path_in_repo,
        path_or_fileobj=str(fpath),
    ))
    size_mb = fpath.stat().st_size / 1024 / 1024
    print(f"[queue]  {name}  ->  {path_in_repo}  ({size_mb:.1f} MB)")

if not operations:
    print("错误: 没有可上传的文件")
    sys.exit(1)

# 5) 提交（大文件会自动走 LFS）
print(f"\n开始上传 {len(operations)} 个文件到 {repo_id} ...")
commit_info = api.create_commit(
    repo_id=repo_id,
    repo_type="model",
    operations=operations,
    commit_message=f"Upload {repo_nm} (checkpoint-1326)",
    token=token,
)
print(f"\n[done]   提交完成: {commit_info.commit_url}")
print(f"         仓库地址: https://huggingface.co/{repo_id}")
PYEOF

echo "============================================"
echo "上传完成！"
echo "============================================"
