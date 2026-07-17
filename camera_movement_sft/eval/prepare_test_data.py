#!/usr/bin/env python3
"""
测试数据格式转换：标注 JSONL → ms-swift infer 标准 JSONL

输入格式（标注结果）:
    {"video_id": "xxx", "cos_url": "http://...", "segments": [...]}

输出格式（ms-swift 标准）:
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "<video>请分析..."},
                  {"role": "assistant", "content": "<ground_truth_json>"}],
     "videos": ["http://...cos_url..."]}

说明:
    - videos 字段直接使用 cos_url，swift 支持 URL 输入
    - assistant 消息包含 ground truth，swift infer 会自动将其作为 labels
    - 同时生成一个 video_id 映射文件，用于后续结果对齐

使用方法:
    python camera_movement_sft/eval/prepare_test_data.py
    python camera_movement_sft/eval/prepare_test_data.py --use_local_video --video_dir /path/to/videos1 /path/to/videos2
    python camera_movement_sft/eval/prepare_test_data.py --use_local_video --video_dir /path/to/videos1 /path/to/videos2 --auto_download
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# 添加项目目录到 Python 路径，以便导入 common 模块
sys.path.insert(0, PROJECT_DIR)

# 从公共模块导入
from common import SYSTEM_PROMPT, USER_PROMPT

# 测试数据路径
DEFAULT_TEST_DATA = "/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/test/500评测集_人工筛选后_with_split_clips.jsonl"
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_data_swift")


def download_video(url, save_path):
    """从 URL 下载视频到指定路径，返回是否成功"""
    if not url:
        return False

    # 有些评测集的 cos_url 实际是本地绝对路径，直接复用即可。
    if os.path.isabs(url) and os.path.exists(url):
        print(f"[INFO] cos_url 是本地文件，直接使用: {url}")
        return url

    tmp_path = save_path + ".downloading"
    try:
        print(f"[INFO] 正在下载: {url}")
        print(f"       保存到: {save_path}")
        result = subprocess.run(
            ["wget", "-q", "--timeout=60", "--tries=2", "-O", tmp_path, url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=120,
        )
        if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            os.rename(tmp_path, save_path)
            print(f"[INFO] 下载成功: {os.path.basename(save_path)} ({os.path.getsize(save_path) / 1024 / 1024:.1f} MB)")
            return True
        else:
            print(f"[WARN] 下载失败 (wget 返回码 {result.returncode}): {url}")
            if result.stderr:
                print(f"       stderr: {result.stderr.strip()}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    except subprocess.TimeoutExpired:
        print(f"[WARN] 下载超时: {url}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    except Exception as e:
        print(f"[WARN] 下载异常: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def find_local_video(video_id, video_dirs, cos_url=None, auto_download=False):
    """在多个候选目录中查找本地视频文件。

    Args:
        video_id: 视频 ID
        video_dirs: 候选目录列表
        cos_url: COS URL，找不到本地视频时用于下载
        auto_download: 是否自动下载缺失的视频到第一个目录

    Returns:
        找到/下载成功返回路径，否则返回 None
    """
    for d in video_dirs:
        path = os.path.join(d, f"{video_id}.mp4")
        if os.path.exists(path):
            return path

    # cos_url 可能已经是本地文件路径。
    if cos_url and os.path.isabs(cos_url) and os.path.exists(cos_url):
        return cos_url

    # 本地都找不到，尝试下载到第一个目录
    if auto_download and cos_url and video_dirs:
        download_dir = video_dirs[0]
        os.makedirs(download_dir, exist_ok=True)
        save_path = os.path.join(download_dir, f"{video_id}.mp4")
        dl_result = download_video(cos_url, save_path)
        if isinstance(dl_result, str):
            return dl_result
        if dl_result:
            return save_path

    return None


def convert_test_data(input_path, output_dir, use_local_video=False, video_dirs=None, auto_download=False):
    """将标注格式转为 swift infer 标准格式

    Args:
        video_dirs: 本地视频候选目录列表，按优先级从高到低排列
        auto_download: 本地找不到视频时是否自动从 cos_url 下载到第一个目录
    """
    os.makedirs(output_dir, exist_ok=True)

    output_jsonl = os.path.join(output_dir, "test_for_infer.jsonl")
    mapping_jsonl = os.path.join(output_dir, "video_id_mapping.jsonl")

    samples = []
    mappings = []
    n_local = 0
    n_downloaded = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] 第 {line_no} 行 JSON 解析失败: {e}")
                continue

            video_id = item["video_id"]
            cos_url = item.get("cos_url", "")
            ground_truth = {"segments": item.get("segments", [])}

            # 视频源：优先使用 cos_url 本地绝对路径；否则在候选目录查找，最后按需下载
            if use_local_video and video_dirs:
                # cos_url 已是可用本地路径时，直接复用，避免逐目录扫描
                if os.path.isabs(cos_url) and os.path.exists(cos_url):
                    video_source = cos_url
                    n_local += 1
                else:
                    # 再尝试在本地候选目录中按 video_id 查找
                    existing_path = None
                    for d in video_dirs:
                        p = os.path.join(d, f"{video_id}.mp4")
                        if os.path.exists(p):
                            existing_path = p
                            break

                    if existing_path:
                        video_source = existing_path
                        n_local += 1
                    else:
                        # 本地没有，尝试下载
                        local_path = find_local_video(video_id, video_dirs, cos_url=cos_url, auto_download=auto_download)
                        if local_path:
                            video_source = local_path
                            n_downloaded += 1
                        else:
                            searched_dirs = "\n".join(f"    - {d}" for d in video_dirs)
                            msg = (
                                f"\n[ERROR] 视频文件未找到: {video_id}.mp4\n"
                                f"  搜索了以下 {len(video_dirs)} 个目录:\n{searched_dirs}\n"
                            )
                            if auto_download:
                                msg += f"  且从 cos_url 下载失败: {cos_url}\n"
                            else:
                                msg += (
                                    f"  提示: 可以加 --auto_download 参数自动从 cos_url 下载缺失的视频\n"
                                    f"  cos_url: {cos_url}\n"
                                )
                            raise FileNotFoundError(msg)
            else:
                video_source = cos_url

            # 构造 ground truth JSON 作为 assistant 消息（swift 会自动提取为 labels）
            gt_json = json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":"))

            swift_sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<video>{USER_PROMPT}"},
                    {"role": "assistant", "content": gt_json},
                ],
                "videos": [video_source],
            }
            samples.append(swift_sample)

            # 映射：行号 → video_id + ground_truth（后续结果对齐用）
            mappings.append({
                "line_no": line_no,
                "video_id": video_id,
                "cos_url": cos_url,
                "ground_truth": ground_truth,
            })

    # 写入 swift 推理数据
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 写入映射文件
    with open(mapping_jsonl, "w", encoding="utf-8") as f:
        for m in mappings:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[INFO] 转换完成!")
    print(f"  输入样本: {len(samples)} 条")
    if use_local_video and video_dirs:
        print(f"  本地已有: {n_local} 条, 新下载: {n_downloaded} 条")
        if auto_download and n_downloaded > 0:
            print(f"  (下载的视频保存在: {video_dirs[0]})")
    print(f"  swift推理数据: {output_jsonl}")
    print(f"  映射文件: {mapping_jsonl}")
    return output_jsonl, mapping_jsonl


def main():
    parser = argparse.ArgumentParser(description="测试数据格式转换")
    parser.add_argument(
        "--input", type=str, default=DEFAULT_TEST_DATA,
        help="输入测试数据 JSONL 路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="输出目录"
    )
    parser.add_argument(
        "--use_local_video", action="store_true",
        help="使用本地视频文件而非 cos_url"
    )
    parser.add_argument(
        "--video_dir", type=str, nargs="+", default=None,
        help="本地视频候选目录，支持多个（按优先级排列，配合 --use_local_video 使用）"
    )
    parser.add_argument(
        "--auto_download", action="store_true",
        help="本地找不到视频时自动从 cos_url 下载到第一个 --video_dir 目录"
    )
    args = parser.parse_args()
    convert_test_data(args.input, args.output_dir, args.use_local_video, args.video_dir, args.auto_download)


if __name__ == "__main__":
    main()
