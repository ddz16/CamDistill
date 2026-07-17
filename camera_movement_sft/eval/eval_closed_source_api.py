#!/usr/bin/env python3
"""
闭源/大模型 API 评测：在同一份 benchmark 上评测 gemini-3.1-pro / gpt-5.4 /
qwen3-vl-235b-a22b-instruct，与我们自己的模型使用**相同的英文 prompt**，并**直接上传视频**
(base64，不抽帧成多图) 做公平对比。评测口径与本地模型完全一致(复用 postprocess 的
parse_json_response + evaluate_camera_movement_fixed.py)。

要点:
- 三个模型都直传视频(base64)：
    * gemini*  -> gemini_multimodal_url(mimeType=video/mp4, encoded=<b64>) + video_metadata.fps=5
    * 其他(gpt-5.4 / qwen3-vl-235b, 走 Venus) -> venus_multimodal_url(url=data:video/mp4;base64,<b64>)
  fps 仅 Gemini 设(=5, 与我们默认一致)；其它模型不设 fps(由模型默认采样)。
- 视频过大(超过 per-provider 阈值)先用 ffmpeg 做简单质量压缩(缩分辨率 + 丢音轨)再上传。
- prompt = common.SYSTEM_PROMPT + common.USER_PROMPT (与本地评测同一份)。
- 多线程 + 失败重试轮次 + 断点续跑(raw_responses.jsonl)。
- 跑完自动解析 -> eval_pred.jsonl，并调用 evaluate_camera_movement_fixed.py 出 eval_results.json。

鉴权(同参考脚本 classify_youtube_videos_with_gemini.py):
  token = ENV_VENUS_OPENAPI_SECRET_ID + "@876"  (或 --token 直接给)

用法:
  export ENV_VENUS_OPENAPI_SECRET_ID=xxxx
  python eval_closed_source_api.py \
      --input /group/40009/dazhaodu/OurBenchmark/youtube_benchmark_subset.jsonl \
      --model gemini-3.1-pro
  # gpt-5.4 / qwen3-vl-235b-a22b-instruct 各跑一次即可(结果目录按 model 分开)
"""

import argparse
import base64
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, _PROJECT_DIR)   # for common
sys.path.insert(0, _EVAL_DIR)      # for postprocess_results

from common import SYSTEM_PROMPT, USER_PROMPT           # 与本地评测同一份英文 prompt
from postprocess_results import parse_json_response      # 与本地评测同一套解析

API_URL_DEFAULT = "http://v2.open.venus.oa.com/llmproxy/chat/completions"


# ============================================================================
# 鉴权(同参考脚本)
# ============================================================================
def build_token(explicit_token: Optional[str], token_suffix: str) -> str:
    if explicit_token:
        return explicit_token
    # 默认写死 secret_id(可用环境变量 ENV_VENUS_OPENAPI_SECRET_ID 覆盖，或 --token 直接给完整 token)
    secret_id = os.environ.get("ENV_VENUS_OPENAPI_SECRET_ID", "").strip() or "BXATHoIdLOKMnryLGdX6u410"
    return f"{secret_id}{token_suffix}"


# ============================================================================
# provider 判定 + 上传大小阈值(base64 直传)
# ============================================================================
def get_provider(model: str, override: Optional[str]) -> str:
    if override:
        return override
    m = model.lower()
    if "gemini" in m:
        return "gemini"
    # OpenAI 系(gpt / o1 / o3 / o4)在本网关只支持图片, 不支持视频 -> 走"带时间戳帧图"
    if "gpt" in m or m.startswith(("o1", "o3", "o4")):
        return "openai"
    # 其它 Venus 多模态模型(如 qwen3-vl-235b) -> 直传视频
    return "venus"


# per-provider 建议的原始文件大小上限(MB)。超过才压缩，否则原样 base64 直传。
# (openai 走抽帧, 不直传视频, 该阈值对它无意义)
DEFAULT_MAX_MB = {"gemini": 9.0, "venus": 9.0, "openai": 9.0}


# ============================================================================
# 视频定位 + 压缩 + base64
# ============================================================================
def resolve_video_path(video_id: str, cos_url: str, video_dirs: List[str]) -> Optional[str]:
    # 优先用 cos_url(若为本地绝对路径)，否则在候选目录里按 {video_id}.mp4 查找本地视频
    if cos_url and os.path.isabs(cos_url) and os.path.exists(cos_url):
        return cos_url
    for d in video_dirs:
        if not d:
            continue
        p = os.path.join(d, f"{video_id}.mp4")
        if os.path.exists(p):
            return p
    return None


def _file_mb(path: str) -> float:
    return os.path.getsize(path) / 1024.0 / 1024.0


def compress_video(src: str, dst_dir: str, video_id: str, target_mb: float) -> str:
    """用 ffmpeg 做简单质量压缩(缩分辨率 + 丢音轨)直到不超过 target_mb；失败则返回原文件。
    结果按 (video_id,target) 缓存，重复运行复用。"""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{video_id}.mp4")
    if os.path.exists(dst) and _file_mb(dst) <= target_mb:
        return dst
    # 本机 ffmpeg 无 libx264/-crf，用广泛可用的 mpeg4 编码器 + -q:v (值越大越小)；丢音轨、缩分辨率。
    for width, qv in [(640, 6), (512, 10), (512, 16), (400, 21), (320, 27)]:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                 "-vf", f"scale='min({width},iw)':-2",
                 "-c:v", "mpeg4", "-q:v", str(qv), "-an", dst],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300,
            )
        except Exception as e:
            print(f"[compress] {video_id} ffmpeg 失败(w={width},q={qv}): {str(e)[:120]}")
            continue
        if os.path.exists(dst) and _file_mb(dst) <= target_mb:
            return dst
    # 尽力而为：即便仍超阈值也返回(交给 API 尝试)
    return dst if os.path.exists(dst) else src


def video_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# 构造 content block(直传视频)
# ============================================================================
def build_video_content(provider: str, b64: str, fps: int) -> dict:
    if provider == "gemini":
        return {
            "type": "gemini_multimodal_url",
            "gemini_multimodal_url": {"mimeType": "video/mp4", "encoded": b64},
            "video_metadata": {"fps": fps},
        }
    # venus (qwen3-vl-235b 等)
    return {
        "type": "venus_multimodal_url",
        "venus_multimodal_url": {"mimeType": "video/mp4", "url": f"data:video/mp4;base64,{b64}"},
    }


# ============================================================================
# OpenAI(gpt-5.4)专用：抽帧 + 时间戳。OpenAI 不支持视频，只能喂图片，
# 因此把每帧时间戳写进 prompt，给模型时序锚点以输出 start_time/end_time。
# 抽帧策略与本地模型/Gemini 对齐：fps=5、最多 max_frames 帧。
# ============================================================================
def extract_frames_with_timestamps(video_path, fps=5, max_frames=100,
                                    max_side=512, jpeg_quality=80):
    import decord
    import cv2
    vr = decord.VideoReader(video_path)
    total = len(vr)
    vfps = float(vr.get_avg_fps()) or float(fps)
    duration = total / vfps if vfps > 0 else 0.0
    n = min(int(duration * fps), max_frames) if duration > 0 else min(max_frames, total)
    n = max(4, min(n, total))
    import numpy as np
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = vr.get_batch(list(idxs)).asnumpy()  # (n, H, W, 3) RGB
    out = []
    for k, idx in enumerate(idxs):
        t = float(idx) / vfps
        img = frames[k]
        h, w = img.shape[:2]
        scale = min(1.0, max_side / float(max(h, w)))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        bgr = img[:, :, ::-1]  # RGB -> BGR for cv2 encode
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            continue
        b64 = base64.b64encode(enc.tobytes()).decode("utf-8")
        out.append((t, f"data:image/jpeg;base64,{b64}"))
    return out, duration


def build_openai_frames_content(frames, duration, detail="low"):
    """user content: 说明(帧率+时间戳规则) + 逐帧[时间戳文本, 图片]。"""
    note = (
        f"\n\nNote: This model cannot read video directly, so the video (duration "
        f"≈ {duration:.1f}s) is uniformly sampled into {len(frames)} frames. "
        f"Each frame below is preceded by its timestamp in seconds (e.g. 't=2.0s'). "
        f"Use these timestamps as the temporal reference for segment start_time/end_time."
    )
    content = [{"type": "text", "text": USER_PROMPT + note}]
    for t, data_url in frames:
        content.append({"type": "text", "text": f"t={t:.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": data_url, "detail": detail}})
    return content


# ============================================================================
# 调用 API
# ============================================================================
def call_api(api_url, token, model, user_content, timeout_sec, max_retries, max_tokens, provider="venus"):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    if provider == "openai":
        # gpt-5 / o 系列: 用 max_completion_tokens, 且只支持默认 temperature(不发 temperature)
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0
    last_err, raw_text = None, ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout_sec)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:400]}"
            else:
                data = resp.json()
                raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                if raw_text.strip():
                    return raw_text, None
                last_err = "empty content"
        except Exception as exc:
            last_err = str(exc)
        if attempt < max_retries:
            time.sleep(min(8, 1.5 ** attempt))
    return raw_text, last_err or "unknown api error"


# ============================================================================
# 单视频处理
# ============================================================================
def infer_one(item: dict, args, token: str) -> dict:
    video_id = str(item.get("video_id", ""))
    cos_url = item.get("cos_url", "")
    provider = get_provider(args.model, args.provider)
    result = {"video_id": video_id, "model": args.model, "provider": provider}

    vpath = resolve_video_path(video_id, cos_url, args.video_dir)
    if not vpath:
        result["status"] = "error"
        result["error"] = "video not found(本地候选目录里没有该 video_id.mp4; 可用 --video_dir 指定)"
        return result

    try:
        if provider == "openai":
            # OpenAI 不支持视频 -> 抽帧(fps=5, 带时间戳) + image_url(detail=low)
            frames, duration = extract_frames_with_timestamps(
                vpath, fps=args.fps, max_frames=args.frame_max,
                max_side=args.frame_max_side, jpeg_quality=args.frame_jpeg_quality)
            if not frames:
                result["status"] = "error"
                result["error"] = "frame extraction returned empty"
                return result
            user_content = build_openai_frames_content(frames, duration, detail=args.detail)
            result["num_frames_sent"] = len(frames)
        else:
            # gemini / venus -> 直传视频(base64)，过大先压缩
            upload_path = vpath
            if _file_mb(vpath) > args.max_upload_mb:
                upload_path = compress_video(vpath, args.compress_dir, video_id, args.max_upload_mb)
            b64 = video_to_base64(upload_path)
            user_content = [{"type": "text", "text": USER_PROMPT},
                            build_video_content(provider, b64, args.fps)]

        raw_text, err = call_api(args.api_url, token, args.model, user_content,
                                 args.timeout, args.max_retries, args.max_tokens, provider=provider)
        result["raw_response"] = raw_text
        if err and not raw_text.strip():
            result["status"] = "error"
            result["error"] = err
            return result
        parsed = parse_json_response(raw_text)
        segments = parsed.get("segments", []) if isinstance(parsed, dict) else []
        result["segments"] = segments
        result["status"] = "ok" if isinstance(parsed, dict) and "segments" in parsed else "parse_fail"
        if result["status"] != "ok":
            result["error"] = "no 'segments' in parsed output"
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        return result


# ============================================================================
# 主流程
# ============================================================================
def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_done(raw_path: str) -> Dict[str, dict]:
    """读取已完成记录(每个 video_id 取最新一条)。容忍中断产生的坏行/半行。"""
    latest = {}
    if not os.path.exists(raw_path):
        return latest
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过被中断写坏的半行
            vid = str(row.get("video_id", ""))
            if vid:
                latest[vid] = row
    return latest


def run_round(round_name, items, args, token, fout, lock):
    ok = err = 0
    failed = []
    total = len(items)
    try:
        from tqdm import tqdm
        _bar = tqdm(total=total, desc=f"[{round_name}]", ncols=100)
    except Exception:
        _bar = None

    def _emit(msg):
        # 不破坏进度条地打印
        if _bar is not None:
            from tqdm import tqdm as _t
            _t.write(msg)
        else:
            print(msg, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(infer_one, it, args, token): it for it in items}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            it = futs[fut]
            res = fut.result()
            if res.get("status") == "ok":
                ok += 1
            else:
                err += 1
                failed.append(it)
                # 中途错误立即打印(video_id + 状态 + 错误信息)
                _emit(f"  [ERR] {res.get('video_id','?')} "
                      f"status={res.get('status')} {str(res.get('error',''))[:180]}")
            with lock:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
            if _bar is not None:
                _bar.update(1)
                _bar.set_postfix(ok=ok, err=err)
            elif i % 20 == 0 or i == total:
                print(f"[{round_name}] {i}/{total} ok={ok} err={err}", flush=True)

    if _bar is not None:
        _bar.close()
    return ok, err, failed


def main():
    ap = argparse.ArgumentParser(description="闭源大模型 API 评测(直传视频, 同 prompt 同口径)")
    ap.add_argument("--input", required=True, help="benchmark jsonl (含 video_id/cos_url/segments)")
    ap.add_argument("--model", default="gemini-3.1-pro",
                    help="gemini-3.1-pro | gpt-5.4 | qwen3-vl-235b-a22b-instruct | qwen3.6-35b-a3b "
                         "(名字含 gemini→gemini直传视频; 含 gpt/o1/o3→抽帧; 其余→Venus 直传视频)")
    ap.add_argument("--provider", default=None, choices=[None, "gemini", "venus", "openai"],
                    help="覆盖 provider: gemini/venus=直传视频, openai=抽帧+时间戳(gpt 系)")
    ap.add_argument("--model_tag", default=None, help="输出目录标签(默认按 model 名生成)")
    ap.add_argument("--output_root", default=os.path.join(_EVAL_DIR, "eval_results"))
    ap.add_argument("--video_dir", nargs="+", default=[
        "/group/40009/dazhaodu/OurBenchmark/YouTube_Videos",
        "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos",
    ], help="本地视频候选目录(按 {video_id}.mp4 查找; cos_url 为本地绝对路径时优先用它)")
    ap.add_argument("--api_url", default=API_URL_DEFAULT)
    ap.add_argument("--token", default=None)
    ap.add_argument("--token_suffix", default="@876")
    ap.add_argument("--fps", type=int, default=5, help="Gemini: video_metadata.fps; OpenAI: 抽帧帧率")
    ap.add_argument("--max_upload_mb", type=float, default=None,
                    help="视频直传: 超过则先压缩(默认 gemini/venus=9.0)")
    # OpenAI 抽帧参数
    ap.add_argument("--frame_max", type=int, default=50,
                    help="OpenAI 抽帧最多帧数(网关单次最多 50 张图, 勿超过 50)")
    ap.add_argument("--frame_max_side", type=int, default=512, help="OpenAI 帧最长边像素(detail=low 内部会再降采样)")
    ap.add_argument("--frame_jpeg_quality", type=int, default=80, help="OpenAI 帧 JPEG 质量")
    ap.add_argument("--detail", default="low", choices=["low", "high"], help="OpenAI image_url detail(默认 low)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=8192,
                    help="输出上限。gemini-3.1-pro/gpt-5 等思考型模型: reasoning+输出共享此预算, "
                         "reasoning 可能上万 tokens, 太小会被吃光导致空输出(MAX_TOKENS), 故默认 8192")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--error_retry_rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条(0=全部)")
    ap.add_argument("--overwrite", action="store_true", help="不续跑, 清空重来")
    ap.add_argument("--no_eval", action="store_true", help="只推理, 不自动跑评测")
    args = ap.parse_args()

    if args.max_upload_mb is None:
        args.max_upload_mb = DEFAULT_MAX_MB[get_provider(args.model, args.provider)]

    token = build_token(args.token, args.token_suffix)

    tag = args.model_tag or ("api_" + re.sub(r"[^0-9a-zA-Z._-]+", "_", args.model))
    testset = Path(args.input).stem
    run_dir = os.path.join(args.output_root, f"{tag}_{testset}")
    os.makedirs(run_dir, exist_ok=True)
    raw_path = os.path.join(run_dir, "raw_responses.jsonl")
    pred_path = os.path.join(run_dir, "eval_pred.jsonl")
    args.compress_dir = os.path.join(run_dir, "compressed")

    items = load_jsonl(args.input)
    if args.limit > 0:
        items = items[: args.limit]

    if args.overwrite and os.path.exists(raw_path):
        os.remove(raw_path)
    done = load_done(raw_path)
    done_ok = {vid for vid, r in done.items() if r.get("status") == "ok"}
    pending = [it for it in items if str(it.get("video_id", "")) not in done_ok]

    prov = get_provider(args.model, args.provider)
    mode = "抽帧+时间戳(fps=%d, max=%d, detail=%s)" % (args.fps, args.frame_max, args.detail) \
        if prov == "openai" else "直传视频" + ("(fps=%d)" % args.fps if prov == "gemini" else "")
    print(f"[INFO] model={args.model} provider={prov} 输入方式={mode}")
    print(f"[INFO] total={len(items)} done_ok={len(done_ok)} pending={len(pending)} -> {run_dir}")

    if pending:
        lock = threading.Lock()
        with open(raw_path, "a", encoding="utf-8") as fout:
            round_items = pending
            for ridx in range(max(0, args.error_retry_rounds) + 1):
                name = "initial" if ridx == 0 else f"retry-{ridx}"
                ok, err, failed = run_round(name, round_items, args, token, fout, lock)
                print(f"[INFO] round={name} done ok={ok} err={err}")
                if not failed or ridx == args.error_retry_rounds:
                    break
                time.sleep(3.0)
                round_items = failed

    # ---- 汇总 eval_pred.jsonl(每个 video_id 取最新 ok 结果) ----
    latest = load_done(raw_path)
    n_pred = 0
    with open(pred_path, "w", encoding="utf-8") as f:
        for vid, r in latest.items():
            if r.get("status") == "ok":
                f.write(json.dumps({"video_id": vid, "segments": r.get("segments", [])},
                                   ensure_ascii=False) + "\n")
                n_pred += 1
    print(f"[INFO] eval_pred.jsonl: {n_pred} 条 -> {pred_path}")

    if args.no_eval:
        return
    if n_pred == 0:
        print("[WARN] 没有成功预测，跳过评测。")
        return

    # ---- 评测(与本地模型同口径) ----
    eval_script = os.path.join(_EVAL_DIR, "evaluate_camera_movement_fixed.py")
    summary = os.path.join(run_dir, "eval_results.json")
    log = os.path.join(run_dir, "eval_log.txt")
    print(f"[INFO] 评测 -> {summary}")
    with open(log, "w", encoding="utf-8") as lf:
        subprocess.run(
            [sys.executable, eval_script, "--gt", args.input, "--pred", pred_path,
             "--iou_thresh", "0.3", "0.5", "0.7", "--output", summary],
            stdout=lf, stderr=subprocess.STDOUT,
        )
    # 回显关键指标
    try:
        d = json.load(open(summary))
        r = list(d.values())[0]
        fl = r["frame_level"]
        print(f"[RESULT] {args.model}: frame micro-F1={fl['basic_movement_with_direction']['f1']} "
              f"macro-F1={fl.get('basic_movement_with_direction_macro', {}).get('f1')}")
        for k, s in r["segment_level"].items():
            print(f"         {k}: Loc-F1={s['segment_localization']['f1']} "
                  f"Det-F1={s['segment_detection']['f1']}")
    except Exception:
        pass
    print(f"[DONE] 日志: {log}")


if __name__ == "__main__":
    main()
