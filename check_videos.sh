#!/bin/bash
# 检验视频文件是否损坏
# 用法: bash check_videos.sh <video_id1> <video_id2> ...
# 或:   bash check_videos.sh   (检查所有已知失败的)

VIDEO_DIR="/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos"

# 如果传了参数就检查指定的，否则检查已知失败的
if [ $# -gt 0 ]; then
    VIDS=("$@")
else
    VIDS=(
        01d23858a9df59437e75d9a13708fade
        01bd87b8c426aab5a246fc1a6b142cf4
        01d7760505377b4ca3d03ea1ef322fa3
        01e4526fd41014c87d9ddd6f824d8a85
        01f66d354337ad91dd998ac6c9ded427
        59d904fadec26dba32e91d69e26f9562
        7bb0d6b17dea8c3887d22eeb4b0c8192
    )
fi

echo "检查 ${#VIDS[@]} 个视频..."
echo ""

ok=0
bad=0
missing=0

for vid in "${VIDS[@]}"; do
    f="${VIDEO_DIR}/${vid}.mp4"
    if [ ! -f "$f" ]; then
        echo "❌ ${vid}: 文件不存在"
        ((missing++))
    elif ffprobe -v error -show_format "$f" > /dev/null 2>&1; then
        size=$(ls -lh "$f" | awk '{print $5}')
        echo "✅ ${vid}: OK (${size})"
        ((ok++))
    else
        size=$(ls -lh "$f" | awk '{print $5}')
        echo "❌ ${vid}: 损坏 (${size})"
        ((bad++))
    fi
done

echo ""
echo "结果: ✅ ${ok} 正常, ❌ ${bad} 损坏, ⚠️  ${missing} 不存在"
