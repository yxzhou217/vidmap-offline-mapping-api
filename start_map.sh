#!/bin/bash
# 启动离线建图 API(后台运行,日志在 map_server.log)
# 用法: bash start_map.sh [端口, 默认 8200]
# 说明: 服务本身不占显卡;每个建图任务提交时才自动挑一张空闲 GPU(需 >20GB 余量)
# 如需指定 vidmap/map 两个 conda 环境的 python 或 VidMap 仓库位置:
#   VIDMAP_PYTHON=/path/to/envs/vidmap/bin/python VIDMAP_DIR=/path/to/vidmap bash start_map.sh
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if pgrep -f "map_server.py" > /dev/null; then
    echo "服务已在运行 (PID: $(pgrep -f 'map_server.py' | tr '\n' ' '))"
    echo "要重启请先: bash stop_map.sh"
    exit 0
fi

nohup ${MAP_PYTHON:-python} "$D/map_server.py" \
    --port "${1:-8200}" > "$D/map_server.log" 2>&1 &

echo "已启动 (PID: $!),端口: ${1:-8200}"
echo "确认就绪: curl http://127.0.0.1:${1:-8200}/health"
