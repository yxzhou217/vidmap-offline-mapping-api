#!/bin/bash
# 停止离线建图 API
# 用法: bash stop_map.sh
# 注意: 正在跑的建图任务(vidmap 子进程)不会被杀,但服务重启后它的状态会丢
if pkill -f "map_server\.py"; then
    echo "已停止"
else
    echo "服务本来就没在运行"
fi
