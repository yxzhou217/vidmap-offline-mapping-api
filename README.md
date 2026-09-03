# 离线建图 API(VidMap Offline Mapping API)

上传一段视频 → 自动完成离线建图,输出 COLMAP 模型 + 首帧定位可直接使用的 npz 地图和检索库的 HTTP 服务。
建图引擎基于 [VidMap](https://github.com/cvg/vidmap)(ECCV 2026,面向视频的离线全局 SfM),输出**米制尺度**的相机位姿、内参和稀疏点云。

**流水线**:VidMap 建图 → 导出位姿 → 转 518×518 定位地图 npz → 建 DINOv2 检索库 → 3D 可视化 HTML。

配套仓库:建好的地图可直接喂给 [first-frame-reloc-api](https://github.com/yxzhou217/first-frame-reloc-api) 做"苏醒拍照定位"。

---

## 环境安装

需要**两个 conda 环境**:

```bash
# ============ 1. vidmap 环境(建图引擎) ============
conda create -n vidmap python=3.10 -y && conda activate vidmap

# COLMAP 4.1 + pycolmap:需从源码编译,见 https://colmap.github.io/install.html#build-from-source

git clone --recursive https://github.com/cvg/vidmap.git
cd vidmap
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install xformers
pip install -e .
cd ..

# ============ 2. map 环境(API 服务 + 转换/建库) ============
# 详细安装见 https://github.com/yxzhou217/first-frame-reloc-api (与本仓库共用同一个 map 环境)
```

模型权重**无需手动下载**:VidMap 前端模型(~9GB)在第一次建图时自动下载,DINOv2-small(~90MB)在第一次建库时自动从 HuggingFace 下载。国内网络建议 `export HF_ENDPOINT=https://hf-mirror.com`。

## 使用方法

### 启动服务

```bash
# 把两个环境的 python 路径和 VidMap 仓库位置告诉服务(不设置则用默认值)
VIDMAP_DIR=/path/to/vidmap \
VIDMAP_PYTHON=/path/to/miniconda3/envs/vidmap/bin/python \
MAP_PYTHON=/path/to/miniconda3/envs/map/bin/python \
LINGBOT_MAP_DIR=/path/to/lingbot-map \
bash start_map.sh          # 默认端口 8200

curl http://127.0.0.1:8200/health   # 确认就绪
```

### ① 提交建图任务

```bash
curl -X POST http://127.0.0.1:8200/map/build -F "file=@video.mp4"
# {"job_id":"20260903_203744_9a5099","scene":"video", ...}
```

可选 form 参数:

| 参数 | 默认 | 说明 |
|---|---|---|
| `scene` | 视频文件名 | 场景名,决定产物名 `<scene>_map.npz` / `<scene>_db.npz`(重名会覆盖) |
| `keyframe_drift` | 0.11 | 关键帧漂移阈值,调小 = 关键帧更密(如 0.08) |
| `make_html` | true | 是否生成 3D 可视化 HTML |

竖屏视频直接传原文件即可(VidMap 自动应用旋转元数据)。

### ② 轮询状态

```bash
curl http://127.0.0.1:8200/map/status/<job_id>
# status: queued → vidmap_frontend → vidmap_mapping → export → map_npz → db → html → done
# 返回带 log_tail(实时日志尾部)和 elapsed_s;失败时 error 字段带原因
```

### ③ 下载产物(status 为 done 后)

```bash
curl -O http://127.0.0.1:8200/map/download/<job_id>/map_npz   # 定位地图(图像+位姿+内参)
curl -O http://127.0.0.1:8200/map/download/<job_id>/db_npz    # DINO 检索库
curl -O http://127.0.0.1:8200/map/download/<job_id>/rec_zip   # COLMAP 原始模型
curl -O http://127.0.0.1:8200/map/download/<job_id>/html      # 3D 交互可视化(浏览器打开)
```

产物同时注册到 `output/<scene>_map.npz` / `<scene>_db.npz`。

### 其它接口

```bash
curl http://127.0.0.1:8200/map/jobs                    # 列出所有任务
curl -X DELETE http://127.0.0.1:8200/map/<job_id>      # 删除任务目录(已注册的 npz 不受影响)
bash stop_map.sh                                       # 停止服务
```

## 输出格式

- `rec/`(COLMAP 模型): 标准 `cameras/images/points3D.bin`,米制尺度,可用 pycolmap / COLMAP GUI 直接读;
- `<scene>_map.npz`: `images (1,N,3,518,518)`、`extrinsic (N,3,4)`(**w2c** 约定)、`intrinsic (N,3,3)`、`names (N,)`;
- `<scene>_db.npz`: `descriptors (N,384)`(DINOv2-small [CLS],L2 归一化) + 各帧位姿内参。

## 注意事项

- 建图是长任务(几十分钟),**单并发**,多提交自动排队;
- 每个任务自动挑一张空闲显存 >20GB 的 GPU(峰值约 15GB),无空卡时会报错提示稍后再试;
- 竖屏视频不要手动旋转,直接上传原文件。

## License

Apache-2.0,见 [LICENSE](LICENSE)。
