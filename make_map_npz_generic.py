"""make_map_npz_generic.py — VidMap/COLMAP rec npz + 抽帧目录 → 定位 API 用的地图 npz

与 make_vidmap_map_npz.py 的区别：帧目录和内参都走参数/数据，不再写死 short.mp4 的值。
内参从 rec npz 的 cam_params 逐帧读取并映射到 518×518 crop 空间。

用法(map 环境, lingbot-map 目录下):
    python make_map_npz_generic.py \
        --rec_npz output/house_rec.npz \
        --frames_dir /path/to/video_frames/house-xxx \
        --out output/house_map.npz
"""

import argparse
import os

import numpy as np

from lingbot_map.utils.load_fn import load_and_preprocess_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec_npz", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rec = np.load(args.rec_npz, allow_pickle=False)
    names = [str(n) for n in rec["names"]]
    c2w = rec["c2w"].astype(np.float64)          # (N,4,4)
    cam_params = rec["cam_params"].astype(np.float64)  # (N,4) fx fy cx cy @ (width x height)
    W0, H0 = int(rec["width"]), int(rec["height"])
    n = len(names)
    print(f"{n} 帧, 原始分辨率 {W0}x{H0}")

    # ---- 图像: 原图 → crop 518×518(与 lingbot 地图同空间; VidMap 帧已是旋转后方向, 不要再转) ----
    frame_paths = [os.path.join(args.frames_dir, nm) for nm in names]
    images = load_and_preprocess_images(
        frame_paths, mode="crop", image_size=518, patch_size=14
    ).half()
    print(f"预处理后: {tuple(images.shape)}")

    # ---- c2w → w2c ----
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]
    w2c = np.zeros((n, 3, 4), dtype=np.float32)
    w2c[:, :3, :3] = R.transpose(0, 2, 1)
    w2c[:, :3, 3] = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)

    # ---- 内参 → 518×518 crop 空间 ----
    # load_fn crop 模式: 宽缩放到 518 (scale = 518/W0), 高 = round(H0*scale/14)*14, 再垂直中心裁剪 518
    s = 518.0 / W0
    h_scaled = round(H0 * s / 14) * 14
    crop_top = (h_scaled - 518) / 2.0
    intrinsic = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        fx, fy, cx, cy = cam_params[i]
        intrinsic[i] = [
            [fx * s, 0.0, cx * s],
            [0.0, fy * s, cy * s - crop_top],
            [0.0, 0.0, 1.0],
        ]
    print(f"crop 空间内参(第0帧): fx={intrinsic[0,0,0]:.1f} fy={intrinsic[0,1,1]:.1f} "
          f"cx={intrinsic[0,0,2]:.1f} cy={intrinsic[0,1,2]:.1f}")

    np.savez(
        args.out,
        images=images.unsqueeze(0).cpu().numpy(),  # (1,N,3,518,518) fp16 [0,1]
        extrinsic=w2c,                             # (N,3,4) w2c
        intrinsic=intrinsic,                       # (N,3,3)
        names=np.array(names),
    )
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
