"""mvs_to_occupancy.py — COLMAP MVS 稠密点云 → 2D 占据栅格(多视角一致深度版)

与 tsdf_to_occupancy.py(单目 DA3 深度版)平行: 深度来源换成 COLMAP patch_match_stereo + stereo_fusion,
几何一致性匹配,边缘锐利,不依赖语义猜测。
自由空间来自 fused.ply.vis(每个点被哪些相机观测到) → 射线投票。

输入:
  --fused_ply   stereo_fusion 的 fused.ply(旁边需有 fused.ply.vis)
  --output_dir  vidmap 输出目录(用 rec 拿位姿/内参)
输出: 与 tsdf_to_occupancy.py 相同(occupancy.pgm/yaml + preview.png + viewer + alignment.json + 点云ply)

用法(vidmap 环境):
    python mvs_to_occupancy.py --output_dir <vidmap输出目录> --fused_ply <dense_ws>/fused.ply --out_dir nav_map_mvs/
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from tsdf_to_occupancy import (estimate_up_axis, rotation_aligning_z,
                               load_rec, write_grid_viewer)


def load_fused_with_vis(ply_path: Path):
    """读 fused.ply(点+颜色) 和 fused.ply.vis(每点的观测相机 id 列表)。"""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    vis_path = Path(str(ply_path) + ".vis")
    # 二进制格式: uint32 点数, 然后每点 uint32 观测数 + N 个 uint32 image_id
    raw = np.fromfile(vis_path, dtype=np.uint32)
    n_pts, off = int(raw[0]), 1
    vis = []
    for _ in range(n_pts):
        nvis = int(raw[off]); off += 1
        vis.append([int(v) for v in raw[off:off + nvis]]); off += nvis
    assert len(vis) == len(points), f"vis 点数 {len(vis)} != 点数 {len(points)}"
    return points, colors, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fused_ply", required=True)
    ap.add_argument("--out_dir", default="nav_map_mvs")
    ap.add_argument("--depth_trunc", type=float, default=15.0)
    ap.add_argument("--z_min", type=float, default=0.10)
    ap.add_argument("--z_max", type=float, default=1.50)
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--occ_thresh", type=float, default=2.0)
    ap.add_argument("--da3_weight", type=float, default=0.6,
                    help="DA3 单目深度投票的权重(0=纯 MVS); 补白墙等弱纹理盲区")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_rec(output_dir / "rec")
    id2fr = {fr["image_id"]: fr for fr in frames}
    points, colors, vis = load_fused_with_vis(Path(args.fused_ply))
    print(f"MVS 稠密点云: {len(points)} 点")

    # 重力对齐(与 DA3 版同法: PCA + 低位 RANSAC)
    cam_centers_all = np.array([fr["c2w"][:3, 3] for fr in frames])
    up = estimate_up_axis(points, cam_centers_all)
    R_total = rotation_aligning_z(up)
    points = points @ R_total.T
    cam_centers_all = cam_centers_all @ R_total.T
    low = points[points[:, 2] < np.percentile(points[:, 2], 15)]
    pcd_low = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low))
    plane, inliers = pcd_low.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=3000)
    n = np.asarray(plane[:3]); n = n if n[2] >= 0 else -n
    if n[2] > 0.95:
        R2 = rotation_aligning_z(n)
        R_total = R2 @ R_total
        points = points @ R2.T
        cam_centers_all = cam_centers_all @ R2.T
        print(f"[重力] 地面 RANSAC 精修, 法线 {n.round(3)}")
    floor_z = float(np.mean(low[inliers, 2])) if len(inliers) > 100 else float(np.percentile(points[:, 2], 5))
    print(f"地面 z={floor_z:.2f}, 相机高度中位 {np.median(cam_centers_all[:, 2] - floor_z):.2f} m")

    # 每相机的对齐后中心
    cam_xy_by_id = {}
    for fr in frames:
        c = R_total @ fr["c2w"][:3, 3]
        cam_xy_by_id[fr["image_id"]] = c[:2]

    # 栅格边界: 占据带点 + 相机
    in_band = (points[:, 2] > floor_z + args.z_min) & (points[:, 2] < floor_z + args.z_max)
    band_xy = points[in_band][:, :2]
    margin = 1.0
    xy_min = np.minimum(band_xy.min(0), cam_centers_all[:, :2].min(0)) - margin
    xy_max = np.maximum(band_xy.max(0), cam_centers_all[:, :2].max(0)) + margin
    res = args.resolution
    W = int(np.ceil((xy_max[0] - xy_min[0]) / res))
    H = int(np.ceil((xy_max[1] - xy_min[1]) / res))

    # log-odds 投票: vis 里每个(点, 观测相机)对就是一条射线
    occ_votes = np.zeros((H, W), np.int32)
    free_votes = np.zeros((H, W), np.int32)
    pt_ij = np.floor((points[:, :2] - xy_min) / res).astype(int)
    pt_ok = (pt_ij[:, 0] >= 0) & (pt_ij[:, 0] < W) & (pt_ij[:, 1] >= 0) & (pt_ij[:, 1] < H)
    cam_mask_cache = {}
    stride = 1
    for pi in range(0, len(points), stride):
        if not pt_ok[pi]:
            continue
        i, j = pt_ij[pi]
        if in_band[pi]:
            occ_votes[j, i] += 1
        for img_id in vis[pi][:8]:          # 每点最多取 8 个观测相机,防过慢
            if img_id not in cam_xy_by_id:
                continue
            cxy = cam_xy_by_id[img_id]
            key = img_id
            ci, cj = np.floor((cxy - xy_min) / res).astype(int)
            if key not in cam_mask_cache:
                cam_mask_cache[key] = np.zeros((H, W), np.uint8)
            cv2.line(cam_mask_cache[key], (ci, cj), (i, j), 1, thickness=1)
        if pi % 300000 == 0:
            print(f"射线投票 {pi}/{len(points)}")
    for mask in cam_mask_cache.values():
        free_votes += mask

    # ---- DA3 单目深度补票(权重小): 补 MVS 在白墙/弱纹理区的盲区 ----
    if args.da3_weight > 0:
        import h5py
        from tsdf_to_occupancy import find_full_depth_h5, load_depth_scales
        depth_h5 = find_full_depth_h5(output_dir)
        scales = load_depth_scales(output_dir)
        da3_occ = np.zeros((H, W), np.int32)
        da3_free = np.zeros((H, W), np.int32)
        with h5py.File(depth_h5) as f:
            for fi, fr in enumerate(frames):
                if fr["name"] not in f:
                    continue
                grp = f[fr["name"]]
                depth = np.asarray(grp["depth"], np.float32) * scales.get(fr["image_id"], 1.0)
                valid = np.asarray(grp["valid"])
                hd, wd = depth.shape
                K = fr["K"].copy()
                K[0] *= wd / fr["w"]; K[1] *= hd / fr["h"]
                stride = 4
                us, vs = np.meshgrid(np.arange(0, wd, stride), np.arange(0, hd, stride))
                d = depth[::stride, ::stride].reshape(-1)
                vv = valid[::stride, ::stride].reshape(-1)
                keep = vv & (d > 0.3) & (d < args.depth_trunc)
                u = us.reshape(-1)[keep]; v_ = vs.reshape(-1)[keep]; d = d[keep]
                rays_c = np.stack([u, v_, np.ones_like(u)], 0)
                Xc = np.linalg.inv(K) @ rays_c * d
                Xa = R_total @ (fr["c2w"][:3, :3] @ Xc + fr["c2w"][:3, 3:4])
                in_b = (Xa[2] > floor_z + args.z_min) & (Xa[2] < floor_z + args.z_max)
                cxy = cam_xy_by_id[fr["image_id"]]
                ci, cj = np.floor((cxy - xy_min) / res).astype(int)
                ij = np.floor((Xa[:2].T - xy_min) / res).astype(int)
                ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
                ij, in_b = ij[ok], in_b[ok]
                occ_ij = ij[in_b]
                np.add.at(da3_occ, (occ_ij[:, 1], occ_ij[:, 0]), 1)
                mask = np.zeros((H, W), np.uint8)
                for i, j in ij[::2]:
                    cv2.line(mask, (ci, cj), (i, j), 1, thickness=1)
                da3_free += mask
                if fi % 60 == 0:
                    print(f"DA3 补票 {fi}/{len(frames)}")
        occ_votes_f = occ_votes.astype(np.float32) + args.da3_weight * da3_occ
        free_votes_f = free_votes.astype(np.float32) + args.da3_weight * da3_free
        print(f"融合投票: MVS 占据 {int(occ_votes.sum())} + DA3×{args.da3_weight} {int(da3_occ.sum())}")
    else:
        occ_votes_f = occ_votes.astype(np.float32)
        free_votes_f = free_votes.astype(np.float32)

    log_odds = 0.85 * occ_votes_f - 0.4 * free_votes_f
    grid = np.full((H, W), -1, np.int8)
    grid[log_odds > args.occ_thresh] = 100
    grid[(log_odds <= args.occ_thresh) & (free_votes_f > 0)] = 0

    occ_mask = (grid == 100).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_OPEN, kernel)
    occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE, kernel)
    grid = np.where(occ_mask == 1, 100, np.where(free_votes_f > 0, 0, -1)).astype(np.int8)

    # 导出(与 DA3 版同格式)
    occ_u8 = np.full_like(grid, 205, np.uint8)
    occ_u8[grid == 0] = 254
    occ_u8[grid == 100] = 0
    Image.fromarray(np.flipud(occ_u8)).save(out_dir / "occupancy.pgm")
    (out_dir / "occupancy.yaml").write_text(
        f"image: occupancy.pgm\nresolution: {res}\n"
        f"origin: [{xy_min[0]:.4f}, {xy_min[1]:.4f}, 0.0]\n"
        f"occupied_thresh: 0.65\nfree_thresh: 0.196\nnegate: 0\n")
    (out_dir / "alignment.json").write_text(json.dumps({
        "R_total": R_total.tolist(), "floor_z": floor_z, "xy_min": xy_min.tolist(),
        "resolution": res, "z_min": args.z_min, "z_max": args.z_max,
        "grid_W": W, "grid_H": H}))
    # 存对齐后的点云(可视化/质检用)
    pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        pcd_out.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_dir / "fused_aligned.ply"), pcd_out)

    cam_ij = np.floor((cam_centers_all[:, :2] - xy_min) / res).astype(int)
    prev = np.stack([occ_u8] * 3, axis=-1)
    for ci, cj in cam_ij:
        if 0 <= cj < H and 0 <= ci < W:
            prev[cj, ci] = [255, 0, 0]
    Image.fromarray(np.flipud(prev)).save(out_dir / "preview.png")
    write_grid_viewer(out_dir / "occupancy_viewer.html", grid, log_odds, cam_ij, res, xy_min)

    print(f"栅格 {W}x{H} @ {res}m | 占据 {int((grid==100).sum())} 格, 空闲 {int((grid==0).sum())} 格")
    print(f"完成: {out_dir}/occupancy.pgm + occupancy.yaml + preview.png + occupancy_viewer.html + fused_aligned.ply")


if __name__ == "__main__":
    main()
