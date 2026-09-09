"""tsdf_to_occupancy.py — VidMap 重建 + DA3 深度 → TSDF 融合 → 2D 占据栅格(导航用)

输入: vidmap.run 的输出目录(需先用 --cache-depth-maps 补齐全分辨率深度,并重跑一次 mapping 以产出 depth_lift_scales.json)
输出: nav_map/{occupancy.pgm, occupancy.yaml, preview.png, mesh.ply}

流程:
  1. 读 rec(COLMAP 模型): 关键帧位姿 w2c + 内参(原始 720×1280 空间)
  2. 读 full_depth_maps h5: 每帧 DA3 全分辨率深度(米制) + valid 掩码
  3. 乘 depth_lift_scales.json 里 BA 优化后的每帧深度尺度
  4. TSDF 融合(Open3D ScalableTSDFVolume): 噪声平均 + 自由空间
  5. 重力对齐(优先 GeoCalib 重力方向,缺失则 RANSAC 地面)
  6. 高度切片(地面以上 [z_min, z_max]) + 相机光线标记空闲 → 占据栅格
  7. 导出 ROS map_server 格式(pgm + yaml) + 预览 PNG + 融合 mesh

用法(vidmap 环境):
    python tsdf_to_occupancy.py --output_dir <vidmap输出目录> --out_dir nav_map/
"""

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image


def find_full_depth_h5(output_dir: Path) -> Path:
    hits = sorted(output_dir.rglob("full_depth_maps*.h5"))
    if not hits:
        raise FileNotFoundError(f"找不到全分辨率深度缓存,先跑: python -m vidmap.frontend <视频> --output {output_dir} --mapper-inputs {output_dir}/mapper_inputs --cache-depth-maps")
    return hits[0]


def load_depth_scales(output_dir: Path) -> dict:
    hits = list(output_dir.rglob("depth_lift_scales.json"))
    if not hits:
        print("[警告] 没有 depth_lift_scales.json,深度不乘 BA 优化尺度(建议重跑一次 mapping)")
        return {}
    payload = json.loads(hits[0].read_text())
    return {int(k): float(v) for k, v in payload["scale_by_image_id"].items()}


def load_rec(rec_dir: Path):
    rec = pycolmap.Reconstruction(str(rec_dir))
    images = sorted(rec.images.values(), key=lambda im: im.name)
    out = []
    for im in images:
        cam = rec.cameras[im.camera_id]
        assert cam.model_name == "PINHOLE", cam.model_name
        w2c = np.eye(4); w2c[:3] = np.asarray(im.cam_from_world().matrix())
        c2w = np.eye(4); c2w[:3] = np.asarray(im.cam_from_world().inverse().matrix())
        out.append({
            "name": im.name,
            "image_id": im.image_id,
            "w2c": w2c,                                # 4x4
            "c2w": c2w,                                # 4x4
            "K": np.asarray(cam.calibration_matrix()),
            "w": int(cam.width), "h": int(cam.height),
        })
    return out


def resize_depth(depth: np.ndarray, w: int, h: int) -> np.ndarray:
    """DA3 预测分辨率 → 原始图像分辨率(双线性)。"""
    img = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32)


def estimate_up_axis(points: np.ndarray, cam_centers: np.ndarray) -> np.ndarray:
    """世界系"上"方向: 对点云做 PCA,最小特征值对应的特征向量即竖直方向
    (房间是扁平盒子: 水平两个方向跨度大,竖直方向跨度小,与坐标轴是否对齐无关)。
    符号由"相机在地面上方"确定。"""
    p = points - points.mean(0)
    cov = p.T @ p / len(p)
    eigvals, eigvecs = np.linalg.eigh(cov)
    up = eigvecs[:, 0]                      # 最小特征值 = 最薄方向 = 竖直
    up = up / np.linalg.norm(up)
    if np.median(cam_centers @ up) < np.median(points @ up):
        up = -up
    print(f"[重力] PCA 特征值 {np.sqrt(eigvals).round(2)}, 上轴 {up.round(3)}")
    return up


def rotation_aligning_z(up: np.ndarray) -> np.ndarray:
    """把 up 轴旋到 +Z 的旋转矩阵。"""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    c = float(np.dot(up, z))
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="vidmap.run 的输出目录")
    ap.add_argument("--out_dir", default="nav_map")
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF 体素边长(米)")
    ap.add_argument("--sdf_trunc", type=float, default=0.06, help="TSDF 截断距离(米)")
    ap.add_argument("--depth_trunc", type=float, default=10.0, help="舍弃超过该距离的深度(米)")
    ap.add_argument("--z_min", type=float, default=0.10, help="障碍切片下界(地面以上米)")
    ap.add_argument("--z_max", type=float, default=1.50, help="障碍切片上界(地面以上米)")
    ap.add_argument("--resolution", type=float, default=0.05, help="占据栅格分辨率(米/格)")
    ap.add_argument("--min_pts_per_cell", type=int, default=2, help="(已废弃,log-odds 模式不用)")
    ap.add_argument("--occ_thresh", type=float, default=2.0, help="log-odds 占据判定阈值,越大越保守")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1~3. 读数据 ----
    frames = load_rec(output_dir / "rec")
    scales = load_depth_scales(output_dir)
    depth_h5 = find_full_depth_h5(output_dir)
    frames_dir = next((output_dir / "video_frames").glob("*"))
    print(f"{len(frames)} 个关键帧, 深度缓存 {depth_h5.name}, 尺度因子 {len(scales)} 个")

    # ---- 4. TSDF 融合 ----
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    with h5py.File(depth_h5) as f:
        for i, fr in enumerate(frames):
            if fr["name"] not in f:
                continue
            grp = f[fr["name"]]
            depth = np.asarray(grp["depth"], dtype=np.float32)
            valid = np.asarray(grp["valid"])
            ow, oh = int(grp.attrs["original_width"]), int(grp.attrs["original_height"])
            depth = resize_depth(depth, ow, oh)
            valid = np.asarray(Image.fromarray(valid.astype(np.uint8) * 255).resize((ow, oh), Image.NEAREST)) > 0
            s = scales.get(fr["image_id"], 1.0)
            depth = depth * s
            depth[~valid] = 0.0

            # 内参从 720×1280( rec 空间 ) → 深度图原始空间(通常相同)
            K = fr["K"].copy()
            K[0] *= ow / fr["w"]; K[1] *= oh / fr["h"]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(ow, oh, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

            color = np.asarray(Image.open(frames_dir / fr["name"]).resize((ow, oh)))
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color), o3d.geometry.Image(depth_mm),
                depth_scale=1000.0, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, fr["w2c"])   # Open3D 要 world→camera
            if i % 50 == 0:
                print(f"融合 {i}/{len(frames)}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_dir / "mesh.ply"), mesh)
    pcd = volume.extract_point_cloud()
    print(f"融合完成: 点云 {len(np.asarray(pcd.points))} 点, mesh 已存 {out_dir/'mesh.ply'}")

    # 统计离群点剔除: 去除悬浮噪声团(玻璃/反射/深度错误)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    points = np.asarray(pcd.points)
    print(f"离群点剔除后: {len(points)} 点")

    # ---- 5. 重力对齐: 先粗(点云 PCA)后精(低位点 RANSAC 地面) ----
    cam_centers = np.array([fr["c2w"][:3, 3] for fr in frames])
    up = estimate_up_axis(points, cam_centers)
    R_total = rotation_aligning_z(up)
    points = points @ R_total.T
    cam_centers = cam_centers @ R_total.T

    # 精修: 最低 15% 的点做 RANSAC 平面(此时地面法线≈Z),再用平面法线重对齐
    low = points[points[:, 2] < np.percentile(points[:, 2], 15)]
    pcd_low = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low))
    plane, inliers = pcd_low.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=3000)
    n = np.asarray(plane[:3]); n = n if n[2] >= 0 else -n
    if n[2] > 0.95:
        R2 = rotation_aligning_z(n)
        R_total = R2 @ R_total
        points = points @ R2.T
        cam_centers = cam_centers @ R2.T
        print(f"[重力] 地面 RANSAC 精修, 法线 {n.round(3)}")
    floor_z = float(np.mean(low[inliers, 2])) if len(inliers) > 100 else float(np.percentile(points[:, 2], 5))
    print(f"地面 z={floor_z:.2f}, 相机高度中位 {np.median(cam_centers[:, 2] - floor_z):.2f} m")

    # ---- 6. log-odds 占据栅格(射线投票: 命中=占据证据, 穿过=空闲证据) ----
    # 幻影点会被其它视角穿过的射线否决, 真障碍被反复命中留下
    import cv2
    res = args.resolution

    # 第一遍: 逐帧反投影(深度图原生分辨率,不放大),收集射线端点
    frame_rays = []   # 每帧: (cam_xy, endpoints_xy(N,2), in_band(N,))
    all_band_xy = []
    with h5py.File(depth_h5) as f:
        for fr in frames:
            if fr["name"] not in f:
                continue
            grp = f[fr["name"]]
            depth = np.asarray(grp["depth"], dtype=np.float32)
            valid = np.asarray(grp["valid"])
            hd, wd = depth.shape
            depth = depth * scales.get(fr["image_id"], 1.0)
            K = fr["K"].copy()
            K[0] *= wd / fr["w"]; K[1] *= hd / fr["h"]     # 内参 → 深度图原生分辨率

            stride = 4
            us, vs = np.meshgrid(np.arange(0, wd, stride), np.arange(0, hd, stride))
            d = depth[::stride, ::stride].reshape(-1)
            vv = valid[::stride, ::stride].reshape(-1)
            keep = vv & (d > 0.3) & (d < args.depth_trunc)
            u = us.reshape(-1)[keep]; v_ = vs.reshape(-1)[keep]; d = d[keep]
            rays_c = np.stack([u, v_, np.ones_like(u)], 0)          # 像素齐次
            Xc = np.linalg.inv(K) @ rays_c * d                       # 相机系
            Xa = R_total @ (fr["c2w"][:3, :3] @ Xc + fr["c2w"][:3, 3:4])  # 对齐系
            in_band = (Xa[2] > floor_z + args.z_min) & (Xa[2] < floor_z + args.z_max)
            cam_xy = (R_total @ fr["c2w"][:3, 3])[:2]
            frame_rays.append((cam_xy, Xa[:2].T, in_band))
            all_band_xy.append(Xa[:2, in_band].T)

    band_xy = np.concatenate(all_band_xy) if all_band_xy else np.zeros((0, 2))
    margin = 1.0
    xy_min = np.minimum(band_xy.min(0), cam_centers[:, :2].min(0)) - margin
    xy_max = np.maximum(band_xy.max(0), cam_centers[:, :2].max(0)) + margin
    W = int(np.ceil((xy_max[0] - xy_min[0]) / res))
    H = int(np.ceil((xy_max[1] - xy_min[1]) / res))

    occ_votes = np.zeros((H, W), np.int32)
    free_votes = np.zeros((H, W), np.int32)
    for fi, (cam_xy, pts_xy, in_band) in enumerate(frame_rays):
        ci, cj = np.floor((cam_xy - xy_min) / res).astype(int)
        ij = np.floor((pts_xy - xy_min) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
        ij, in_b = ij[ok], in_band[ok]
        occ_ij = ij[in_b]                                # 占据投票: 带内端点
        np.add.at(occ_votes, (occ_ij[:, 1], occ_ij[:, 0]), 1)
        mask = np.zeros((H, W), np.uint8)                # 空闲投票: 射线途经(隔一抽稀)
        for i, j in ij[::2]:
            cv2.line(mask, (ci, cj), (i, j), 1, thickness=1)
        free_votes += mask
        if fi % 60 == 0:
            print(f"射线投票 {fi}/{len(frame_rays)}")

    log_odds = 0.85 * occ_votes - 0.4 * free_votes
    grid = np.full((H, W), -1, dtype=np.int8)            # -1 未知, 0 空闲, 100 占据
    grid[log_odds > args.occ_thresh] = 100
    grid[(log_odds <= args.occ_thresh) & (free_votes > 0)] = 0

    # 形态学清理: 开运算去掉孤立噪点格, 闭运算补障碍内部小洞
    occ_mask = (grid == 100).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_OPEN, kernel)
    occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE, kernel)
    grid = np.where(occ_mask == 1, 100, np.where(free_votes > 0, 0, -1)).astype(np.int8)
    cam_ij = np.floor((cam_centers[:, :2] - xy_min) / res).astype(int)

    # ---- 7. 导出 ----
    occ_u8 = np.full_like(grid, 205, dtype=np.uint8)      # ROS: 未知=205
    occ_u8[grid == 0] = 254                               # 空闲=白
    occ_u8[grid == 100] = 0                               # 占据=黑
    Image.fromarray(np.flipud(occ_u8)).save(out_dir / "occupancy.pgm")
    (out_dir / "occupancy.yaml").write_text(
        f"image: occupancy.pgm\nresolution: {res}\n"
        f"origin: [{xy_min[0]:.4f}, {xy_min[1]:.4f}, 0.0]\n"
        f"occupied_thresh: 0.65\nfree_thresh: 0.196\nnegate: 0\n")

    # 预览 PNG: 叠加相机轨迹
    prev = np.stack([occ_u8] * 3, axis=-1)
    for ci, cj in cam_ij:
        if 0 <= cj < H and 0 <= ci < W:
            prev[cj, ci] = [255, 0, 0]
    Image.fromarray(np.flipud(prev)).save(out_dir / "preview.png")

    n_occ = int((grid == 100).sum()); n_free = int((grid == 0).sum())
    print(f"栅格 {W}x{H} @ {res}m | 占据 {n_occ} 格, 空闲 {n_free} 格, 其余未知")
    print(f"完成: {out_dir}/occupancy.pgm + occupancy.yaml + preview.png + mesh.ply")


if __name__ == "__main__":
    main()
