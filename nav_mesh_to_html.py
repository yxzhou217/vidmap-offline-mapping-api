"""nav_mesh_to_html.py — 导航地图最终产物 → 自包含交互式 3D HTML

把整条管线的最终输出叠到一个 3D 场景里:
  - TSDF mesh(半透明,可开关)
  - 占据栅格的障碍立柱(每个占据格从地面立起一根柱子,可开关)
  - 建图轨迹(红线,可开关)
坐标系与占据栅格严格一致(读 nav_map/alignment.json,不重复做重力对齐)。

用法(vidmap 环境):
    python nav_mesh_to_html.py --nav_map map_jobs/<job>/nav_map \
        --mesh map_jobs/<job>/nav_map/mesh.ply \
        --output_dir map_jobs/<job>/output --out nav_3d.html
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from tsdf_to_occupancy import load_rec


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav_map", required=True, help="tsdf_to_occupancy.py / mvs_to_occupancy.py 的输出目录")
    ap.add_argument("--mesh", default=None, help="mesh.ply 路径(默认 <nav_map>/mesh.ply)")
    ap.add_argument("--points", default=None,
                    help="点云 ply 路径(如 fused_aligned.ply, 已对齐过, 只补地面平移; 与 --mesh 二选一)")
    ap.add_argument("--output_dir", default=None, help="vidmap 输出目录(叠加轨迹用,可选)")
    ap.add_argument("--out", default=None, help="默认 <nav_map>/nav_3d.html")
    ap.add_argument("--target_tris", type=int, default=400000)
    ap.add_argument("--max_points", type=int, default=800000)
    args = ap.parse_args()

    nav_map = Path(args.nav_map)
    out_path = Path(args.out) if args.out else nav_map / "nav_3d.html"
    al = json.loads((nav_map / "alignment.json").read_text())
    R_total = np.array(al["R_total"])
    floor_z, res = al["floor_z"], al["resolution"]
    z_max = al["z_max"]
    xy_min = np.array(al["xy_min"])

    # ---- 几何: mesh 或点云; 地面移到 z=0 ----
    point_mode = args.points is not None
    if point_mode:
        pcd = o3d.io.read_point_cloud(args.points)
        verts = np.asarray(pcd.points)              # fused_aligned.ply 已乘过 R_total
        verts[:, 2] -= floor_z
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        if len(verts) > args.max_points:            # 抽稀控制文件体积
            sel = np.random.default_rng(0).choice(len(verts), args.max_points, replace=False)
            verts, colors = verts[sel], colors[sel]
        tris = np.zeros((0, 3), np.uint32)
        print(f"点云: {len(verts)} 点(已对齐)")
    else:
        mesh = o3d.io.read_triangle_mesh(str(args.mesh or nav_map / "mesh.ply"))
        verts = np.asarray(mesh.vertices) @ R_total.T
        verts[:, 2] -= floor_z
        tris = np.asarray(mesh.triangles)
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        print(f"mesh: {len(verts)} 顶点, {len(tris)} 三角形")
        if len(tris) > args.target_tris:
            m2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                           o3d.utility.Vector3iVector(tris))
            m2.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)
            m2 = m2.simplify_quadric_decimation(args.target_tris)
            verts = np.asarray(m2.vertices); tris = np.asarray(m2.triangles)
            colors = (np.asarray(m2.vertex_colors) * 255).astype(np.uint8)
            print(f"抽稀后: {len(verts)} 顶点, {len(tris)} 三角形")

    # ---- 占据栅格 → 障碍立柱位置(pgm 读回: 0=占据 254=空闲 205=未知, 存储时 flipud 过) ----
    grid_img = np.asarray(Image.open(nav_map / "occupancy.pgm"))
    grid = np.flipud(grid_img)                       # 回到 grid[row=y, col=x]
    occ_rc = np.argwhere(grid == 0)                  # 占据格 (row, col)
    occ_pos = np.stack([
        xy_min[0] + (occ_rc[:, 1] + 0.5) * res,      # x
        xy_min[1] + (occ_rc[:, 0] + 0.5) * res,      # y
        np.full(len(occ_rc), z_max / 2),             # 柱子中心高(从地面立到 z_max)
    ], axis=1).astype(np.float32)
    print(f"障碍立柱: {len(occ_pos)} 根, 高度 {z_max}m")

    # ---- 轨迹 ----
    cams = None
    if args.output_dir:
        frames = load_rec(Path(args.output_dir) / "rec")
        cams = np.array([fr["c2w"][:3, 3] for fr in frames]) @ R_total.T
        cams[:, 2] -= floor_z

    # 显示居中
    center = verts.mean(0)
    verts = verts - center
    occ_pos = occ_pos - center.astype(np.float32)
    if cams is not None:
        cams = cams - center

    payload = {
        "positions": b64(verts.astype(np.float32)),
        "mode": "points" if point_mode else "mesh",
        "colors": b64(colors),
        "indices": b64(tris.astype(np.uint32).reshape(-1)),
        "occPos": b64(occ_pos), "occSize": [res, res, z_max],
        "camPath": b64(cams.astype(np.float32)) if cams is not None else None,
    }
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    out_path.write_text(html)
    print(f"完成: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>导航地图 3D</title>
<style>body{margin:0}#bar{position:fixed;top:8px;left:8px;color:#fff;font:13px sans-serif;
background:rgba(0,0,0,.55);padding:6px 10px;border-radius:6px;user-select:none}
#bar label{margin-right:10px;cursor:pointer}</style></head>
<body><div id="bar">
<label><input type="checkbox" id="lyMesh" checked> mesh</label>
<label><input type="checkbox" id="lyOcc" checked> 障碍立柱</label>
<label><input type="checkbox" id="lyTraj" checked> 轨迹</label>
<span style="color:#999">左键旋转 · 右键平移 · 滚轮缩放</span></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D = __DATA__;
function f32(b){return new Float32Array(Uint8Array.from(atob(b),c=>c.charCodeAt(0)).buffer);}
function u8(b){return Uint8Array.from(atob(b),c=>c.charCodeAt(0));}
function u32(b){return new Uint32Array(Uint8Array.from(atob(b),c=>c.charCodeAt(0)).buffer);}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202428);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.01, 500);
const controls = new THREE.OrbitControls(camera, renderer.domElement);

// 数据是 Z-up, 包一层 group 转成 three.js 的 Y-up
const world = new THREE.Group();
world.rotation.x = -Math.PI/2;
scene.add(world);

// mesh(半透明)
const pos = f32(D.positions), col = u8(D.colors), idx = u32(D.indices);
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
const colF = new Float32Array(col.length);
for (let i=0;i<col.length;i++) colF[i]=col[i]/255;
geo.setAttribute('color', new THREE.BufferAttribute(colF, 3));
let mesh;
if (D.mode === 'points') {
  mesh = new THREE.Points(geo, new THREE.PointsMaterial({vertexColors:true, size:0.02}));
} else {
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    vertexColors:true, side:THREE.DoubleSide, transparent:true, opacity:0.35}));
}
world.add(mesh);

// 障碍立柱(InstancedMesh)
const op = f32(D.occPos), nOcc = op.length/3, [sx, sy, h] = D.occSize;
const occGeo = new THREE.BoxGeometry(sx, sy, h);
const occMesh = new THREE.InstancedMesh(occGeo,
  new THREE.MeshStandardMaterial({color:0xcc3333}), nOcc);
const m4 = new THREE.Matrix4();
for (let i=0;i<nOcc;i++){
  m4.makeTranslation(op[3*i], op[3*i+1], op[3*i+2]);
  occMesh.setMatrixAt(i, m4);
}
occMesh.instanceMatrix.needsUpdate = true;
world.add(occMesh);

// 地面参考平面
const bb = geo.boundingBox ?? (geo.computeBoundingBox(), geo.boundingBox);
const size = bb.getSize(new THREE.Vector3()).length();
const ground = new THREE.Mesh(new THREE.PlaneGeometry(size*1.2, size*1.2),
  new THREE.MeshBasicMaterial({color:0xdddddd, transparent:true, opacity:0.25, side:THREE.DoubleSide}));
world.add(ground);

// 轨迹
let line = null;
if (D.camPath) {
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.BufferAttribute(f32(D.camPath), 3));
  line = new THREE.Line(lg, new THREE.LineBasicMaterial({color:0x22ff66}));
  world.add(line);
}

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const dl = new THREE.DirectionalLight(0xffffff, 0.7);
dl.position.set(1, 2, 1.5);
scene.add(dl);

camera.position.set(size*0.35, size*0.45, size*0.35);
controls.target.set(0,0,0);
controls.update();

document.getElementById('lyMesh').onchange = e => mesh.visible = e.target.checked;
document.getElementById('lyOcc').onchange = e => occMesh.visible = e.target.checked;
document.getElementById('lyTraj').onchange = e => { if (line) line.visible = e.target.checked; };

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script></body></html>"""


if __name__ == "__main__":
    main()
