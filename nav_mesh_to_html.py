"""nav_mesh_to_html.py — TSDF 融合 mesh → 自包含交互式 3D HTML(浏览器打开,可旋转/缩放)

输入: tsdf_to_occupancy.py 产出的 nav_map/mesh.ply(带顶点色),可选 vidmap 输出目录用于叠加相机轨迹
输出: 单个 HTML 文件(three.js 走 CDN,数据 base64 内嵌)

用法(vidmap 环境):
    python nav_mesh_to_html.py --mesh map_jobs/<job>/nav_map/mesh.ply \
        --output_dir map_jobs/<job>/output --out nav_3d.html
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from tsdf_to_occupancy import estimate_up_axis, rotation_aligning_z, load_rec


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--output_dir", default=None, help="vidmap 输出目录(给相机轨迹叠加用,可选)")
    ap.add_argument("--out", default="nav_3d.html")
    ap.add_argument("--target_tris", type=int, default=400000, help="mesh 抽稀目标三角形数")
    ap.add_argument("--max_points", type=int, default=600000, help="若转点云显示的最大点数")
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if not mesh.has_vertex_colors():
        raise ValueError("mesh 没有顶点色(确认 TSDF 用的是 RGB8 颜色模式)")
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    print(f"原始 mesh: {len(verts)} 顶点, {len(tris)} 三角形")

    # 重力对齐(与占据栅格同一坐标系)
    cams = None
    if args.output_dir:
        frames = load_rec(Path(args.output_dir) / "rec")
        cams = np.array([fr["c2w"][:3, 3] for fr in frames])
    up = estimate_up_axis(verts, cams if cams is not None else verts[:1])
    R = rotation_aligning_z(up)
    verts = verts @ R.T
    center = verts.mean(0)
    verts = verts - center
    if cams is not None:
        cams = (cams @ R.T) - center

    # 抽稀控制文件体积
    if len(tris) > args.target_tris:
        mesh2 = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(tris))
        mesh2.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)
        mesh2 = mesh2.simplify_quadric_decimation(args.target_tris)
        verts = np.asarray(mesh2.vertices)
        tris = np.asarray(mesh2.triangles)
        colors = (np.asarray(mesh2.vertex_colors) * 255).astype(np.uint8)
        print(f"抽稀后: {len(verts)} 顶点, {len(tris)} 三角形")

    payload = {
        "positions": b64(verts.astype(np.float32)),
        "colors": b64(colors),
        "indices": b64(tris.astype(np.uint32).reshape(-1)),
        "nVerts": len(verts),
        "nIdx": int(tris.size),
        "camPath": b64(cams.astype(np.float32)) if cams is not None else None,
        "nCams": 0 if cams is None else len(cams),
    }

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    Path(args.out).write_text(html)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"完成: {args.out} ({size_mb:.1f} MB)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>导航地图 3D</title>
<style>body{margin:0}#info{position:fixed;top:8px;left:8px;color:#fff;font:13px sans-serif;
background:rgba(0,0,0,.5);padding:6px 10px;border-radius:6px}</style></head>
<body><div id="info">左键旋转 · 右键平移 · 滚轮缩放(红线=建图轨迹)</div>
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

// mesh
const pos = f32(D.positions), col = u8(D.colors), idx = u32(D.indices);
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
const colF = new Float32Array(col.length);
for (let i=0;i<col.length;i++) colF[i]=col[i]/255;
geo.setAttribute('color', new THREE.BufferAttribute(colF, 3));
geo.setIndex(new THREE.BufferAttribute(idx, 1));
geo.computeVertexNormals();
const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({vertexColors:true, side:THREE.DoubleSide}));
// three.js 是 Y-up: 我们的数据是 Z-up,整体旋转
mesh.rotation.x = -Math.PI/2;
scene.add(mesh);

// 相机轨迹(同样做 Y-up 旋转)
if (D.camPath) {
  const cp = f32(D.camPath);
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.BufferAttribute(cp, 3));
  const line = new THREE.Line(lg, new THREE.LineBasicMaterial({color:0xff2222}));
  line.rotation.x = -Math.PI/2;
  scene.add(line);
}

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(1, 2, 1.5);
scene.add(dl);

// 初始视角: 斜上方俯视
const bb = geo.boundingBox ?? (geo.computeBoundingBox(), geo.boundingBox);
const size = bb.getSize(new THREE.Vector3()).length();
camera.position.set(size*0.35, size*0.45, size*0.35);
controls.target.set(0,0,0);
controls.update();

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script></body></html>"""


if __name__ == "__main__":
    main()
