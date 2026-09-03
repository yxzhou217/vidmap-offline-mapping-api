"""Export a VidMap/COLMAP reconstruction dir to rec npz (names, c2w, intrinsic).

Usage (vidmap env): python export_rec_npz.py <rec_dir> <out.npz>
"""

import sys
from pathlib import Path

import numpy as np
import pycolmap


def main():
    rec_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    m = pycolmap.Reconstruction(str(rec_dir))

    images = sorted(m.images.values(), key=lambda im: im.name)
    names, c2w,Ks = [], [], []
    for im in images:
        names.append(im.name)
        c2w.append(np.asarray(im.cam_from_world().inverse().matrix(), dtype=np.float64))
        cam = m.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        Ks.append([fx, fy, cx, cy])
        assert cam.model_name == "PINHOLE", cam.model_name

    np.savez(
        out_path,
        names=np.array(names),
        c2w=np.stack(c2w).astype(np.float32),      # (N,4,4)
        cam_params=np.array(Ks, dtype=np.float32),  # (N,4) fx fy cx cy
        width=np.int64(m.cameras[images[0].camera_id].width),
        height=np.int64(m.cameras[images[0].camera_id].height),
    )
    print(f"{len(names)} images -> {out_path}; K0={Ks[0]}, size={m.cameras[images[0].camera_id].width}x{m.cameras[images[0].camera_id].height}")


if __name__ == "__main__":
    main()
