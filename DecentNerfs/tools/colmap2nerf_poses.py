import os
import sys
import numpy as np
import struct
import math

def read_next_bytes(fid, num_bytes, format_char_sequence, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + format_char_sequence, data)

def read_images_binary(path):
    images = {}

    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):
            image_id = read_next_bytes(fid, 8, "Q")[0]

            qvec = np.array(read_next_bytes(fid, 32, "dddd"))
            tvec = np.array(read_next_bytes(fid, 24, "ddd"))
            camera_id = read_next_bytes(fid, 8, "Q")[0]

            # read image name
            name = ""
            while True:
                c = fid.read(1).decode("utf-8", "ignore")
                if c == "\x00":
                    break
                name += c

            # skip 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2D)

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }

    return images

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return None
    return q / norm

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
    ])

def main(colmap_sparse_dir, out_poses_path):
    images_bin = os.path.join(colmap_sparse_dir, "images.bin")
    images = read_images_binary(images_bin)

    poses = []
    names = []

    for image_id, data in images.items():
        q = data["qvec"]
        t = data["tvec"]

        # Validate q and t
        if np.any(np.isnan(q)) or np.any(np.isnan(t)):
            print(f"[WARN] Skipping image with NaN qvec/tvec: {data['name']}")
            continue

        q_norm = normalize_quaternion(q)
        if q_norm is None:
            print(f"[WARN] Skipping image with invalid quaternion: {data['name']}")
            continue

        R = qvec2rotmat(q_norm)
        if np.any(np.isnan(R)) or np.any(np.isinf(R)):
            print(f"[WARN] Skipping invalid rotation matrix: {data['name']}")
            continue

        # Compute camera-to-world
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t

        if np.any(np.isnan(c2w)) or np.any(np.isinf(c2w)):
            print(f"[WARN] Skipping invalid c2w pose: {data['name']}")
            continue

        poses.append(c2w)
        names.append(data["name"])

    # Sort by filename so NeRF ordering is correct
    poses = [p for _, p in sorted(zip(names, poses))]
    poses = np.stack(poses)

    np.save(out_poses_path, poses)
    print(f"[OK] Saved cleaned poses to: {out_poses_path}")
    print(f"[OK] Valid poses: {poses.shape[0]}")

if __name__ == "__main__":
    colmap_sparse = sys.argv[1]
    out = sys.argv[2]
    main(colmap_sparse, out)
