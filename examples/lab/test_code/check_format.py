import h5py
from idna import intranges_contain
import numpy as np
from PIL import Image  # Added for image saving
from PIL import ImageDraw

def is_image_dataset(name, obj):
    # simple heuristic: image datasets are uint8 and 4D with last dim = 3
    return (
        isinstance(obj, h5py.Dataset)
        and obj.dtype == np.uint8
        and obj.ndim == 4
        and obj.shape[-1] == 3
    )

def print_hdf5_structure_with_examples(h5_path):
    with h5py.File(h5_path, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"[GROUP]   {name}")

            elif isinstance(obj, h5py.Dataset):
                print(f"[DATASET] {name}")
                print(f"  shape: {obj.shape}")
                print(f"  dtype: {obj.dtype}")

                if is_image_dataset(name, obj):
                    try:
                        # Grab the first frame for inspection
                        first_frame = obj[0]
                        
                        # Save as RGB
                        img_rgb = Image.fromarray(first_frame)
                        rgb_filename = f"check_rgb_{name.replace('/', '_')}.png"
                        img_rgb.save(rgb_filename)
                        
                        # Save as BGR (by reversing the last dimension)
                        img_bgr = Image.fromarray(first_frame[:, :, ::-1])
                        bgr_filename = f"check_bgr_{name.replace('/', '_')}.png"
                        img_bgr.save(bgr_filename)
                        
                        print(f"  [IMAGE SAVED] Saved test frames to {rgb_filename} and {bgr_filename}")
                    except Exception as e:
                        print(f"  [ERROR] Could not save image: {e}")
                else:
                    try:
                        example = obj[0]
                        print(f"  example[0]: {example}")
                    except Exception as e:
                        print(f"  example: <could not read> ({e})")

        f.visititems(visitor)


def project_points_with_matrix(points_3d, projection_matrix):
    """
    Project 3D points to image coordinates with a projection matrix.

    Args:
        points_3d: (N, 3) array.
        projection_matrix: (3, 4) or (4, 4) array.

    Returns:
        uv: (N, 2) projected pixel coordinates.
        valid: (N,) bool mask for numerically valid projections.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    proj = np.asarray(projection_matrix, dtype=np.float64)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must be (N,3), got {points_3d.shape}")
    if proj.shape not in ((3, 4), (4, 4)):
        raise ValueError(f"projection_matrix must be (3,4) or (4,4), got {proj.shape}")

    pts_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)], axis=1)
    p = (proj @ pts_h.T).T

    if proj.shape == (3, 4):
        denom = p[:, 2]
        valid = np.abs(denom) > 1e-9
        uv = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float64)
        uv[valid] = p[valid, :2] / denom[valid, None]
        return uv, valid

    # For 4x4 projective transforms, divide by w then use x,y.
    denom = p[:, 3]
    valid = np.abs(denom) > 1e-9
    uv = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid] = p[valid, :2] / denom[valid, None]
    return uv, valid


def sample_three_frames_and_project_ee_pose(
    h5_path,
    view_dataset_name,
    projection_matrix,
    pose_dataset_name=None,
    following_frames=16,
    output_prefix="projected_ee",
    seed=None,
    radius=4,
):
    """
    Pick 3 random start frames from one camera view.
    For each start frame, overlay a short EE trajectory on that start frame only.

    `following_frames` is the number of overlaid points, including the start frame.
    """
    rng = np.random.default_rng(seed)

    if pose_dataset_name is None:
        pose_name, pose = pick_pose_dataset(h5_path)
    else:
        with h5py.File(h5_path, "r") as f:
            if pose_dataset_name not in f:
                raise KeyError(f"Pose dataset not found: {pose_dataset_name}")
            pose = f[pose_dataset_name][:]
        pose_name = pose_dataset_name

    if pose.ndim != 2 or pose.shape[1] < 3:
        raise ValueError(f"Expected pose shape (T,>=3), got {pose.shape}")

    ee_xyz = np.asarray(pose[:, :3], dtype=np.float64)

    with h5py.File(h5_path, "r") as f:
        resolved_view_name = view_dataset_name
        if resolved_view_name not in f:
            candidate = f"observations/images/{view_dataset_name}"
            if candidate in f:
                resolved_view_name = candidate
            else:
                raise KeyError(
                    "Image dataset not found. Tried "
                    f"'{view_dataset_name}' and '{candidate}'."
                )
        images = f[resolved_view_name]

        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected image dataset shape (T,H,W,3), got {images.shape}")

        T = images.shape[0]
        if ee_xyz.shape[0] != T:
            raise ValueError(
                f"Time dimension mismatch between image frames ({T}) and pose ({ee_xyz.shape[0]})"
            )

        window_len = max(1, int(following_frames))
        max_start = max(0, T - window_len)
        num = min(3, T)
        start_ids = rng.choice(max_start + 1, size=num, replace=False)

        for start_idx in np.sort(start_ids):
            end_idx = min(T, start_idx + window_len)
            base_frame = np.asarray(images[start_idx])
            h, w = base_frame.shape[:2]
            img = Image.fromarray(base_frame)
            draw = ImageDraw.Draw(img)

            frame_ids = np.arange(start_idx, end_idx)
            xyz_seq = ee_xyz[frame_ids]
            uv_seq, valid_seq = project_points_with_matrix(xyz_seq, projection_matrix)

            n = len(frame_ids)
            for i, ((u, v), ok) in enumerate(zip(uv_seq, valid_seq)):
                if not ok:
                    continue
                if not (0 <= u < w and 0 <= v < h):
                    continue

                # Start is largest, end is smallest.
                rr = max(1.0, float(radius) * (1.0 - 0.75 * (i / max(1, n - 1))))
                if i == 0:
                    color = (0, 255, 0)      # start
                elif i == n - 1:
                    color = (255, 0, 0)      # end
                else:
                    color = (255, 165, 0)    # middle

                draw.ellipse(
                    [(u - rr, v - rr), (u + rr, v + rr)],
                    outline=color,
                    fill=color,
                )

            out_name = (
                f"{output_prefix}_{resolved_view_name.replace('/', '_')}"
                f"_start_{int(start_idx)}_len_{int(n)}.png"
            )
            img.save(out_name)
            print(f"[PROJECTED] pose={pose_name} saved: {out_name}")

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ---------- math utils ----------
def quat_to_rotmat_wxyz(q):
    """q: (..., 4) in (w, x, y, z). Returns (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2*(yy + zz)
    R[..., 0, 1] = 2*(xy - wz)
    R[..., 0, 2] = 2*(xz + wy)

    R[..., 1, 0] = 2*(xy + wz)
    R[..., 1, 1] = 1 - 2*(xx + zz)
    R[..., 1, 2] = 2*(yz - wx)

    R[..., 2, 0] = 2*(xz - wy)
    R[..., 2, 1] = 2*(yz + wx)
    R[..., 2, 2] = 1 - 2*(xx + yy)
    return R


def pick_pose_dataset(h5_path, preferred_names=(
    "observations/ee_pose",
    "observations/eef_pose",
    "ee_pose",
    "eef_pose",
    "actions/ee_pose",
)):
    """
    Tries preferred names first, then searches for a dataset that looks like (T,7) or (T,8).
    Returns (name, numpy_array).
    """
    with h5py.File(h5_path, "r") as f:
        # 1) preferred keys
        for k in preferred_names:
            if k in f and isinstance(f[k], h5py.Dataset):
                arr = f[k][:]
                return k, arr

        # 2) heuristic search
        candidates = []

        def visitor(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.ndim != 2:
                return
            if obj.shape[1] not in (7, 8):
                return
            if not np.issubdtype(obj.dtype, np.floating):
                return

            lname = name.lower()
            # bias towards names containing ee/eef and pose
            score = 0
            if "pose" in lname: score += 2
            if "ee" in lname or "eef" in lname: score += 2
            if "obs" in lname or "observ" in lname: score += 1
            candidates.append((score, name))

        f.visititems(visitor)

        if not candidates:
            raise RuntimeError(
                "Couldn't find any (T,7) or (T,8) float dataset that looks like an EE pose.\n"
                "Tip: print your HDF5 tree and tell me which dataset is the end-effector pose."
            )

        candidates.sort(reverse=True)
        best_name = candidates[0][1]
        arr = f[best_name][:]
        return best_name, arr


def guess_quat_order(pose):
    """
    pose: (T,7) or (T,8). Assumes first 3 are xyz.
    Returns "wxyz" or "xyzw" guess based on which yields quats closer to unit norm.
    """
    q = pose[:, 3:7]
    # treat as wxyz
    n1 = np.mean(np.abs(np.linalg.norm(q, axis=1) - 1.0))
    # treat as xyzw -> reorder to wxyz
    q2 = q[:, [3, 0, 1, 2]]
    n2 = np.mean(np.abs(np.linalg.norm(q2, axis=1) - 1.0))
    return "wxyz" if n1 <= n2 else "xyzw"


# ---------- plotting ----------
def plot_ee_trajectory_and_pose(
    h5_path,
    pose_dataset_name=None,
    quat_order="auto",     # "auto" | "wxyz" | "xyzw"
    stride=10,             # how often to draw orientation axes
    axis_len=0.05,         # length of orientation axes in same units as xyz
    title=None,
    drift_x=0.0,           # total added to x from start->end
    drift_y=0.0,           # total added to y from start->end
    drift_z=0.0,           # total added to z from start->end
):
    # load pose
    if pose_dataset_name is None:
        name, pose = pick_pose_dataset(h5_path)
    else:
        with h5py.File(h5_path, "r") as f:
            pose = f[pose_dataset_name][:]
        name = pose_dataset_name

    if pose.shape[1] < 7:
        raise ValueError(f"{name} has shape {pose.shape}, expected at least 7 dims (xyz + quat).")

    xyz = pose[:, 0:3].astype(np.float64)

    # ---- apply drift / offset over time (linear ramp) ----
    T = xyz.shape[0]
    if T > 1:
        t = np.linspace(0.0, 1.0, T)[:, None]                 # (T,1)
    else:
        t = np.zeros((T, 1), dtype=np.float64)
    drift = t * np.array([drift_x, drift_y, drift_z], dtype=np.float64)[None, :]
    xyz = xyz + drift
    # -----------------------------------------------------

    if quat_order == "auto":
        quat_order = guess_quat_order(pose)

    q_raw = pose[:, 3:7]
    if quat_order == "xyzw":
        q = q_raw[:, [3, 0, 1, 2]]
    elif quat_order == "wxyz":
        q = q_raw
    else:
        raise ValueError("quat_order must be 'auto', 'wxyz', or 'xyzw'.")

    R = quat_to_rotmat_wxyz(q)  # (T,3,3)

    idx = np.arange(0, xyz.shape[0], max(1, int(stride)))
    P = xyz[idx]
    Rs = R[idx]

    x_dir = Rs[:, :, 0]
    y_dir = Rs[:, :, 1]
    z_dir = Rs[:, :, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color='green', marker="o", s=100, label='Start')
    # ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color='red', marker="x", s=100, label='End')
    ax.scatter(0, 0, 0, color='black', marker="*", s=100, label='Origin')
    ax.legend()

    ax.quiver(P[:, 0], P[:, 1], P[:, 2], x_dir[:, 0], x_dir[:, 1], x_dir[:, 2],
              length=axis_len, color='r', normalize=True)
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], y_dir[:, 0], y_dir[:, 1], y_dir[:, 2],
              length=axis_len, color='g', normalize=True)
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], z_dir[:, 0], z_dir[:, 1], z_dir[:, 2],
              length=axis_len, color='b', normalize=True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    tstr = title or (
        f"EE trajectory + pose frames\n{h5_path}\n"
        f"(dataset: {name}, quat: {quat_order}, drift: [{drift_x},{drift_y},{drift_z}])"
    )
    ax.set_title(tstr)

    mins = np.minimum(xyz.min(axis=0), 0)
    maxs = np.maximum(xyz.max(axis=0), 0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max()
    ax.set_xlim(center[0] - span/2, center[0] + span/2)
    ax.set_ylim(center[1] - span/2, center[1] + span/2)
    ax.set_zlim(center[2] - span/2, center[2] + span/2)

    plt.savefig("ee_trajectory_and_pose.png")

    def set_view_and_save(elev, azim, fname):
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(fname)

    set_view_and_save(elev=90, azim=-90, fname="ee_trajectory_top_view.png")
    set_view_and_save(elev=0, azim=-90, fname="ee_trajectory_right_of_robot.png")
    set_view_and_save(elev=0, azim=0, fname="ee_trajectory_front_view.png")

def print_all_joint_actions(h5_path, key="joint_action"):
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"Key not found in HDF5 file: {key}")

        joint_action = f[key][:]  # load full dataset into memory

        print(f"\n=== {key} ===")
        print(f"shape: {joint_action.shape}")
        print(f"dtype: {joint_action.dtype}")
        print("\nValues:\n")

        with np.printoptions(precision=6, suppress=True, linewidth=200):
            print(joint_action)
        
        key = "observations/joint_vel"
        joint_action = f[key][:, 7:9]  # load full dataset into memory

        print(f"\n=== {key} === gripper component ===")
        print(f"shape: {joint_action.shape}")
        print(f"dtype: {joint_action.dtype}")
        print("\nValues:\n")
        
        with np.printoptions(precision=6, suppress=True, linewidth=200):
            print(joint_action)
        
        key = "observations/joint_pos"
        joint_action = f[key][:, 7:9]  # load full dataset into memory

        print(f"\n=== {key} === gripper component ===")
        print(f"shape: {joint_action.shape}")
        print(f"dtype: {joint_action.dtype}")
        print("\nValues:\n")

        with np.printoptions(precision=6, suppress=True, linewidth=200):
            print(joint_action)

        
        joint_pos = f["observations/joint_pos"][:, 7:9]
        joint_vel = f["observations/joint_vel"][:, 7:9]
        
        together = np.concatenate([joint_pos, joint_vel], axis=1)
        
        print(f"\n=== joint_pos and joint_vel concatenated ===")
        print(f"shape: {together.shape}")
        print(f"dtype: {together.dtype}")
        print("\nValues:\n")

        with np.printoptions(precision=6, suppress=True, linewidth=200):
            print(together)

        # key = "stage"
        # joint_action = f[key]  # load full dataset into memory

        # print(f"\n=== {key} ===")
        # print(f"shape: {joint_action.shape}")
        # print(f"dtype: {joint_action.dtype}")
        # print("\nValues:\n")

        # with np.printoptions(precision=6, suppress=True, linewidth=200):
        #     print(joint_action)



if __name__ == "__main__":
    path = "/home/t-qimhuang/disk/datasets/danze_data/paired_106/phantom_real_02_25_rgb/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk/datasets/danze_data/paired_106/rendered_videos_and_actions_02_25_fixed_hdf5/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk/datasets/danze_data/paired_106/phantom_real_02_25_rgb/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk/datasets/danze_data/paired_106/phantom_real_02_25/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk2/danze_syn_data/rendered_videos_and_actions_02_25_fixed_hdf5/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk/datasets/danze_data/lab_training_matched_fake_real_data/rendered_videos_and_actions_02_25_hdf5/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk/datasets/danze_data/rendered_videos_and_actions_02_25/rendered_videos_and_actions_02_25_hdf5/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk/datasets/danze_data/phantom_real_02_25/phantom_real_02_25/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk/datasets/rendered_videos_and_actions_02_09_hdf5/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk2/lab_training_orange_cube_single_point/orange_cube/episode_0.hdf5"
    # path = "/home/t-qimhuang/disk2/labdata_test/pick_and_place/episode_0.hdf5"
    print_hdf5_structure_with_examples(path)
    # plot_ee_trajectory_and_pose(
    #     path,
    #     quat_order="wxyz",
    #     stride=10,
    #     axis_len=0.05,
    #     drift_x=0.0,
    #     drift_y=1.0,
    #     drift_z=0.0,
    # )
    print_all_joint_actions(path, key="joint_action")

    ## projection matrix is P = K @ E_world_to_camera.
    R_c2w = np.array([
        [0.02816316,  0.2178868,  -0.97556762],
        [0.99959024, -0.00114196,  0.0286016 ],
        [0.00511786, -0.97597338, -0.21782968],
    ], dtype=np.float64)
    t_c2w = np.array([1.10002696, -0.00701879, 0.2589829], dtype=np.float64)

    T_c2w = np.eye(4, dtype=np.float64)
    T_c2w[:3, :3] = R_c2w
    T_c2w[:3, 3] = t_c2w
    E_w2c = np.linalg.inv(T_c2w)[:3, :]
    
    ## intrinsic matrix. 

    # intrinsic_3x3 = np.array([
    #     [455.90661621,   0.0,         330.47070312],
    #     [  0.0,         455.78945923, 184.86445618],
    #     [  0.0,           0.0,           1.0       ],
    # ], dtype=np.float64)

    intrinsic_3x3 = np.array([
        [607.875,   0.0,   333.961],
        [  0.0,   607.719, 246.486],
        [  0.0,     0.0,     1.0  ],
    ], dtype=np.float64)

    ## final projection matrix is K @ E_world_to_camera
    P = intrinsic_3x3 @ E_w2c

    
    sample_three_frames_and_project_ee_pose(
        h5_path=path,
        view_dataset_name="camera_front_color",  # or full key
        projection_matrix=P,
        following_frames=16,
        seed=0,
    )
