"""
Create 3-panel rollout analysis videos.

For each rollout, creates a video with:
  [Third-Person View | 3D Trajectory Trace | Gripper Action Plot]

The trajectory panel animates showing the path up to the current frame.
The gripper panel shows a time cursor moving across the gripper action plot.

Usage:
    python scripts/create_rollout_panel_videos.py \
        --eval-dir data/libero/3task_eval/e7_ah40_step4999
    
    # Filter specific tasks
    python scripts/create_rollout_panel_videos.py \
        --eval-dir data/libero/3task_eval/e7_ah40_step4999 \
        --task-filter task08

    # Multiple experiments
    python scripts/create_rollout_panel_videos.py \
        --eval-dir data/libero/3task_eval/e4_speed_varied_half_step2499 \
                   data/libero/3task_eval/e5_original_half_step2499 \
                   data/libero/3task_eval/e6_ah20_step4999 \
                   data/libero/3task_eval/e7_ah40_step4999
"""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_H = 360
PANEL_W = 360
FPS = 10


def load_video_frames(video_path):
    """Load all frames from an mp4 video using imageio."""
    import imageio
    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        frames.append(np.array(frame))
    reader.close()
    return frames


def fig_to_array(fig, h, w):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def resize_frame(frame, h, w):
    img = Image.fromarray(frame)
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def prerender_trajectory_base(traj, chunk_info, xlim, ylim, zlim):
    """Pre-render a static 3D trajectory base image (full path, start marker)."""
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    n = len(traj)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2], color=colors[i], linewidth=1.5, alpha=0.4)
    for ci in chunk_info:
        s = ci["start_step"]
        if s < n:
            ax.scatter([traj[s, 0]], [traj[s, 1]], [traj[s, 2]],
                       color="#2171b5", s=12, marker="D", zorder=8, depthshade=False, alpha=0.3)
    ax.scatter(*traj[0], color="green", s=40, marker="o", zorder=9, depthshade=False)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7); ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=5)
    ax.set_title("EE Trajectory", fontsize=9)
    # Get the renderer transform for projecting 3D→2D
    fig.canvas.draw()
    base_img = fig_to_array(fig, PANEL_H, PANEL_W)
    # Project all trajectory points to pixel coords
    from mpl_toolkits.mplot3d import proj3d
    pixel_coords = []
    for i in range(n):
        x2, y2, _ = proj3d.proj_transform(traj[i, 0], traj[i, 1], traj[i, 2], ax.get_proj())
        x_px, y_px = ax.transData.transform((x2, y2))
        pixel_coords.append((x_px, y_px))
    # Convert from figure coords to image coords
    dpi = fig.dpi
    fig_w, fig_h = fig.get_size_inches()
    fig_w_px, fig_h_px = int(fig_w * dpi), int(fig_h * dpi)
    plt.close(fig)
    return base_img, pixel_coords, (fig_w_px, fig_h_px)


def render_trajectory_frame_fast(base_img, pixel_coords, fig_size, frame_idx):
    """Overlay current position dot on pre-rendered trajectory image using PIL."""
    img = Image.fromarray(base_img.copy())
    draw = ImageDraw.Draw(img)

    # Scale pixel coords from figure space to panel space
    fig_w, fig_h = fig_size
    sx = PANEL_W / fig_w
    sy = PANEL_H / fig_h

    # Draw traversed path highlight (thick blue line up to current frame)
    if frame_idx > 1:
        for i in range(max(0, frame_idx - 30), frame_idx):
            x1, y1 = pixel_coords[i]
            x2, y2 = pixel_coords[i + 1]
            x1, y1 = int(x1 * sx), int((fig_h - y1) * sy)
            x2, y2 = int(x2 * sx), int((fig_h - y2) * sy)
            draw.line([(x1, y1), (x2, y2)], fill=(33, 113, 181), width=3)

    # Current position (red dot)
    px, py = pixel_coords[frame_idx]
    px, py = int(px * sx), int((fig_h - py) * sy)
    r = 6
    draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 0, 0), outline=(200, 0, 0))

    return np.array(img)


def prerender_gripper_base(gripper_executed, chunk_grippers, n_steps, boundaries):
    """Pre-render the static gripper plot base."""
    fig, ax = plt.subplots(figsize=(4, 4))
    for times, values in chunk_grippers:
        ax.plot(times, values, color="#e6550d", linewidth=0.4, alpha=0.15)
    timesteps = np.arange(n_steps)
    ax.plot(timesteps, gripper_executed, color="#2171b5", linewidth=1.5, zorder=5)
    for start in boundaries:
        ax.axvline(x=start, color="gray", linewidth=0.3, alpha=0.2, linestyle="--")
    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("Gripper", fontsize=9)
    ax.set_title("Gripper Action", fontsize=9)
    ax.set_xlim(0, n_steps)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7)
    fig.canvas.draw()
    # Get data→pixel transform
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    base_img = fig_to_array(fig, PANEL_H, PANEL_W)
    # Compute pixel coords for the time cursor
    dpi = fig.dpi
    fig_w, fig_h = fig.get_size_inches()
    fig_w_px, fig_h_px = int(fig_w * dpi), int(fig_h * dpi)
    cursor_coords = []
    for t in range(n_steps):
        x_data, y_data = t, gripper_executed[t]
        x_px, y_px = ax.transData.transform((x_data, y_data))
        cursor_coords.append((x_px, y_px))
    plt.close(fig)
    return base_img, cursor_coords, (fig_w_px, fig_h_px)


def render_gripper_frame_fast(base_img, cursor_coords, fig_size, frame_idx, n_steps):
    """Overlay time cursor on pre-rendered gripper plot."""
    img = Image.fromarray(base_img.copy())
    draw = ImageDraw.Draw(img)
    fig_w, fig_h = fig_size
    sx = PANEL_W / fig_w
    sy = PANEL_H / fig_h

    idx = min(frame_idx, n_steps - 1)
    px, py = cursor_coords[idx]
    px, py = int(px * sx), int((fig_h - py) * sy)

    # Vertical cursor line
    # Map frame_idx x position
    x_line = px
    draw.line([(x_line, 0), (x_line, PANEL_H)], fill=(255, 0, 0, 180), width=2)

    # Dot at current value
    r = 5
    draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 0, 0), outline=(200, 0, 0))

    return np.array(img)


def extract_executed_gripper(traj_data):
    boundaries = traj_data["chunk_boundaries"]
    chunks = traj_data["chunk_full_actions"]
    replan = traj_data["replan_steps"]
    n_steps = traj_data["num_steps"]

    gripper = np.zeros(n_steps)
    for start, chunk in zip(boundaries, chunks):
        chunk = np.array(chunk)
        for j in range(replan):
            t = start + j
            if t < n_steps and j < len(chunk):
                gripper[t] = chunk[j, 6]
    return gripper


def extract_full_chunk_gripper(traj_data):
    boundaries = traj_data["chunk_boundaries"]
    chunks = traj_data["chunk_full_actions"]
    n_steps = traj_data["num_steps"]

    result = []
    for start, chunk in zip(boundaries, chunks):
        chunk = np.array(chunk)
        times = np.arange(start, min(start + len(chunk), n_steps))
        values = chunk[:len(times), 6]
        result.append((times, values))
    return result


def compute_chunk_info(traj_data):
    boundaries = traj_data["chunk_boundaries"]
    replan = traj_data["replan_steps"]
    horizon = traj_data["action_horizon"]
    n_steps = traj_data["num_steps"]

    chunk_info = []
    for i, start in enumerate(boundaries):
        chunk_info.append({
            "start_step": start,
            "exec_end_step": min(start + replan, n_steps - 1),
            "horizon_end_step": min(start + horizon, n_steps - 1),
            "chunk_idx": i,
        })
    return chunk_info


def write_video_ffmpeg(frames, output_path, fps=10):
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait(timeout=120)


def add_header(panel_row, text, h=30, bg=(20, 20, 20), text_color=(255, 255, 255)):
    """Add a text header bar above the panel row."""
    w = panel_row.shape[1]
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((10, 5), text, fill=text_color, font=font)

    result_color = (46, 204, 113) if "[SUCCESS]" in text else (231, 76, 60)
    # Draw result tag on right side
    tag = "[SUCCESS]" if "[SUCCESS]" in text else "[FAILURE]"
    bbox = draw.textbbox((0, 0), tag, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((w - tw - 10, 5), tag, fill=result_color, font=font)

    return np.vstack([np.array(img), panel_row])


def process_rollout(video_path, json_path, output_path):
    """Create a 3-panel video for one rollout."""
    traj_data = json.load(open(json_path))
    video_frames = load_video_frames(video_path)

    ee = np.array(traj_data["ee_positions"])
    n_steps = traj_data["num_steps"]
    chunk_info = compute_chunk_info(traj_data)
    gripper_executed = extract_executed_gripper(traj_data)
    chunk_grippers = extract_full_chunk_gripper(traj_data)

    # Compute axis limits with padding
    margin = 0.02
    xlim = (ee[:, 0].min() - margin, ee[:, 0].max() + margin)
    ylim = (ee[:, 1].min() - margin, ee[:, 1].max() + margin)
    zlim = (ee[:, 2].min() - margin, ee[:, 2].max() + margin)

    success = traj_data["success"]
    task_desc = traj_data["task_description"]
    ep_idx = traj_data["episode_idx"]
    ah = traj_data["action_horizon"]
    replan = traj_data["replan_steps"]
    result_str = "SUCCESS" if success else "FAILURE"
    header = f"Ep {ep_idx}: {task_desc} (ah={ah}, replan={replan})  [{result_str}]"

    n_video_frames = len(video_frames)
    n_total = max(n_steps, n_video_frames)

    # Pre-render static base images (one-time cost)
    traj_base, traj_coords, traj_fig_size = prerender_trajectory_base(
        ee, chunk_info, xlim, ylim, zlim)
    grip_base, grip_coords, grip_fig_size = prerender_gripper_base(
        gripper_executed, chunk_grippers, n_steps, traj_data["chunk_boundaries"])

    combined_frames = []
    print(f"    Rendering {n_total} frames...", end="", flush=True)

    for t in range(n_total):
        # Third-person view
        vid_idx = min(t, n_video_frames - 1)
        tp_frame = resize_frame(video_frames[vid_idx], PANEL_H, PANEL_W)

        # Trajectory (fast overlay)
        traj_idx = min(t, n_steps - 1)
        traj_frame = render_trajectory_frame_fast(traj_base, traj_coords, traj_fig_size, traj_idx)

        # Gripper (fast overlay)
        grip_frame = render_gripper_frame_fast(grip_base, grip_coords, grip_fig_size, traj_idx, n_steps)

        # Combine horizontally
        row = np.hstack([tp_frame, traj_frame, grip_frame])
        row = add_header(row, header)
        combined_frames.append(row)

        if (t + 1) % 100 == 0:
            print(f" {t+1}", end="", flush=True)

    print(" writing...", end="", flush=True)
    write_video_ffmpeg(combined_frames, output_path, fps=FPS)
    print(" done")


def main():
    parser = argparse.ArgumentParser(description="Create 3-panel rollout analysis videos")
    parser.add_argument("--eval-dir", type=str, nargs="+", required=True)
    parser.add_argument("--task-filter", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    for eval_dir_str in args.eval_dir:
        eval_dir = Path(eval_dir_str)
        output_dir = eval_dir / "panel_videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all rollout mp4 + json pairs
        mp4_files = sorted(eval_dir.rglob("rollout_*.mp4"))
        if args.task_filter:
            mp4_files = [f for f in mp4_files if args.task_filter in f.name]
        if args.max_episodes:
            mp4_files = mp4_files[:args.max_episodes]

        print(f"\n=== {eval_dir.name}: {len(mp4_files)} rollouts ===")

        for mp4_path in mp4_files:
            json_path = mp4_path.with_suffix(".json")
            if not json_path.exists():
                print(f"  SKIP {mp4_path.name} (no JSON)")
                continue

            out_path = output_dir / mp4_path.name
            traj_data = json.load(open(json_path))
            result = "OK" if traj_data["success"] else "FAIL"
            print(f"  [{result}] {mp4_path.name}")
            process_rollout(mp4_path, json_path, out_path)

        print(f"  Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
