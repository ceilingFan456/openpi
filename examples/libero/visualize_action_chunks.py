"""
Visualize eval rollouts with action chunk boundaries.

Creates a 2-panel video:
  [ Third-Person View | 3D Trajectory with Action Chunk Markers ]

The trajectory panel shows:
  - Gray line: full trajectory
  - Blue line: traversed path up to current frame
  - Red dot: current position
  - Green dot: start position
  - Orange 'X' markers: end of each action chunk (where the chunk's predictions end)
  - Dashed orange segments: predicted-but-not-executed portion of each chunk
  - Vertical dashed lines on the timeline bar showing chunk boundaries

Usage:
    python visualize_action_chunks.py --eval-dir data/libero/3task_eval/e1_original_local_0429_step4999
    python visualize_action_chunks.py --eval-dir data/libero/3task_eval/e1_original_local_0429_step4999 --max-episodes 3
"""

import argparse
import io
import json
import subprocess
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_trajectory_data(json_path):
    with open(json_path) as f:
        return json.load(f)


def fig_to_array(fig, target_h, target_w):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=120)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img)


def add_label(frame, text, position=(5, 5), font_size=14, color=(255, 255, 255)):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    x, y = position
    # Black outline
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=color, font=font)
    return np.array(img)


def write_video_ffmpeg(frames, output_path, fps=10):
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait(timeout=120)


def compute_chunk_endpoints(traj_data):
    """
    For each action chunk, compute where the full predicted chunk would end
    in the EE position space.

    Each chunk predicts `action_horizon` actions (e.g., 10), but only
    `replan_steps` (e.g., 5) are executed. We compute the cumulative
    displacement from the chunk's start to show where the full prediction
    reaches.

    Returns a list of dicts with:
      - start_step: step index where chunk was queried
      - exec_end_step: step index where executed portion ends
      - horizon_end_step: step index where full prediction would end
      - chunk_idx: which chunk number this is
    """
    boundaries = traj_data["chunk_boundaries"]
    replan = traj_data["replan_steps"]
    horizon = traj_data["action_horizon"]
    n_steps = traj_data["num_steps"]

    chunk_info = []
    for i, start in enumerate(boundaries):
        exec_end = min(start + replan, n_steps - 1)
        horizon_end = min(start + horizon, n_steps - 1)
        chunk_info.append({
            "start_step": start,
            "exec_end_step": exec_end,
            "horizon_end_step": horizon_end,
            "chunk_idx": i,
        })
    return chunk_info


def render_trajectory_frame_with_chunks(ax, traj, frame_idx, chunk_info,
                                         xlim, ylim, zlim, replan_steps, action_horizon):
    """Render one frame of the 3D trajectory with action chunk markers."""
    ax.cla()

    # Full trajectory (faint gray)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            color="lightgray", linewidth=1.0, alpha=0.5)

    # Traversed path (blue)
    if frame_idx > 0:
        ax.plot(traj[:frame_idx + 1, 0],
                traj[:frame_idx + 1, 1],
                traj[:frame_idx + 1, 2],
                color="#2171b5", linewidth=2.0, alpha=0.9)

    # Mark action chunk boundaries and horizons
    for ci in chunk_info:
        start = ci["start_step"]
        horizon_end = ci["horizon_end_step"]

        # Only show chunks that have been triggered by current frame
        if start > frame_idx:
            continue

        # Mark where chunk was queried (small blue diamond)
        if start < len(traj):
            ax.scatter([traj[start, 0]], [traj[start, 1]], [traj[start, 2]],
                       color="#2171b5", s=30, marker="D", zorder=8, depthshade=False, alpha=0.6)

        # Mark where the full action horizon ends (orange X)
        if horizon_end < len(traj):
            ax.scatter([traj[horizon_end, 0]], [traj[horizon_end, 1]], [traj[horizon_end, 2]],
                       color="#e6550d", s=50, marker="X", zorder=9, depthshade=False, alpha=0.8)

        # Draw dashed line from exec_end to horizon_end (predicted but not executed)
        exec_end = ci["exec_end_step"]
        if exec_end < horizon_end and exec_end < len(traj) and horizon_end < len(traj):
            seg = traj[exec_end:horizon_end + 1]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color="#e6550d", linewidth=1.5, linestyle="--", alpha=0.5)

    # Current point (red)
    ax.scatter([traj[frame_idx, 0]], [traj[frame_idx, 1]], [traj[frame_idx, 2]],
               color="red", s=80, zorder=10, depthshade=False)

    # Start marker (green)
    ax.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
               color="green", s=50, marker="o", zorder=9, depthshade=False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title("EE Trajectory + Action Chunks", fontsize=9)


def render_timeline_bar(frame_idx, n_frames, chunk_info, replan_steps, action_horizon, bar_w, bar_h=40):
    """Render a horizontal timeline bar showing current position and chunk boundaries."""
    img = Image.new("RGB", (bar_w, bar_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (IOError, OSError):
        font = ImageFont.load_default()

    margin = 10
    track_y = 15
    track_h = 8
    track_left = margin
    track_right = bar_w - margin
    track_w = track_right - track_left

    # Background track
    draw.rectangle([track_left, track_y, track_right, track_y + track_h], fill=(60, 60, 60))

    # Draw chunk regions
    for ci in chunk_info:
        start = ci["start_step"]
        horizon_end = ci["horizon_end_step"]
        exec_end = ci["exec_end_step"]

        # Executed portion (blue)
        x0 = track_left + int(start / max(n_frames - 1, 1) * track_w)
        x1 = track_left + int(exec_end / max(n_frames - 1, 1) * track_w)
        draw.rectangle([x0, track_y, x1, track_y + track_h], fill=(33, 113, 181))

        # Predicted but not executed (orange, thinner)
        x2 = track_left + int(horizon_end / max(n_frames - 1, 1) * track_w)
        if x2 > x1:
            draw.rectangle([x1, track_y + 2, x2, track_y + track_h - 2], fill=(230, 85, 13))

        # Chunk boundary tick
        draw.line([x0, track_y - 3, x0, track_y + track_h + 3], fill=(150, 150, 150), width=1)

    # Current position indicator
    cx = track_left + int(frame_idx / max(n_frames - 1, 1) * track_w)
    draw.rectangle([cx - 1, track_y - 5, cx + 1, track_y + track_h + 5], fill=(255, 50, 50))

    # Labels
    draw.text((track_left, track_y + track_h + 5), f"Step {frame_idx}/{n_frames - 1}", fill=(200, 200, 200), font=font)

    # Legend on right
    legend_x = track_right - 220
    draw.rectangle([legend_x, 2, legend_x + 8, 10], fill=(33, 113, 181))
    draw.text((legend_x + 12, 0), "executed", fill=(200, 200, 200), font=font)
    draw.rectangle([legend_x + 75, 2, legend_x + 83, 10], fill=(230, 85, 13))
    draw.text((legend_x + 87, 0), "predicted", fill=(200, 200, 200), font=font)
    draw.rectangle([legend_x + 155, 2, legend_x + 163, 10], fill=(150, 150, 150))
    draw.text((legend_x + 167, 0), "replan", fill=(200, 200, 200), font=font)

    return np.array(img)


def process_episode(video_path, json_path, output_path):
    """Generate a 2-panel + timeline video for one rollout."""
    traj_data = load_trajectory_data(json_path)
    ee_pos = np.array(traj_data["ee_positions"])
    chunk_info = compute_chunk_endpoints(traj_data)
    replan_steps = traj_data["replan_steps"]
    action_horizon = traj_data["action_horizon"]

    # Load video frames
    reader = imageio.get_reader(str(video_path))
    video_frames = [np.array(frame) for frame in reader]
    reader.close()
    n_frames = len(video_frames)

    # Ensure ee_pos length matches video frames
    if len(ee_pos) > n_frames:
        ee_pos = ee_pos[:n_frames]
    elif len(ee_pos) < n_frames:
        # Pad with last position
        pad = np.tile(ee_pos[-1:], (n_frames - len(ee_pos), 1))
        ee_pos = np.vstack([ee_pos, pad])

    frame_h, frame_w = video_frames[0].shape[:2]

    # Precompute trajectory axis limits
    margin = 0.05
    tmin = ee_pos.min(axis=0)
    tmax = ee_pos.max(axis=0)
    trange = tmax - tmin
    trange = np.where(trange < 1e-6, 0.01, trange)
    xlim = (tmin[0] - margin * trange[0], tmax[0] + margin * trange[0])
    ylim = (tmin[1] - margin * trange[1], tmax[1] + margin * trange[1])
    zlim = (tmin[2] - margin * trange[2], tmax[2] + margin * trange[2])

    # Set up trajectory figure
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    combined_w = frame_w + frame_h  # trajectory panel is square, same height as video
    combined_frames = []

    for i in range(n_frames):
        # Video frame
        vid_frame = video_frames[i].copy()
        vid_frame = add_label(vid_frame, "Agent View", position=(5, 5), font_size=12)

        # Determine which chunk we're in
        current_chunk = None
        for ci in chunk_info:
            if ci["start_step"] <= i:
                current_chunk = ci
        chunk_label = f"Chunk {current_chunk['chunk_idx']}" if current_chunk else ""
        steps_into_chunk = i - current_chunk["start_step"] if current_chunk else 0
        chunk_detail = f"{chunk_label} (step {steps_into_chunk}/{replan_steps})" if current_chunk else ""
        vid_frame = add_label(vid_frame, chunk_detail,
                             position=(5, frame_h - 20), font_size=11, color=(255, 200, 100))

        # Trajectory frame
        render_trajectory_frame_with_chunks(ax, ee_pos, i, chunk_info,
                                             xlim, ylim, zlim, replan_steps, action_horizon)
        traj_frame = fig_to_array(fig, frame_h, frame_h)  # square

        # Combine horizontally
        combined = np.concatenate([vid_frame, traj_frame], axis=1)

        # Add timeline bar at bottom
        timeline = render_timeline_bar(i, n_frames, chunk_info, replan_steps, action_horizon, combined.shape[1])
        combined = np.concatenate([combined, timeline], axis=0)

        # Task description at top
        task_desc = traj_data.get("task_description", "")
        success = "SUCCESS" if traj_data.get("success") else "FAILURE"
        combined = add_label(combined, f"{task_desc}  [{success}]",
                            position=(5, 2), font_size=11,
                            color=(100, 255, 100) if traj_data.get("success") else (255, 100, 100))

        combined_frames.append(combined)

    plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video_ffmpeg(combined_frames, output_path, fps=traj_data.get("fps", 10))
    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path.name} ({size_kb:.1f} KB, {n_frames} frames, "
          f"{len(chunk_info)} chunks, horizon={action_horizon}, replan={replan_steps})")


def main():
    parser = argparse.ArgumentParser(description="Visualize eval rollouts with action chunk boundaries")
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Path to eval output dir (e.g. data/libero/3task_eval/e1_...)")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Max episodes to process per task suite (0=all)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output dir for visualizations (default: {eval-dir}/chunk_visualizations)")
    parser.add_argument("--success-only", action="store_true",
                        help="Only process successful episodes")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    video_dir = eval_dir / "videos"
    if not video_dir.exists():
        print(f"No videos directory found at {video_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else eval_dir / "chunk_visualizations"

    # Find all json trajectory files
    json_files = sorted(video_dir.rglob("*.json"))
    if not json_files:
        print(f"No trajectory JSON files found in {video_dir}")
        print("Re-run evaluation with the updated main.py to generate trajectory data.")
        return

    print(f"Found {len(json_files)} trajectory files")

    # Group by suite
    suite_episodes = {}
    for jf in json_files:
        suite = jf.parent.name
        suite_episodes.setdefault(suite, []).append(jf)

    total = 0
    for suite, episodes in sorted(suite_episodes.items()):
        print(f"\n=== {suite} ({len(episodes)} episodes) ===")
        count = 0
        for jf in episodes:
            traj_data = load_trajectory_data(jf)
            if args.success_only and not traj_data.get("success"):
                continue

            video_path = jf.with_suffix(".mp4")
            if not video_path.exists():
                print(f"  SKIP: no video for {jf.name}")
                continue

            out_path = output_dir / suite / jf.with_suffix(".mp4").name
            print(f"  Processing: {jf.stem}...")
            process_episode(video_path, jf, out_path)
            count += 1
            total += 1

            if args.max_episodes and count >= args.max_episodes:
                break

    print(f"\nDone. Processed {total} episodes → {output_dir}")


if __name__ == "__main__":
    main()
