import os
import time
import threading
import copy
import numpy as np
import pybullet as p
import sys

import cv2
import psutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend so plots save without GUI conflicts
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure gym-pybullet-drones can be imported
sys.path.append(r'c:\Users\HOME\Documents\pybullet_test\gym-pybullet-drones')

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Load real YOLO model
try:
    from ultralytics import YOLO
    import torch
    print("Loading YOLOv8m model...")
    yolo_model = YOLO('yolov8m.pt')  # medium model — much better accuracy than nano
    if torch.cuda.is_available():
        yolo_model.to('cuda')
    print("Model loaded successfully!")
except ImportError:
    print("ERROR: ultralytics is not installed. Please run `pip install ultralytics`.")
    yolo_model = None

# ===============================
# REUSE LOGIC & CACHE CONFIG
# ===============================
PIXEL_CHANGE_THRESH = 40
REGION_CHANGE_RATIO = 0.25
FULL_REFRESH_INTERVAL = 50
STALE_FRAMES_REQUIRED = 3  # require N consecutive stale frames before invalidating

# Classes that make sense in aerial / top-down drone views
AERIAL_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'train',
    'traffic light', 'stop sign', 'parking meter', 'fire hydrant',
    'dog', 'cat', 'horse', 'cow', 'sheep', 'bird',
    'bench', 'umbrella', 'backpack', 'suitcase',
}
CONF_THRESHOLD = 0.35  # minimum confidence to keep a detection

def run_yolo_detection(frame, yolo_model):
    """Run YOLO on the full frame. Returns list of (x1, y1, x2, y2, class_name).
    Filters out low-confidence and non-aerial classes."""
    if yolo_model is None:
        return []
    results = yolo_model(frame, verbose=False, imgsz=640)
    dets = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf)
            if conf < CONF_THRESHOLD:
                continue
            cls_name = yolo_model.names[int(box.cls)]
            if cls_name not in AERIAL_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append((x1, y1, x2, y2, cls_name))
    return dets

def extract_region(gray, x1, y1, x2, y2, h, w):
    return gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()

def region_changed(prev_crop, curr_gray, x1, y1, x2, y2, h, w):
    curr_crop = curr_gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if prev_crop.shape != curr_crop.shape or prev_crop.size == 0:
        return True
    diff = cv2.absdiff(prev_crop, curr_crop)
    changed = np.count_nonzero(diff > PIXEL_CHANGE_THRESH)
    return (changed / max(diff.size, 1)) > REGION_CHANGE_RATIO

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / max(union, 1)

def compute_detection_accuracy(reused_dets, ground_truth_dets, iou_threshold=0.5):
    """Compare reused detections against full YOLO ground truth.
    Returns (precision, recall, f1).
    """
    if not ground_truth_dets and not reused_dets:
        return 1.0, 1.0, 1.0  # both empty = perfect agreement
    if not ground_truth_dets:
        return 0.0, 1.0, 0.0  # reused found things GT didn't — false positives
    if not reused_dets:
        return 1.0, 0.0, 0.0  # reused missed everything

    matched_gt = set()
    true_positives = 0

    for det in reused_dets:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth_dets):
            if gt_idx in matched_gt:
                continue
            # Only match same class
            if det[4] != gt[4]:
                continue
            iou = compute_iou(det[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    precision = true_positives / len(reused_dets) if reused_dets else 0.0
    recall = true_positives / len(ground_truth_dets) if ground_truth_dets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def capture_drone_frame(drone_id):
    """Captures and returns a BGR and grayscale image from the drone."""
    pos, quat = p.getBasePositionAndOrientation(drone_id)
    eye_pos = pos
    target_pos = [pos[0], pos[1], pos[2] - 1.0]
    up_vector = [1, 0, 0]
    view_matrix = p.computeViewMatrix(eye_pos, target_pos, up_vector)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=4/3, nearVal=0.1, farVal=100.0)
    width, height, rgb_img, _, _ = p.getCameraImage(
        width=640, height=480,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER
    )
    rgb_array = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))
    bgr_frame = cv2.cvtColor(rgb_array[:, :, :3], cv2.COLOR_RGB2BGR)
    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return bgr_frame, gray_frame


# ===============================
# COMPARISON GRAPH GENERATION
# ===============================
def generate_comparison_graphs(tel_reuse, tel_no_reuse, graphs_dir):
    """Generate and save comparison graphs. Called periodically so graphs are
    always available even if the simulation never ends."""
    n = min(len(tel_reuse['latency_ms']), len(tel_no_reuse['latency_ms']))
    if n < 2:
        return

    os.makedirs(graphs_dir, exist_ok=True)
    frames = list(range(n))

    # Trim to same length
    tr = {k: v[:n] for k, v in tel_reuse.items()}
    tn = {k: v[:n] for k, v in tel_no_reuse.items()}

    avg_cpu_r = np.mean(tr['cpu_percent'])
    avg_cpu_n = np.mean(tn['cpu_percent'])
    avg_lat_r = np.mean(tr['latency_ms'])
    avg_lat_n = np.mean(tn['latency_ms'])
    total_e_r = tr['energy_joules'][-1]
    total_e_n = tn['energy_joules'][-1]

    # Accuracy metrics (only present in tel_reuse)
    has_accuracy = 'precision' in tr and len(tr['precision']) >= n
    if has_accuracy:
        avg_prec = np.mean(tr['precision'])
        avg_rec = np.mean(tr['recall'])
        avg_f1 = np.mean(tr['f1_score'])

    common_legend = [
        Line2D([0], [0], color='#2ecc71', lw=2, label='With Reuse'),
        Line2D([0], [0], color='#e74c3c', lw=2, label='Without Reuse'),
    ]

    try:
        # --- 1. CPU Usage ---
        fig1, ax1 = plt.subplots(figsize=(13, 5))
        ax1.plot(frames, tr['cpu_percent'], color='#2ecc71', linewidth=1.5, alpha=0.85, label='With Reuse')
        ax1.plot(frames, tn['cpu_percent'], color='#e74c3c', linewidth=1.5, alpha=0.85, label='Without Reuse')
        ax1.axhline(y=avg_cpu_r, color='#27ae60', linestyle='--', linewidth=1, alpha=0.7)
        ax1.axhline(y=avg_cpu_n, color='#c0392b', linestyle='--', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Frame Number', fontsize=12)
        ax1.set_ylabel('CPU Usage (%)', fontsize=12)
        ax1.set_title('CPU Usage: With vs Without Computation Reuse', fontsize=14, fontweight='bold')
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3)
        ax1.legend(handles=[
            Line2D([0], [0], color='#2ecc71', lw=2, label=f'With Reuse (avg: {avg_cpu_r:.1f}%)'),
            Line2D([0], [0], color='#e74c3c', lw=2, label=f'Without Reuse (avg: {avg_cpu_n:.1f}%)'),
        ], loc='upper right', fontsize=10)
        fig1.tight_layout()
        fig1.savefig(os.path.join(graphs_dir, 'comparison_cpu_usage.png'), dpi=150)
        plt.close(fig1)

        # --- 2. Cumulative Energy ---
        fig2, ax2 = plt.subplots(figsize=(13, 5))
        ax2.fill_between(frames, tr['energy_joules'], color='#2ecc71', alpha=0.15)
        ax2.fill_between(frames, tn['energy_joules'], color='#e74c3c', alpha=0.15)
        ax2.plot(frames, tr['energy_joules'], color='#27ae60', linewidth=2)
        ax2.plot(frames, tn['energy_joules'], color='#c0392b', linewidth=2)
        ax2.set_xlabel('Frame Number', fontsize=12)
        ax2.set_ylabel('Estimated Energy (Joules)', fontsize=12)
        ax2.set_title('Cumulative Energy Consumption: With vs Without Computation Reuse', fontsize=14, fontweight='bold')
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
        ax2.legend(handles=[
            Line2D([0], [0], color='#27ae60', lw=2, label=f'With Reuse (total: {total_e_r:.2f} J)'),
            Line2D([0], [0], color='#c0392b', lw=2, label=f'Without Reuse (total: {total_e_n:.2f} J)'),
        ], loc='upper left', fontsize=10)
        fig2.tight_layout()
        fig2.savefig(os.path.join(graphs_dir, 'comparison_energy.png'), dpi=150)
        plt.close(fig2)

        # --- 3. Latency (grouped bars) ---
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        bar_w = 0.4
        x_pos = np.arange(n)
        ax3.bar(x_pos - bar_w/2, tr['latency_ms'], width=bar_w, color='#2ecc71', alpha=0.8, label='With Reuse')
        ax3.bar(x_pos + bar_w/2, tn['latency_ms'], width=bar_w, color='#e74c3c', alpha=0.8, label='Without Reuse')
        ax3.axhline(y=avg_lat_r, color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.8)
        ax3.axhline(y=avg_lat_n, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.8)
        ax3.set_xlabel('Frame Number', fontsize=12)
        ax3.set_ylabel('Latency (ms)', fontsize=12)
        ax3.set_title('Per-Frame Latency: With vs Without Computation Reuse', fontsize=14, fontweight='bold')
        ax3.set_ylim(bottom=0)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(handles=[
            Line2D([0], [0], color='#2ecc71', lw=8, label=f'With Reuse (avg: {avg_lat_r:.1f} ms)'),
            Line2D([0], [0], color='#e74c3c', lw=8, label=f'Without Reuse (avg: {avg_lat_n:.1f} ms)'),
        ], loc='upper right', fontsize=10)
        fig3.tight_layout()
        fig3.savefig(os.path.join(graphs_dir, 'comparison_latency.png'), dpi=150)
        plt.close(fig3)

        # --- 4. Dashboard (3 panels) ---
        fig4, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        fig4.suptitle('Performance Comparison: With vs Without Computation Reuse', fontsize=16, fontweight='bold', y=0.98)

        axes[0].plot(frames, tr['cpu_percent'], color='#2ecc71', linewidth=1.5, alpha=0.85)
        axes[0].plot(frames, tn['cpu_percent'], color='#e74c3c', linewidth=1.5, alpha=0.85)
        axes[0].set_ylabel('CPU Usage (%)', fontsize=11)
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(handles=common_legend, loc='upper right', fontsize=9)

        axes[1].fill_between(frames, tr['energy_joules'], color='#2ecc71', alpha=0.15)
        axes[1].fill_between(frames, tn['energy_joules'], color='#e74c3c', alpha=0.15)
        axes[1].plot(frames, tr['energy_joules'], color='#27ae60', linewidth=2)
        axes[1].plot(frames, tn['energy_joules'], color='#c0392b', linewidth=2)
        axes[1].set_ylabel('Energy (J)', fontsize=11)
        axes[1].set_ylim(bottom=0)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(handles=common_legend, loc='upper left', fontsize=9)

        axes[2].bar(x_pos - bar_w/2, tr['latency_ms'], width=bar_w, color='#2ecc71', alpha=0.8)
        axes[2].bar(x_pos + bar_w/2, tn['latency_ms'], width=bar_w, color='#e74c3c', alpha=0.8)
        axes[2].set_ylabel('Latency (ms)', fontsize=11)
        axes[2].set_xlabel('Frame Number', fontsize=12)
        axes[2].set_ylim(bottom=0)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].legend(handles=common_legend, loc='upper right', fontsize=9)

        fig4.tight_layout(rect=[0, 0, 1, 0.96])
        fig4.savefig(os.path.join(graphs_dir, 'comparison_dashboard.png'), dpi=150)
        plt.close(fig4)

        # --- 5. Summary Bar Chart ---
        fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
        fig5.suptitle('Average Performance: With vs Without Computation Reuse', fontsize=14, fontweight='bold')

        cats = ['With Reuse', 'Without Reuse']
        bc = ['#2ecc71', '#e74c3c']

        vals_cpu = [avg_cpu_r, avg_cpu_n]
        bars0 = axes5[0].bar(cats, vals_cpu, color=bc, alpha=0.85, edgecolor='white', linewidth=1.5)
        axes5[0].set_ylabel('CPU Usage (%)')
        axes5[0].set_title('Average CPU Usage', fontweight='bold')
        axes5[0].set_ylim(bottom=0)
        axes5[0].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars0, vals_cpu):
            axes5[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

        vals_lat = [avg_lat_r, avg_lat_n]
        bars1 = axes5[1].bar(cats, vals_lat, color=bc, alpha=0.85, edgecolor='white', linewidth=1.5)
        axes5[1].set_ylabel('Latency (ms)')
        axes5[1].set_title('Average Latency', fontweight='bold')
        axes5[1].set_ylim(bottom=0)
        axes5[1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars1, vals_lat):
            axes5[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f'{val:.1f} ms', ha='center', va='bottom', fontweight='bold')

        vals_e = [total_e_r, total_e_n]
        bars2 = axes5[2].bar(cats, vals_e, color=bc, alpha=0.85, edgecolor='white', linewidth=1.5)
        axes5[2].set_ylabel('Energy (Joules)')
        axes5[2].set_title('Total Energy Consumed', fontweight='bold')
        axes5[2].set_ylim(bottom=0)
        axes5[2].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars2, vals_e):
            axes5[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{val:.2f} J', ha='center', va='bottom', fontweight='bold')

        fig5.tight_layout(rect=[0, 0, 1, 0.93])
        fig5.savefig(os.path.join(graphs_dir, 'comparison_summary.png'), dpi=150)
        plt.close(fig5)

        # --- 6. Accuracy Over Time (precision, recall, F1) ---
        if has_accuracy:
            fig6, ax6 = plt.subplots(figsize=(13, 5))
            ax6.plot(frames, tr['precision'][:n], color='#3498db', linewidth=1.5, alpha=0.85, label=f'Precision (avg: {avg_prec:.2f})')
            ax6.plot(frames, tr['recall'][:n], color='#e67e22', linewidth=1.5, alpha=0.85, label=f'Recall (avg: {avg_rec:.2f})')
            ax6.plot(frames, tr['f1_score'][:n], color='#9b59b6', linewidth=2.0, alpha=0.90, label=f'F1 Score (avg: {avg_f1:.2f})')
            ax6.axhline(y=avg_f1, color='#8e44ad', linestyle='--', linewidth=1, alpha=0.6)
            ax6.set_xlabel('Frame Number', fontsize=12)
            ax6.set_ylabel('Score', fontsize=12)
            ax6.set_title('Detection Accuracy: Reused Detections vs Full YOLO Ground Truth', fontsize=14, fontweight='bold')
            ax6.set_ylim(-0.05, 1.05)
            ax6.grid(True, alpha=0.3)
            ax6.legend(loc='lower right', fontsize=10)
            fig6.tight_layout()
            fig6.savefig(os.path.join(graphs_dir, 'accuracy_over_time.png'), dpi=150)
            plt.close(fig6)

            # --- 7. Accuracy vs Latency Tradeoff ---
            fig7, ax7 = plt.subplots(figsize=(8, 6))
            lat_savings = [nr - r for r, nr in zip(tr['latency_ms'], tn['latency_ms'])]
            f1_scores = tr['f1_score'][:n]
            # Color by YOLO status: green = SKIPPED (reused), red = RAN
            colors = ['#2ecc71' if s == 'SKIPPED' else '#e74c3c' for s in tr['yolo_status'][:n]]
            ax7.scatter(lat_savings, f1_scores, c=colors, alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
            ax7.set_xlabel('Latency Saved (ms) — higher is better', fontsize=12)
            ax7.set_ylabel('F1 Score — higher is better', fontsize=12)
            ax7.set_title('Accuracy vs Performance Tradeoff', fontsize=14, fontweight='bold')
            ax7.set_ylim(-0.05, 1.05)
            ax7.grid(True, alpha=0.3)
            ax7.legend(handles=[
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='YOLO Skipped (reused)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='YOLO Ran'),
            ], loc='lower right', fontsize=10)
            fig7.tight_layout()
            fig7.savefig(os.path.join(graphs_dir, 'accuracy_vs_latency.png'), dpi=150)
            plt.close(fig7)

        print(f"[GRAPHS] Comparison graphs updated in: {graphs_dir}")

    except Exception as e:
        print(f"[GRAPHS] Error generating graphs: {e}")
        import traceback
        traceback.print_exc()


# ===============================
# MAIN SIMULATION
# ===============================
def run_simulation():
    from pathlib import Path

    possible_paths = [
        Path(os.getcwd()) / "VisDrone" / "data",
        Path.home() / "fiftyone" / "visdrone-2019" / "validation" / "data"
    ]

    image_files = []
    for base_path in possible_paths:
        print(f"[DEBUG] Checking if directory exists: {base_path}")
        if base_path.exists():
            print(f"[DEBUG] Directory exists! Scanning for images...")
            for f in base_path.iterdir():
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_files.append(str(f))
            if image_files:
                break
        else:
            print(f"[DEBUG] Directory does not exist: {base_path}")
            parts = list(base_path.parts)
            for i in range(1, len(parts)+1):
                sub_path = Path(*parts[:i])
                if not sub_path.exists():
                    print(f"[DEBUG] Broken at: {sub_path}")
                    break

    if not image_files:
        print("No images found in any checked directories.")
        return

    image_files = image_files[:30]
    num_images = len(image_files)
    print(f"Loaded {num_images} images from the VisDrone dataset.")

    # Configuration
    NUM_DRONES = 1
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION_SEC = max(20, num_images * 2)
    INIT_XYZS = np.array([[-1.0, 0, 0.8]])

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True,
        record=False
    )

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(NUM_DRONES)]
    obs, info = env.reset()

    video_dir = os.path.join(os.getcwd(), "captured_videos")
    os.makedirs(video_dir, exist_ok=True)
    graphs_dir = os.path.join(os.getcwd(), "performance_graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # --- TEMPORAL REUSE STATE ---
    prev_gray = None
    cached_detections = []
    cached_regions = []
    stale_counts = []  # per-detection consecutive stale frame counter
    frame_id = 0
    capture_every_n_steps = 5  # capture every N control steps (CTRL_FREQ/5 ≈ 10 Hz)

    # --- VIDEO WRITER SETUP ---
    video_path = os.path.join(video_dir, "drone_topdown_recording.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = 1.5  # slow playback so each frame is visible (~0.5s per frame)
    video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (640, 480))
    if not video_writer.isOpened():
        print("WARNING: mp4v codec failed. Trying XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = video_path.replace('.mp4', '.avi')
        video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (640, 480))
    print(f"Recording video to: {video_path} at {video_fps:.1f} FPS")

    # --- TELEMETRY: two parallel records ---
    # "with reuse" telemetry records what actually happens (reuse logic)
    # "without reuse" telemetry records YOLO-every-frame baseline measured on the same frames
    tel_reuse = {
        'cpu_percent': [], 'latency_ms': [], 'energy_joules': [], 'yolo_status': [],
        'precision': [], 'recall': [], 'f1_score': [],
    }
    tel_no_reuse = {
        'cpu_percent': [], 'latency_ms': [], 'energy_joules': [], 'yolo_status': [],
    }
    cum_energy_reuse = 0.0
    cum_energy_no_reuse = 0.0
    TDP_WATTS = 65.0
    process = psutil.Process(os.getpid())

    GRAPH_UPDATE_INTERVAL = 30  # Update graphs every N frames
    _graph_thread = None  # background thread for graph generation

    # Generate OBJ plane
    temp_folder = os.path.dirname(__file__)
    obj_path = os.path.join(temp_folder, "perfect_uv_plane.obj")
    with open(obj_path, "w") as f:
        f.write("v -0.5 -0.5 0.0\n")
        f.write("v 0.5 -0.5 0.0\n")
        f.write("v 0.5 0.5 0.0\n")
        f.write("v -0.5 0.5 0.0\n")
        f.write("vt 0.0 0.0\n")
        f.write("vt 1.0 0.0\n")
        f.write("vt 1.0 1.0\n")
        f.write("vt 0.0 1.0\n")
        f.write("vn 0.0 0.0 1.0\n")
        f.write("f 1/1/1 2/2/1 3/3/1\n")
        f.write("f 1/1/1 3/3/1 4/4/1\n")

    img_xs = [i * 0.7 for i in range(num_images)]
    max_x = img_xs[-1] + 1.0  # stop shortly after last image

    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        try:
            from PIL import Image
            temp_png_path = os.path.join(temp_folder, f"temp_visdrone_{i}.png")
            with Image.open(img_path) as img_pil:
                img_resized = img_pil.resize((512, 512))
                img_resized.save(temp_png_path, format="PNG")
            tex_id = p.loadTexture(temp_png_path)
            if tex_id == -1:
                print(f"PyBullet failed to load texture: {temp_png_path}")
            x = img_xs[i]
            z = 0.01 + (i * 0.001)
            vis_id = p.createVisualShape(
                shapeType=p.GEOM_MESH, fileName=obj_path,
                meshScale=[0.8, 0.8, 1.0], rgbaColor=[1, 1, 1, 1]
            )
            body_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis_id,
                basePosition=[x, 0.0, z]
            )
            if tex_id != -1:
                p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
        except Exception as e:
            print(f"Skipping {img_name} due to texture loading error: {e}")

    if p.isConnected():
        p.resetDebugVisualizerCamera(
            cameraDistance=5, cameraYaw=0, cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0]
        )

    steps = DURATION_SEC * CTRL_FREQ
    speed = 2.0  # faster drone traversal
    z_height = 0.8

    print("Starting simulation — measuring WITH and WITHOUT reuse on each frame...")
    print(f"Graphs will update every {GRAPH_UPDATE_INTERVAL} frames in: {graphs_dir}")

    try:
        for i in range(steps):
            time_sec = i / CTRL_FREQ

            action = np.zeros((NUM_DRONES, 4))
            x = speed * time_sec - 1.0
            if x > max_x:
                print(f"Drone passed last image (x={x:.1f} > {max_x:.1f}). Ending simulation.")
                break
            target_pos = np.array([x, 0.0, z_height])
            state = obs[0]

            rpm, _, _ = ctrl[0].computeControlFromState(
                control_timestep=1/CTRL_FREQ, state=state,
                target_pos=target_pos, target_rpy=np.array([0, 0, 0])
            )
            action[0] = rpm

            if i % capture_every_n_steps == 0:
                bgr_frame, gray_frame = capture_drone_frame(env.DRONE_IDS[0])
                h, w = gray_frame.shape

                # =============================================================
                # MEASURE 1: "WITHOUT REUSE" — always run YOLO, time it
                # =============================================================
                t0_no_reuse = time.perf_counter()
                ct_before_no = process.cpu_times()
                baseline_dets = run_yolo_detection(bgr_frame, yolo_model)
                t1_no_reuse = time.perf_counter()
                ct_after_no = process.cpu_times()

                elapsed_no = t1_no_reuse - t0_no_reuse
                lat_no_reuse = elapsed_no * 1000.0
                cpu_used_no = (ct_after_no.user - ct_before_no.user) + (ct_after_no.system - ct_before_no.system)
                cpu_no_reuse = (cpu_used_no / elapsed_no) * 100.0 if elapsed_no > 0 else 0.0
                energy_no = (cpu_no_reuse / 100.0) * TDP_WATTS * elapsed_no
                cum_energy_no_reuse += energy_no

                tel_no_reuse['cpu_percent'].append(cpu_no_reuse)
                tel_no_reuse['latency_ms'].append(lat_no_reuse)
                tel_no_reuse['energy_joules'].append(cum_energy_no_reuse)
                tel_no_reuse['yolo_status'].append('RAN')

                # =============================================================
                # MEASURE 2: "WITH REUSE" — use temporal reuse logic, time it
                # =============================================================
                t0_reuse = time.perf_counter()
                ct_before_r = process.cpu_times()

                force_full = (frame_id % FULL_REFRESH_INTERVAL == 0)
                reused_flags = []

                if prev_gray is None or force_full:
                    # Must run YOLO — reuse the result we already computed above
                    cached_detections = baseline_dets
                    cached_regions = [extract_region(gray_frame, d[0], d[1], d[2], d[3], h, w) for d in cached_detections]
                    stale_counts = [0] * len(cached_detections)
                    reused_flags = [False] * len(cached_detections)
                else:
                    # Phase correlation based reuse logic
                    prev_f = np.float32(prev_gray)
                    curr_f = np.float32(gray_frame)
                    shift, _ = cv2.phaseCorrelate(prev_f, curr_f)
                    dx, dy = int(round(shift[0])), int(round(shift[1]))

                    still_valid = []
                    truly_stale = []
                    shifted_cached_detections = []
                    new_stale_counts = []

                    for k, (det, crop) in enumerate(zip(cached_detections, cached_regions)):
                        x1, y1, x2, y2, cls_name = det
                        nx1, ny1 = max(0, x1 + dx), max(0, y1 + dy)
                        nx2, ny2 = min(w, x2 + dx), min(h, y2 + dy)
                        shifted_det = (nx1, ny1, nx2, ny2, cls_name)
                        shifted_cached_detections.append(shifted_det)

                        if region_changed(crop, gray_frame, nx1, ny1, nx2, ny2, h, w):
                            count = stale_counts[k] + 1 if k < len(stale_counts) else 1
                            new_stale_counts.append(count)
                            if count >= STALE_FRAMES_REQUIRED:
                                truly_stale.append(k)
                            else:
                                still_valid.append(k)  # not stale enough yet, keep as reused
                        else:
                            new_stale_counts.append(0)  # reset counter on stable frame
                            still_valid.append(k)

                    cached_detections = shifted_cached_detections
                    stale_counts = new_stale_counts

                    if not truly_stale and len(cached_detections) > 0:
                        reused_flags = [True] * len(cached_detections)
                        cached_regions = [extract_region(gray_frame, d[0], d[1], d[2], d[3], h, w) for d in cached_detections]
                    else:
                        # Reuse the YOLO result from baseline measurement
                        new_dets = baseline_dets
                        final_dets = []
                        final_flags = []
                        final_stale = []

                        for k_idx in still_valid:
                            final_dets.append(cached_detections[k_idx])
                            final_flags.append(True)
                            final_stale.append(stale_counts[k_idx])

                        # Match new YOLO detections against ALL previous cached
                        # detections. Re-detected objects are kept as REUSED,
                        # genuinely new objects are marked NEW.
                        IOU_THRESH = 0.2
                        all_prev_boxes = cached_detections  # includes stale + valid
                        for new_box in new_dets:
                            b1 = new_box[:4]
                            cls_new = new_box[4]
                            best_iou = max(
                                (compute_iou(b1, prev[:4]) for prev in all_prev_boxes if prev[4] == cls_new),
                                default=0.0
                            )
                            if best_iou > IOU_THRESH:
                                # Re-detected — mark as REUSED (was already known)
                                final_dets.append(new_box)
                                final_flags.append(True)
                                final_stale.append(0)
                            else:
                                # Genuinely new detection
                                final_dets.append(new_box)
                                final_flags.append(False)
                                final_stale.append(0)

                        # Deduplicate: if a reused box and a new YOLO box overlap
                        # heavily, keep the YOLO box (more accurate position)
                        deduped_dets = []
                        deduped_flags = []
                        deduped_stale = []
                        for idx, (det, flag, sc) in enumerate(zip(final_dets, final_flags, final_stale)):
                            is_dup = False
                            for idx2 in range(idx):
                                if compute_iou(det[:4], final_dets[idx2][:4]) > 0.4 and det[4] == final_dets[idx2][4]:
                                    is_dup = True
                                    break
                            if not is_dup:
                                deduped_dets.append(det)
                                deduped_flags.append(flag)
                                deduped_stale.append(sc)

                        cached_detections = deduped_dets
                        cached_regions = [extract_region(gray_frame, d[0], d[1], d[2], d[3], h, w) for d in cached_detections]
                        stale_counts = deduped_stale
                        reused_flags = deduped_flags

                t1_reuse = time.perf_counter()
                ct_after_r = process.cpu_times()

                yolo_status = "SKIPPED" if all(reused_flags) and len(reused_flags) > 0 else "RAN"

                # CPU time consumed by the reuse/merge logic only
                elapsed_reuse_only = t1_reuse - t0_reuse
                cpu_used_r = (ct_after_r.user - ct_before_r.user) + (ct_after_r.system - ct_before_r.system)
                cpu_reuse_only = (cpu_used_r / elapsed_reuse_only) * 100.0 if elapsed_reuse_only > 0 else 0.0

                if yolo_status == "SKIPPED":
                    # Only reuse logic ran — no YOLO cost
                    lat_reuse = elapsed_reuse_only * 1000.0
                    cpu_reuse = cpu_reuse_only
                    energy_r = (cpu_reuse / 100.0) * TDP_WATTS * elapsed_reuse_only
                else:
                    # YOLO + merge: combine both stages correctly
                    lat_reuse = lat_no_reuse + elapsed_reuse_only * 1000.0
                    total_elapsed = elapsed_no + elapsed_reuse_only
                    # Weighted-average CPU% across YOLO and merge stages
                    cpu_reuse = (cpu_no_reuse * elapsed_no + cpu_reuse_only * elapsed_reuse_only) / total_elapsed if total_elapsed > 0 else 0.0
                    # Energy = YOLO energy + merge overhead energy (no double-count)
                    energy_r = energy_no + (cpu_reuse_only / 100.0) * TDP_WATTS * elapsed_reuse_only

                cum_energy_reuse += energy_r

                tel_reuse['cpu_percent'].append(cpu_reuse)
                tel_reuse['latency_ms'].append(lat_reuse)
                tel_reuse['energy_joules'].append(cum_energy_reuse)
                tel_reuse['yolo_status'].append(yolo_status)

                # --- ACCURACY VALIDATION: compare reused dets vs full YOLO ---
                precision, recall, f1 = compute_detection_accuracy(
                    cached_detections, baseline_dets
                )
                tel_reuse['precision'].append(precision)
                tel_reuse['recall'].append(recall)
                tel_reuse['f1_score'].append(f1)

                # Draw bounding boxes on display frame
                COLOR_NEW = (0, 255, 0)
                COLOR_REUSED = (0, 165, 255)
                for det, reused in zip(cached_detections, reused_flags):
                    x1, y1, x2, y2, cls_name = det
                    color = COLOR_REUSED if reused else COLOR_NEW
                    label = f"REUSED {cls_name}" if reused else f"NEW {cls_name}"
                    cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(bgr_frame, label, (x1, max(y1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.putText(bgr_frame, f"Dets: {len(cached_detections)} | YOLO: {yolo_status}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(bgr_frame, f"Reuse: {lat_reuse:.0f}ms | NoReuse: {lat_no_reuse:.0f}ms",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 2)
                cv2.putText(bgr_frame, f"P:{precision:.2f} R:{recall:.2f} F1:{f1:.2f}",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 128), 2)

                # Write frame to video
                video_writer.write(bgr_frame)

                n_reused = sum(reused_flags)
                n_new = len(reused_flags) - n_reused
                print(f"F{frame_id:04d} | {yolo_status:7s} | R:{lat_reuse:.0f}ms NR:{lat_no_reuse:.0f}ms | dets:{len(cached_detections)} (reused:{n_reused} new:{n_new}) | P:{precision:.2f} R:{recall:.2f} F1:{f1:.2f}")

                # --- PERIODIC GRAPH UPDATE (background thread) ---
                if frame_id > 0 and frame_id % GRAPH_UPDATE_INTERVAL == 0:
                    if _graph_thread is None or not _graph_thread.is_alive():
                        snapshot_reuse = copy.deepcopy(tel_reuse)
                        snapshot_no_reuse = copy.deepcopy(tel_no_reuse)
                        _graph_thread = threading.Thread(
                            target=generate_comparison_graphs,
                            args=(snapshot_reuse, snapshot_no_reuse, graphs_dir),
                            daemon=True
                        )
                        _graph_thread.start()

                prev_gray = gray_frame
                frame_id += 1

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        # Release video writer
        video_writer.release()
        print(f"\nVideo saved to: {video_path}")
        env.close()

        # Generate final graphs
        print("\nGenerating final comparison graphs...")
        generate_comparison_graphs(tel_reuse, tel_no_reuse, graphs_dir)

        n = len(tel_reuse['latency_ms'])
        if n > 0:
            print(f"\n{'='*55}")
            print(f"  RESULTS OVER {n} FRAMES")
            print(f"{'='*55}")
            print(f"  {'Metric':<25} {'With Reuse':>12} {'Without Reuse':>14}")
            print(f"  {'-'*51}")
            print(f"  {'Avg CPU (%)' :<25} {np.mean(tel_reuse['cpu_percent']):>12.1f} {np.mean(tel_no_reuse['cpu_percent']):>14.1f}")
            print(f"  {'Avg Latency (ms)':<25} {np.mean(tel_reuse['latency_ms']):>12.1f} {np.mean(tel_no_reuse['latency_ms']):>14.1f}")
            print(f"  {'Total Energy (J)':<25} {tel_reuse['energy_joules'][-1]:>12.2f} {tel_no_reuse['energy_joules'][-1]:>14.2f}")
            print(f"  {'-'*51}")
            if tel_reuse['precision']:
                print(f"\n  {'Reuse Accuracy':<25} {'Value':>12}")
                print(f"  {'-'*37}")
                print(f"  {'Avg Precision':<25} {np.mean(tel_reuse['precision']):>12.2f}")
                print(f"  {'Avg Recall':<25} {np.mean(tel_reuse['recall']):>12.2f}")
                print(f"  {'Avg F1 Score':<25} {np.mean(tel_reuse['f1_score']):>12.2f}")
                print(f"  {'Min F1 Score':<25} {np.min(tel_reuse['f1_score']):>12.2f}")
                print(f"  {'-'*37}")
            print(f"\n  Graphs saved to: {graphs_dir}")
        print("Simulation finished.")


if __name__ == "__main__":
    run_simulation()
