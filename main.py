import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict, deque

# --- CONFIG ---
VIDEO_PATH = 'video.mp4'
OUTPUT_PATH = 'output.mp4'
YOLO_MODEL = 'yolov8n.pt'
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
SIZE_THRESHOLDS = {'small': 4000, 'medium': 12000}  # bounding box area
TRAJECTORY_LENGTH = 30  # frames
GRAPH_WIDTH = 400

# --- UTILS ---
def get_vehicle_size(area):
    if area < SIZE_THRESHOLDS['small']:
        return 'small'
    elif area < SIZE_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'large'

def fig_to_img(fig, width, height):
    """Converts a Matplotlib figure into a BGR NumPy array directly in memory."""
    fig.canvas.draw()
    
    # 1. Use buffer_rgba() instead of tostring_rgb()
    rgba_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    
    # 2. Reshape using 4 channels (RGBA) instead of 3 (RGB)
    rgba_img = rgba_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # 3. Convert from RGBA to BGR for OpenCV
    bgr_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr_img, (width, height))

# --- MAIN ---
def main():
    # Load YOLO model
    model = YOLO(YOLO_MODEL)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width + GRAPH_WIDTH, height))

    # Tracking state
    trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_LENGTH))
    vehicle_sizes = defaultdict(str)
    vehicle_counts = []
    size_distributions = []
    total_vehicle_ids = set()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection and tracking
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
        boxes = results[0].boxes
        ids = results[0].boxes.id if boxes.id is not None else []
        cls = results[0].boxes.cls if boxes.cls is not None else []
        xyxy = results[0].boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))

        # Per-frame analytics
        frame_vehicle_count = 0
        size_count = {'small': 0, 'medium': 0, 'large': 0}

        for i, box in enumerate(xyxy):
            class_idx = int(cls[i]) if i < len(cls) else None
            class_name = model.names[class_idx] if class_idx is not None else None
            if class_name not in VEHICLE_CLASSES:
                continue
            vid = int(ids[i]) if i < len(ids) else None
            if vid is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            size = get_vehicle_size(area)
            vehicle_sizes[vid] = size
            size_count[size] += 1
            frame_vehicle_count += 1
            total_vehicle_ids.add(vid)

            # Draw bounding box
            color = (0, 255, 0) if size == 'small' else (0, 255, 255) if size == 'medium' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name} ID:{vid} {size}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update trajectory
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            trajectories[vid].append(center)

        # Draw trajectories
        for vid, points in trajectories.items():
            if len(points) > 1:
                for j in range(1, len(points)):
                    color = (0, 255, 0) if vehicle_sizes[vid] == 'small' else (0, 255, 255) if vehicle_sizes[vid] == 'medium' else (0, 0, 255)
                    cv2.line(frame, points[j - 1], points[j], color, 2)

        vehicle_counts.append(frame_vehicle_count)
        size_distributions.append(size_count.copy())

        # --- IN-MEMORY GRAPH DASHBOARD ---
        dashboard = np.ones((height, GRAPH_WIDTH, 3), dtype=np.uint8) * 240  # light gray
        pad = 20
        graph_w = GRAPH_WIDTH - 2 * pad
        graph_h = (height - 3 * pad) // 2

        # Traffic density line graph
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(vehicle_counts, color='blue', linewidth=2)
        ax1.set_title('Traffic Density', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frame', fontsize=10)
        ax1.set_ylabel('Vehicles', fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        density_img = fig_to_img(fig1, graph_w, graph_h)
        plt.close(fig1)
        dashboard[pad:pad + graph_h, pad:GRAPH_WIDTH - pad, :] = density_img

        # Vehicle size distribution bar chart
        sizes = ['small', 'medium', 'large']
        counts = [size_count[s] for s in sizes]
        fig2, ax2 = plt.subplots(figsize=(4, 2))
        ax2.bar(sizes, counts, color=['green', 'orange', 'red']) # Changed yellow to orange for better canvas visibility
        ax2.set_title('Vehicle Size Distribution', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        
        size_img = fig_to_img(fig2, graph_w, graph_h)
        plt.close(fig2)
        dashboard[pad * 2 + graph_h:pad * 2 + 2 * graph_h, pad:GRAPH_WIDTH - pad, :] = size_img

        # Add header text & border to dashboard
        cv2.putText(dashboard, 'Analytics Pipeline', (pad, pad - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(dashboard, (0, 0), (GRAPH_WIDTH - 1, height - 1), (0, 0, 0), 2)

        # Combine video and dashboard side-by-side
        combined = np.hstack((frame, dashboard))
        out.write(combined)

        frame_idx += 1
        print(f'Processed frame {frame_idx}', end='\r')

    cap.release()
    out.release()
    print(f'\nProcessing complete. Total unique vehicles tracked: {len(total_vehicle_ids)}')

if __name__ == '__main__':
    main()