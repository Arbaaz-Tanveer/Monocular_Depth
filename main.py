import cv2
import time
import threading
from ultralytics import YOLO
# Import only what is needed for ground coordinate estimation
from groundmapping import pixel_to_ground, compute_calibration_params, CoordinateEstimator

# ---------------------------------------------------
# Global calibration variables (to be computed once)
# ---------------------------------------------------
calibration = {}  # Will hold K, D, new_K, estimator

# ---------------------------------------------------
# Global shared dictionaries and lock
# ---------------------------------------------------
global_lock = threading.Lock()
# Store each frame as a tuple: (frame, timestamp)
frames_dict = {}         # {camera_index: (latest captured frame, frame timestamp)}
results_dict = {}        # {camera_index: latest inference result (with YOLO overlays)}
capture_fps_dict = {}    # {camera_index: latest capture FPS}
# detection_results stores all detections for each camera.
# Each detection dict now includes a 'timestamp' entry.
detection_results = {}   # {camera_index: [detection, ...]}

##-----uncomment this to export model to TensorRT-----
# model = YOLO("yolo11n.pt")
# # Export the model to TensorRT
# model.export(format="engine")  # creates 'yolo11n.engine'

# # Load the exported TensorRT model
# trt_model = YOLO("yolo11n.engine")
# ---------------------------------------------------
# Load the YOLO TensorRT engine
# ---------------------------------------------------
try:
    trt_model = YOLO("yolo11n.engine", verbose=False)
    print("TensorRT model loaded successfully.")
except Exception as e:
    print(f"Error loading TensorRT model: {e}")
    exit()

# ---------------------------------------------------
# Capture thread function (supports optional custom settings)
# ---------------------------------------------------
def capture_thread_func(camera_index, settings=None):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2 if settings else 0)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    if settings:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.get('width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.get('height', 480))
        cap.set(cv2.CAP_PROP_FPS, settings.get('fps', 30))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.get('fourcc', 'MJPG')))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, settings.get('buffersize', 1))
        cap.set(cv2.CAP_PROP_BRIGHTNESS, settings.get('brightness', 30))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, settings.get('auto_exposure', 1))
        cap.set(cv2.CAP_PROP_EXPOSURE, settings.get('exposure', 300))
        print(f"Camera {camera_index} opened with custom settings: {settings}")
    else:
        print(f"Camera {camera_index} opened with default settings.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps_val = frame_count / elapsed
            with global_lock:
                capture_fps_dict[camera_index] = fps_val
            frame_count = 0
            start_time = time.time()

        timestamp = time.time()
        with global_lock:
            frames_dict[camera_index] = (frame.copy(), timestamp)

        time.sleep(0.005)

    cap.release()

# ---------------------------------------------------
# Inference thread function: processes each camera’s frame
# ---------------------------------------------------
def inference_thread_func():
    model_names = trt_model.names if hasattr(trt_model, 'names') else {}
    while True:
        with global_lock:
            current_frames = {cam: (frame.copy(), ts) for cam, (frame, ts) in frames_dict.items()}
        if not current_frames:
            time.sleep(0.01)
            continue

        for cam_index, (frame, frame_timestamp) in current_frames.items():
            detections_info = []
            try:
                results = trt_model(frame)
                inferred_frame = frame
                for r in results:
                    inferred_frame = r.plot() or inferred_frame
                    if hasattr(r, "boxes") and r.boxes is not None:
                        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else r.boxes.xyxy
                        classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else r.boxes.cls
                        for idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = box
                            center_x = (x1 + x2) / 2
                            bottom_center_y = y2
                            pixel_coord = (center_x, bottom_center_y)

                            cls_id = int(classes[idx])
                            obj_label = model_names.get(cls_id, f"Class {cls_id}")

                            # Estimate ground coordinates if calibration is ready
                            if calibration:
                                ground_coords = pixel_to_ground(
                                    [pixel_coord],
                                    calibration["estimator"],
                                    calibration["K"],
                                    calibration["D"],
                                    calibration["new_K"],
                                    show=False
                                )
                                ground_coord = ground_coords[0] if ground_coords else None
                            else:
                                ground_coord = None

                            detections_info.append({
                                "camera": f"Camera {cam_index}",
                                "object": obj_label,
                                "pixel_coord": pixel_coord,
                                "ground_coord": ground_coord,
                                "timestamp": frame_timestamp
                            })
            except Exception as e:
                print(f"Error during inference on camera {cam_index}: {e}")

            with global_lock:
                results_dict[cam_index] = inferred_frame
                detection_results[cam_index] = detections_info

        time.sleep(0.001)

# ---------------------------------------------------
# Main function: initializes calibration, starts threads, and displays results.
# ---------------------------------------------------
def main():
    # Compute calibration parameters once
    h, w = 960, 1280
    K, D, new_K = None, None, None
    map1, map2, K, D, new_K = compute_calibration_params(h, w, distortion_param=0.05, show=False)
    estimator = CoordinateEstimator(
        image_width=w,
        image_height=h,
        fov_horizontal=95,
        fov_vertical=78,
        camera_height=0.75,
        camera_tilt=30
    )
    with global_lock:
        calibration["K"] = K
        calibration["D"] = D
        calibration["new_K"] = new_K
        calibration["estimator"] = estimator

    # Define camera settings and indices.
    custom_settings = {
        'width': w,
        'height': h,
        'fps': 30,
        'fourcc': 'MJPG',
        'buffersize': 1,
        'brightness': 20,
        'auto_exposure': 1,
        'exposure': 300
    }
    camera_indices = [0,2,4,6]

    # Start capture threads
    for cam_idx in camera_indices:
        cap_temp = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        if not cap_temp.isOpened():
            print(f"Camera {cam_idx} not available.")
            continue
        cap_temp.release()
        threading.Thread(target=capture_thread_func, args=(cam_idx, custom_settings), daemon=True).start()

    # Start inference thread
    threading.Thread(target=inference_thread_func, daemon=True).start()

    # Main display loop
    while True:
        with global_lock:
            for cam_index, detections in detection_results.items():
                print(f"Detections from Camera {cam_index}:")
                for det in detections:
                    print(
                        f"  Object: {det['object']} | Pixel: {det['pixel_coord']} | "
                        f"Ground: {det['ground_coord']} | Timestamp: {det['timestamp']:.3f}"
                    )
        with global_lock:
            current_results = {cam: frame.copy() for cam, frame in results_dict.items() if frame is not None}
            current_fps = capture_fps_dict.copy()

        # Display each camera's inference result.
        for cam_index, frame in current_results.items():
            fps_val = current_fps.get(cam_index, 0.0)
            cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            window_name = f"Camera {cam_index} Inference"
            cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
