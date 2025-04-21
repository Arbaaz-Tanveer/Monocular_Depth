# Object Detection & Ground-Position Estimation

This repository detects objects (e.g., balls) in a camera feed, computes their ground‐plane coordinates (corrected for object radius), and prints results in real time.

---

## Calibration

- **Field of View (FOV)** and **Camera Tilt Angle** must be accurately calibrated for precise ground‐position estimates.  
- Use the provided **`live_grid.py`** script to visualize a known grid pattern and fine‑tune your camera’s FOV and tilt parameters until grid intersections align on screen.  

---

## Dependencies & Installation

1. **Python 3.8+**  
2. **OpenCV** 
   ```bash
   pip install opencv-python
   ```
3. **NumPy**  
   ```bash
   pip install numpy
   ```
4. **Ultralytics/Yolo**  
   ```bash
   pip install ultralytics
   ```
5. **TensorRT Conversion** (optional but recommended for speed)  
   - Convert your YOLO model to TensorRT using the Ultralytics export command:  
     ```bash
     yolo export model=yolo11n.pt format=engine
     ```
   - Ensure your CUDA and TensorRT installations match the Ultralytics requirements.

---

## Configuration & Tips

- **Camera Index**: By default, cameras are indexed `[0,2,4,6]`. Adjust accordingly
- **Camera Settings**: Lighting greatly affects detection quality. Tweak camera properties in the code (brightness, exposure, auto‐exposure) to suit indoor vs. outdoor environments.  .  
- **Ball Radius**: Set your object’s physical radius (`ball_radius`) in meters for accurate center‐point correction.

---

## Running the System

1. Calibrate FOV/tilt via `python live_grid.py`.  
2. Update calibration parameters in `groundmapping.py`.  
3. Launch detection script:  
   ```bash
   python main.py
   ```
4. Press **`q`** to quit the live display.

---

## Notes
- TensorRT engines will need re‑export when changing YOLO version or model parameters.  




