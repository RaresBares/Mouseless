# Moueless – Hand Gesture Control for Windows

## Overview
Moueless enables full PC control through real-time hand gesture recognition using a standard camera — no mouse or keyboard required.

## System Architecture
1. **Capture:** Video frames are streamed from the webcam.  
2. **Detection:** Hands are localized via a lightweight detector (e.g., MediaPipe or YOLO).  
3. **Classification:** Each cropped hand is classified by a CNN (palm, fist, two_fingers, hands_on_ears).  
4. **Temporal Logic:** A finite state machine interprets gesture sequences and holds to detect meaningful actions.  
5. **Action Dispatch:** Mapped OS-level commands are executed through the system API.

## Core Gestures (v1)
| Gesture | Description | Action |
|----------|--------------|--------|
| 🖐️ → 🖐️ within 1.2 s | Double palm open | Close all windows |
| 🤲 (hands near ears) held ≥ 0.6 s | Mute/Unmute toggle | Toggle system mute |
| ✌️ (two fingers up) | Cursor mode | Enables pointer control |
| ✊ (fist) short tap while in cursor mode | Click | Mouse click event |

## Key Features
- Real-time hand tracking (>30 FPS)
- Cross-platform action layer (Windows/macOS/Linux)
- Configurable gesture–action mapping via YAML
- Confidence-based smoothing and hysteresis to prevent false triggers
- Optional INT8 quantization for low-latency inference

## Project Structure
data/ # Raw and processed training data
models/ # Hand detector and gesture classifier
src/ # Runtime logic (detection, classification, temporal FSM, OS actions)
training/ # Dataset and model training scripts
configs/ # Default and gesture mapping configuration
tests/ # Unit and latency tests
scripts/ # Model export, quantization, and benchmarking
assets/ # Icons and UI assets


## Example Pipeline
1. Camera captures frame (640×480 @ 30 FPS)  
2. Hand detector outputs bounding box  
3. Classifier infers gesture + confidence  
4. Temporal FSM checks gesture history  
5. Action dispatcher triggers corresponding OS event (e.g., Alt + F4, mute, mouse move)

## Dependencies
- Python ≥ 3.10  
- OpenCV, PyTorch, ONNXRuntime  
- PyAutoGUI (Windows), Quartz (macOS), Xlib (Linux)  
- YAML for configuration handling

## Performance Targets
- Total latency < 50 ms  
- Stable FPS ≥ 30  
- Gesture recognition F1 ≥ 0.9 on unseen subjects  

## Future Extensions
- Additional gestures (scroll, zoom, drag)  
- Multi-hand interaction  
- Continuous cursor depth control via z-axis estimation  

## License
MIT License – free for personal and research use.
