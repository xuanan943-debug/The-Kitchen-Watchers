Welcome to **The Kitchen Watchers** - Pickleball Video Analysis

An advanced tool for tracking player movement, ball trajectories, and game dynamics using cutting-edge computer vision and machine learning techniques. This project analyzes pickleball matches to provide detailed insights into player positioning, ball movement patterns, and game flow.

✨ Features

👉 **Player & Ball Detection** 🎯
- Uses YOLO v8 for highly accurate player detection
- Tracks all players and the pickleball in real-time

👉 **Court Mapping & Visualization** 🎾🏙️
- Generates a "mini court" visualization showing player & ball positions
- Real-time top-down view of the game

👉 **Advanced Ball Tracking** ⚡📊
- Ball trajectory detection across frames
- Interpolation of missing detections for smooth motion
- Ball shot speed analysis

👉 **Player Tracking & Movement Analysis** 🏃️
- Tracks up to 4 players across the match
- Monitors player positioning and movement patterns

## 📊 Project Structure

```
The-Kitchen-Watchers/
├── README.md
├── main.py                      # Entry point - runs full pipeline
├── input_videos/                # Input video files
├── output_videos/               # Generated output videos
├── models/                      # Pre-trained models
│   ├── yolov8x.pt
│   └── keypoints_model.pth
├── tracker_stubs/               # Cached detection results
├── constants/
│   └── __init__.py
├── trackers/
│   ├── __init__.py
│   ├── player_tracker.py        # Player detection & tracking
│   └── ball_tracker.py          # Ball detection & tracking
├── court_line_detector/
│   ├── __init__.py
│   └── court_line_detector.py
├── mini_court/
│   ├── __init__.py
│   └── mini_court.py
└── utils/
    ├── __init__.py
    ├── bbox_utils.py
    ├── conversions.py
    ├── video_utils.py
    └── player_stats_drawer_utils.py
```

## 🔬 Pipeline Architecture

```
Input Video (24 FPS)
    ↓
[1] YOLO Detection
    Detect players & ball in each frame → Cache results
    ↓
[2] Player Tracking
    Track player positions across frames
    ↓
[3] Ball Tracking & Interpolation
    Detect ball, interpolate missing frames for smooth trajectory
    ↓
[4] Court Detection
    Identify court boundaries & corner keypoints from first frame
    ↓
[5] Coordinate Conversion
    Convert pixel coordinates to court coordinates (top-down mini court)
    ↓
[6] Rendering & Annotation
    Draw bounding boxes, court lines, mini court on video
    ↓
Output Video with Full Annotations
```

## 📍 Pipeline Stage Explanation

### Stage 1️⃣ - YOLO Detection
**What it does:**
- YOLOv8x model scans every frame to detect players and the ball
- Generates bounding boxes for each detected object
- Results are cached to `tracker_stubs/` for faster reruns

**Output:** Detected players and ball positions per frame

### Stage 2️⃣ - Player Tracking
**What it does:**
- Assigns unique IDs to each player across consecutive frames
- Filters out referees by selecting 4 closest players to court center
- Maintains player identity throughout the match

**Output:** Player trajectories and IDs

### Stage 3️⃣ - Ball Tracking & Interpolation
**What it does:**
- Tracks the ball across frames using fine-tuned detection model
- Interpolates missing detections (when ball is occluded)
- Creates smooth ball trajectory for analysis

**Output:** Complete ball trajectory across all frames

### Stage 4️⃣ - Court Detection
**What it does:**
- Detects court lines and corner keypoints from the first frame
- **Current approach:** Manual keypoint input (optimal for static cameras)
  - Uses manually identified pixel coordinates of 4 court corners
  - Faster and more accurate for fixed camera angles
- **Alternative approach:** Hough Transform based detection (available in `feature/hough-transform-court-detection` branch)
  - Automatic line detection algorithm
  - Useful for dynamic or variable camera angles

**Note:** The manual approach was chosen because it produces superior results for fixed camera recordings like this project. Code for Hough Transform court detection is maintained in a separate branch for reference and future improvements.

**Output:** Court boundary coordinates and keypoints

### Stage 5️⃣ - Coordinate Conversion
**What it does:**
- Converts all bounding box coordinates from pixel space to court space
- Uses standard pickleball court dimensions as reference
- Creates top-down (bird's eye) mini court representation
- Enables real-world distance and speed calculations

**Output:** Player and ball positions in court coordinates

### Stage 6️⃣ - Rendering & Annotation
**What it does:**
- Draws bounding boxes around detected players and ball
- Marks court boundaries and keypoints
- Renders mini court with player/ball positions
- Adds frame counter and visual elements
- Saves annotated video to output

**Output:** Fully annotated video file showing all analysis

## 🎬 Output Visualization

The generated video displays:
- ✔️ **Bounding Boxes** - Yellow boxes around players and ball
- ✔️ **Court Lines** - Detected court boundaries with corner markers
- ✔️ **Mini Court** - Real-time top-down view with player positions
- ✔️ **Ball Position** - Tracked ball location on mini court
- ✔️ **Frame Counter** - Current frame number in top-left corner

## ⚙️ Technologies Used

- **YOLOv8** - State-of-the-art object detection
- **OpenCV** - Computer vision and video processing
- **Pandas** - Data structure and analysis
- **NumPy** - Numerical computations
- **Python 3.8+** - Programming language

## 📌 Key Specifications

- **Frame Rate:** 24 FPS
- **Court Width:** 6.10 meters (20 feet) - USA Pickleball standard
- **Court Length:** 13.41 meters (44 feet) - USA Pickleball standard
- **Optimized for:** Fixed camera angle recordings
- **Player Tracking:** Up to 4 players

## 🎓 Final Project for Computer Vision Class
