# The Kitchen Watchers - Pickleball Video Analysis

A real-time video analysis system for pickleball matches that detects players, tracks movement, analyzes ball trajectories, and generates annotated output videos.

## Project Pipeline

```
Input Video
    ↓
[1] YOLO Detection → Detect players & ball in each frame
    ↓
[2] Player Tracking → Track player positions across frames
    ↓
[3] Ball Tracking → Track ball positions & interpolate missing frames
    ↓
[4] Court Detection → Identify court boundaries & lines
    ↓
[5] Mini Court Conversion → Convert pixel coordinates to court coordinates (top-down view)
    ↓
[6] Rendering → Draw bounding boxes, court lines, mini court on video
    ↓
Output Video with Annotations
```

## Features

- 🎥 **Player Detection**: Uses YOLOv8 to detect players and filter referees
- 👁️ **Player Tracking**: Tracks up to 4 players across video frames
- 🏓 **Ball Tracking**: Detects and interpolates ball trajectories
- 📍 **Court Line Detection**: Automatically identifies pickleball court boundaries
- 🗺️ **Mini Court Visualization**: Top-down 2D court representation
- 📊 **Frame Annotation**: Adds frame numbers and visualizations to output

## Project Structure

```
The-Kitchen-Watchers/
├── README.md
├── main.py                      # Entry point - runs full pipeline
├── input_videos/                # Input video files
│   └── Video Project 5.mp4
├── output_videos/               # Generated output videos
│   └── output_video.mp4
├── models/                      # Pre-trained models
│   ├── yolov8x.pt              # YOLO object detection model
│   └── keypoints_model.pth      # Keypoint detection model
├── tracker_stubs/               # Cached detection results
├── constants/
│   └── __init__.py
├── trackers/                    # Tracking modules
│   ├── __init__.py
│   ├── player_tracker.py        # Player detection & tracking
│   └── ball_tracker.py          # Ball detection & tracking
├── court_line_detector/         # Court detection module
│   ├── __init__.py
│   └── court_line_detector.py
├── mini_court/                  # Top-down court visualization
│   ├── __init__.py
│   └── mini_court.py
└── utils/                       # Utility functions
    ├── __init__.py
    ├── bbox_utils.py
    ├── conversions.py
    ├── video_utils.py
    └── player_stats_drawer_utils.py
```

## Quick Start

### Run Analysis
```bash
python main.py
```

This will:
1. Read video from `input_videos/Video Project 5.mp4`
2. Process all frames through the detection and tracking pipeline
3. Generate annotated video to `output_videos/output_video.mp4`

## Pipeline Explanation

### 1️⃣ Detection (YOLO)
- YOLOv8 detects all players and the ball in each frame
- Results are cached to speed up reruns

### 2️⃣ Tracking
- Player positions are tracked across consecutive frames
- Ball positions are interpolated to fill missing detections
- System identifies and filters to 4 main players

### 3️⃣ Court Detection
- Court lines are detected from the first frame
- Identifies key court points: corners and net positions

### 4️⃣ Coordinate Conversion
- Pixel coordinates are converted to court coordinates
- Mini court uses a 2D top-down representation

### 5️⃣ Rendering
- All annotations are drawn onto video frames
- Output includes: bounding boxes, court lines, mini court, frame numbers

## Output Visualization

The output video shows:
- **Yellow bounding boxes** around detected players
- **Yellow circle** for detected ball
- **Court boundaries** with keypoints
- **Mini court (top-down view)** with real-time player and ball positions
- **Frame counter** in top-left corner

## Technologies Used

- **YOLOv8** - Real-time object detection
- **OpenCV** - Video processing and drawing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations

## Notes

- Video processes at 24 FPS
- System optimized for pickleball court analysis
- Final project for Computer Vision class
