# Hough Transform Court Detection Feature Branch

This branch contains an alternative approach for court line detection using the Hough Transform algorithm.

## Overview

The Hough Transform is an algorithm that detects lines in images by finding intersections of edge-detected points. It's useful for automatically detecting court boundaries without manual keypoint input.

## When to Use

✅ **Use Hough Transform when:**
- Camera angle is variable/dynamic
- Analyzing multiple videos from different sources
- Need fully automated detection pipeline
- Court is at different angles or distances

❌ **Don't use Hough Transform when:**
- Working with fixed/static camera (like this project)
- High accuracy is critical
- Processing speed is important
- Camera angle is consistent

## Implementation

### File Location
```
court_line_detector/
├── court_line_detector.py              # Current: Manual keypoint approach
└── hough_transform_court_detector.py   # New: Automated Hough approach
```

### How to Switch

In `main.py`, replace:
```python
from court_line_detector import CourtLineDetector
```

With:
```python
from court_line_detector.hough_transform_court_detector import HoughTransformCourtDetector as CourtLineDetector
```

Then use it identically:
```python
court_line_detector = CourtLineDetector()
court_keypoints = court_line_detector.predict(video_frames[0])
```

## Algorithm Steps

1. **Preprocessing**
   - Convert frame to grayscale
   - Apply Gaussian blur to reduce noise

2. **Edge Detection**
   - Use Canny edge detector
   - Identify pixels that represent edges

3. **Hough Line Transform**
   - Find all lines in the edge map
   - Convert from image space to Hough space

4. **Line Classification**
   - Separate horizontal and vertical lines
   - Based on angle detection

5. **Corner Detection**
   - Find intersections of lines
   - Identify 4 court corners

6. **Keypoint Organization**
   - Format output as standard 28-point array
   - Compatible with rest of pipeline

## Accuracy Trade-offs

| Aspect | Manual Keypoint | Hough Transform |
|--------|-----------------|-----------------|
| Speed | ⚡ Very Fast | ⚠️ Slower |
| Accuracy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Automation | ❌ Manual | ✅ Automatic |
| Fixed Cameras | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Dynamic Angles | ⭐⭐ | ⭐⭐⭐⭐ |

## Key Parameters

Adjust in `HoughTransformCourtDetector.__init__()`:

```python
threshold = 50              # Hough voting threshold (lower = more lines detected)
min_line_length = 100       # Minimum line length in pixels
max_line_gap = 10           # Max gap allowed in a line
```

## Debugging

Use the visualization utility:
```python
detector = HoughTransformCourtDetector()
lines = cv2.HoughLinesP(...)  # Get detected lines
frame_with_lines = detector.visualize_lines(frame, lines)
```

## Why Manual Keypoint for This Project

The current implementation uses manual keypoints because:
1. **Camera is fixed** - Same angle throughout video
2. **Higher accuracy** - Precisely defined court corners
3. **Better performance** - No need for image processing each frame
4. **Reliability** - Consistent results across frames

This is a common trade-off: for production systems with fixed cameras, manual calibration beats automated approaches.

## Future Improvements

- [ ] Machine Learning based corner detection
- [ ] Hybrid approach (Hough + manual refinement)
- [ ] Real-time line detection with confidence scores
- [ ] Support for distorted/wide-angle cameras
- [ ] Automatic camera calibration

## References

- Hough Transform: https://en.wikipedia.org/wiki/Hough_transform
- OpenCV Hough Lines: https://docs.opencv.org/master/d3/d8e/group__imgproc__shape.html#gaf849da36ab52dc04ce51b1b871e59f21
- Computer Vision: Algorithms and Applications - Richard Szeliski
