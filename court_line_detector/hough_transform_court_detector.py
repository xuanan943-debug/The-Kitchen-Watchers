import cv2
import numpy as np

class HoughTransformCourtDetector:
    """
    Alternative court detection method using Hough Transform for line detection.
    This approach automatically detects court lines without manual keypoint input.
    
    Use this for:
    - Dynamic camera angles
    - Variable video sources
    - Automated analysis pipelines
    
    Note: Manual keypoint detection is preferred for fixed camera recordings
    because it produces more accurate results faster.
    """
    
    def __init__(self, 
                 threshold=50, 
                 min_line_length=100, 
                 max_line_gap=10):
        """
        Initialize Hough Transform detector
        
        Args:
            threshold: Edge detection threshold (higher = stricter)
            min_line_length: Minimum line length to detect
            max_line_gap: Maximum gap allowed in lines
        """
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def predict(self, frame):
        """
        Detect court lines using Hough Transform
        Returns keypoints in same format as manual detector (28 values)
        
        Detection steps:
        1. Convert to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Edge detection (Canny)
        4. Hough Line Transform
        5. Find intersection points (court corners)
        6. Organize keypoints
        """
        # Step 1: Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Step 2: Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Step 3: Hough Line Transform
        lines = cv2.HoughLinesP(edges, 
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.threshold,
                               minLineLength=self.min_line_length,
                               maxLineGap=self.max_line_gap)
        
        # Step 4: Extract and classify lines
        if lines is not None:
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle
                angle = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle)
                
                # Classify as horizontal or vertical
                if abs(angle_deg) < 30 or abs(angle_deg - 180) < 30:
                    horizontal_lines.append(line[0])
                elif abs(angle_deg - 90) < 30 or abs(angle_deg + 90) < 30:
                    vertical_lines.append(line[0])
            
            # Sort lines by position
            horizontal_lines.sort(key=lambda l: l[1])  # Sort by y
            vertical_lines.sort(key=lambda l: l[0])    # Sort by x
            
            # Step 5: Find intersection points (court corners)
            keypoints = self._find_court_keypoints(
                horizontal_lines, 
                vertical_lines,
                frame.shape
            )
        else:
            # Fallback if no lines detected
            keypoints = [0] * 28
        
        return keypoints
    
    def _find_court_keypoints(self, horizontal_lines, vertical_lines, frame_shape):
        """
        Find the 4 corners of the court from detected lines
        
        Expected court layout:
        - 2 horizontal lines (top and bottom boundaries)
        - 2 vertical lines (left and right boundaries)
        - 1 horizontal line in middle (net line)
        """
        height, width = frame_shape[:2]
        keypoints = [0] * 28
        
        try:
            # Select the most prominent lines
            # Typically: 2 horizontal (top/bottom) + 1 horizontal (net) = 3 horizontal
            # And 2 vertical (left/right) = 2 vertical
            
            if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                # Sort to get extreme lines
                h_lines_sorted = sorted(
                    horizontal_lines,
                    key=lambda l: l[1]  # Sort by y coordinate
                )
                v_lines_sorted = sorted(
                    vertical_lines,
                    key=lambda l: l[0]  # Sort by x coordinate
                )
                
                # Get approximate court boundaries
                bottom_line = h_lines_sorted[0]      # Lowest line (visually)
                top_line = h_lines_sorted[-1]        # Highest line (visually)
                
                left_line = v_lines_sorted[0]        # Leftmost line
                right_line = v_lines_sorted[-1]      # Rightmost line
                
                # Calculate approximate corner coordinates
                # Bottom-Left
                bl_x = int((bottom_line[0] + bottom_line[2]) / 2)
                bl_y = int((bottom_line[1] + bottom_line[3]) / 2)
                
                # Bottom-Right
                br_x = int((bottom_line[0] + bottom_line[2]) / 2)
                br_y = int((bottom_line[1] + bottom_line[3]) / 2)
                
                # Top-Left
                tl_x = int((top_line[0] + top_line[2]) / 2)
                tl_y = int((top_line[1] + top_line[3]) / 2)
                
                # Top-Right
                tr_x = int((top_line[0] + top_line[2]) / 2)
                tr_y = int((top_line[1] + top_line[3]) / 2)
                
                # Use line endpoints for better accuracy
                if len(h_lines_sorted) > 0:
                    bottom_y = int(np.mean([bottom_line[1], bottom_line[3]]))
                    keypoints[0], keypoints[1] = left_line[0], bottom_y      # Bottom-Left
                    keypoints[2], keypoints[3] = right_line[0], bottom_y     # Bottom-Right
                
                if len(h_lines_sorted) >= 2:
                    top_y = int(np.mean([top_line[1], top_line[3]]))
                    keypoints[8], keypoints[9] = left_line[0], top_y         # Top-Left
                    keypoints[10], keypoints[11] = right_line[0], top_y      # Top-Right
                
                # Net line (middle horizontal)
                if len(h_lines_sorted) >= 3:
                    net_line = h_lines_sorted[len(h_lines_sorted) // 2]
                    net_y = int(np.mean([net_line[1], net_line[3]]))
                    keypoints[4], keypoints[5] = left_line[0], net_y         # Net-Left
                    keypoints[6], keypoints[7] = right_line[0], net_y        # Net-Right
                else:
                    # Estimate net line as middle
                    net_y = int((bottom_y + top_y) / 2)
                    keypoints[4], keypoints[5] = left_line[0], net_y
                    keypoints[6], keypoints[7] = right_line[0], net_y
                
                # Padding
                keypoints[12:28] = [0, 0, 0, 0, 0, 0, 0, 0, 
                                   keypoints[8], keypoints[9],
                                   keypoints[10], keypoints[11]]
        
        except Exception as e:
            print(f"Error detecting court keypoints: {e}")
            # Return default keypoints on error
            keypoints = [0] * 28
        
        return keypoints
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draw court outline and detected lines on video frames
        """
        output_frames = []
        for frame in video_frames:
            f = frame.copy()
            
            # Draw court boundary polygon
            pts = np.array([
                [keypoints[0], keypoints[1]],    # Bottom-Left
                [keypoints[2], keypoints[3]],    # Bottom-Right
                [keypoints[10], keypoints[11]],  # Top-Right
                [keypoints[8], keypoints[9]]     # Top-Left
            ], np.int32)
            
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(f, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Draw keypoint markers
            for i in range(0, 12, 2):
                if keypoints[i] > 0 and keypoints[i+1] > 0:
                    cv2.circle(f, (keypoints[i], keypoints[i+1]), 5, (0, 0, 255), -1)
            
            output_frames.append(f)
        
        return output_frames
    
    @staticmethod
    def visualize_lines(frame, lines):
        """
        Utility function to visualize detected lines for debugging
        """
        output = frame.copy()
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return output
