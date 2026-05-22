import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self):
        pass

    def predict(self, frame):
        """
        Trả về tọa độ 6 điểm chuẩn của sân (Dưới, Lưới, Trên)
        đã được canh chỉnh thủ công cho góc camera hiện tại.
        """
        keypoints = [
            360, 965,   # 0, 1: Bottom-Left
            1528, 959,  # 2, 3: Bottom-Right
            508, 614,   # 4, 5: Net-Left
            1388, 614,  # 6, 7: Net-Right
            594, 423,   # 8, 9: Top-Left
            1304, 414,  # 10, 11: Top-Right
            # Padding cho đủ 28 điểm theo cấu trúc MiniCourt yêu cầu
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 594, 423, 1304, 414
        ]
        return keypoints

    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Vẽ khung sân đa giác màu đỏ lên video để kiểm tra trực quan.
        Nối 4 điểm: Dưới Trái -> Dưới Phải -> Trên Phải -> Trên Trái
        """
        output_frames = []
        for frame in video_frames:
            f = frame.copy()
            
            # Lấy 4 điểm góc sân ngoài cùng để vẽ viền đa giác
            pts = np.array([
                [keypoints[0], keypoints[1]],   # Bottom-Left
                [keypoints[2], keypoints[3]],   # Bottom-Right
                [keypoints[10], keypoints[11]], # Top-Right
                [keypoints[8], keypoints[9]]    # Top-Left
            ], np.int32)
            
            # Reshape mảng pts để phù hợp với hàm polylines của OpenCV
            pts = pts.reshape((-1, 1, 2))
            
            # Vẽ viền sân (đường màu đỏ, độ dày 3 pixel)
            cv2.polylines(f, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
            
            output_frames.append(f)
            
        return output_frames