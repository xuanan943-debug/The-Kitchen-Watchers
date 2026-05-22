from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        # Lưu lại bản gốc
        original_positions = ball_positions 
        ball_positions_list = [x.get(1, []) for x in ball_positions]
        
        try:
            df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1','y1','x2','y2'])
        except ValueError:
            print("Không tìm thấy bóng rõ ràng, bỏ qua vẽ quỹ đạo bóng...")
            return original_positions 

        # Xử lý nội suy
        # 1. Điền các frame bị thiếu (interpolate)
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit_direction='both')
        df_ball_positions = df_ball_positions.bfill()

        # 2. Xử lý làm mượt quỹ đạo bóng (Smooth trajectory) bằng Savitzky-Golay
        # Tránh lỗi nếu số frame quá ít
        if len(df_ball_positions) > 15:
            try:
                window_length = 15 # Phải là số lẻ
                polyorder = 3
                df_ball_positions['x1'] = savgol_filter(df_ball_positions['x1'], window_length, polyorder)
                df_ball_positions['y1'] = savgol_filter(df_ball_positions['y1'], window_length, polyorder)
                df_ball_positions['x2'] = savgol_filter(df_ball_positions['x2'], window_length, polyorder)
                df_ball_positions['y2'] = savgol_filter(df_ball_positions['y2'], window_length, polyorder)
            except Exception as e:
                print(f"Lỗi khi làm mượt quỹ đạo bóng (Bỏ qua bước này): {e}")

        # Chuyển đổi lại về định dạng ban đầu
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions_list = [x.get(1, []) for x in ball_positions]
        
        try:
            df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1','y1','x2','y2'])
        except ValueError:
            print("Không đủ dữ liệu bóng để đếm số cú đánh, bỏ qua...")
            return [] 

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        
        minimum_change_frames_for_hit = 25
        
        # Chỉ quét khi có đủ frame
        if len(df_ball_positions) > int(minimum_change_frames_for_hit * 1.2):
            for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
                negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
                positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

                if negative_position_change or positive_position_change:
                    change_count = 0 
                    for change_frame in range(i+1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                        negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                        positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                        if negative_position_change and negative_position_change_following_frame:
                            change_count += 1
                        elif positive_position_change and positive_position_change_following_frame:
                            change_count += 1
                
                    if change_count > minimum_change_frames_for_hit - 1:
                        df_ball_positions.iloc[i, df_ball_positions.columns.get_loc('ball_hit')] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        # Hạ thấp ngưỡng confidence (conf) để ép mô hình tìm bóng, dù mờ
        # Đảm bảo chỉ nhận diện class bóng thể thao (thường là 32 trong COCO)
        results = self.model.predict(frame, conf=0.1, classes=[32])[0]
        ball_dict = {}
        
        # Chỉ lấy bounding box đầu tiên (nếu có) vì chỉ có 1 quả bóng
        if len(results.boxes) > 0:
            result = results.boxes.xyxy.tolist()[0]
            ball_dict[1] = result
            
        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # Vẽ điểm (chấm tròn) đại diện cho tâm bóng thay vì khung chữ nhật lớn
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"Ball", (center_x - 10, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames