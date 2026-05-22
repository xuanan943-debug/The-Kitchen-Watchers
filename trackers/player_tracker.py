from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # 1. Quét tìm khung hình có ít nhất 4 người để xác định ai là người chơi
        chosen_players = []
        for player_dict in player_detections:
            if len(player_dict) >= 4:
                chosen_players = self.choose_players(court_keypoints, player_dict)
                break
        
        # 2. Nếu không tìm thấy khung hình 4 người, lấy tạm từ khung hình đầu tiên
        if not chosen_players and len(player_detections) > 0:
            chosen_players = self.choose_players(court_keypoints, player_detections[0])

        # 3. Lọc danh sách: Chỉ giữ lại các ID thuộc về 4 người chơi đã chọn
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {
                track_id: bbox for track_id, bbox in player_dict.items() 
                if track_id in chosen_players
            }
            filtered_player_detections.append(filtered_player_dict)
            
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        # Tính trục giữa sân theo phương X (trung bình cộng của tất cả keypoints)
        court_x_coords = court_keypoints[0::2]
        court_center_x = sum(court_x_coords) / len(court_x_coords)
        
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            
            # Khoảng cách tới trục giữa sân (người chơi sẽ có khoảng cách nhỏ, trọng tài/khán giả sẽ có khoảng cách lớn)
            dist_to_center = abs(player_center[0] - court_center_x)
            distances.append((track_id, dist_to_center))
        
        # Sắp xếp theo khoảng cách tăng dần (gần trục giữa nhất lên đầu)
        distances.sort(key=lambda x: x[1])
        
        # Lấy 4 người gần trục giữa nhất
        chosen_players = [d[0] for d in distances[:4]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame):
        # Dùng track để giữ ID ổn định
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        player_dict = {}
        
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.tolist()
            track_ids = results.boxes.id.tolist()
            class_ids = results.boxes.cls.tolist()
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                object_cls_name = id_name_dict[int(class_id)]
                if object_cls_name == "person":
                    player_dict[int(track_id)] = box
        
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                label = f"P{track_id}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames