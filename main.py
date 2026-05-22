from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
import numpy as np
from copy import deepcopy

def main():
    # ---------------------------------------------------------
    # 1. ĐỌC VIDEO ĐẦU VÀO
    # ---------------------------------------------------------
    input_video_path = "input_videos/Video Project 5.mp4"
    video_frames = read_video(input_video_path)

    # ---------------------------------------------------------
    # 2. KHỞI TẠO CÁC MÔ HÌNH NHẬN DIỆN VÀ THEO DÕI
    # ---------------------------------------------------------
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolov8x.pt')

    # Theo dõi người chơi (Đã cấu hình để bắt 4 người)
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    # Theo dõi quả bóng
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    
    # Nội suy quỹ đạo bóng để làm mượt các khung hình bị mất
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # ---------------------------------------------------------
    # 3. NHẬN DIỆN SÂN VÀ CHỌN LỌC NGƯỜI CHƠI
    # ---------------------------------------------------------
    court_line_detector = CourtLineDetector()
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Lọc bỏ trọng tài, chỉ giữ lại 4 người chơi gần trục giữa sân nhất
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Khởi tạo mô-đun vẽ sân Mini 2D
    mini_court = MiniCourt(video_frames[0]) 

    # Đếm số khung hình có sự thay đổi quỹ đạo bóng (Cú đánh/Chạm vợt)
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Chuyển đổi tọa độ từ pixel sang tọa độ sân mini (Top-down view)
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
                                                                                                          player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    # ---------------------------------------------------------
    # 4. TÍNH TOÁN THỐNG KÊ (HẬU TRƯỜNG)
    # ---------------------------------------------------------
    # Khởi tạo dữ liệu cho 4 người chơi
    initial_stats = {'frame_num': 0}
    for i in range(1, 5):
        initial_stats[f'player_{i}_number_of_shots'] = 0
        initial_stats[f'player_{i}_total_shot_speed'] = 0
        initial_stats[f'player_{i}_last_shot_speed'] = 0
        initial_stats[f'player_{i}_total_player_speed'] = 0
        initial_stats[f'player_{i}_last_player_speed'] = 0

    player_stats_data = [initial_stats]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24.0

        # Tốc độ quả bóng
        if 1 in ball_mini_court_detections[start_frame] and 1 in ball_mini_court_detections[end_frame]:
            distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                               ball_mini_court_detections[end_frame][1])
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court()) 
            speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        else:
            speed_of_ball_shot = 0.0

        # Xác định người đánh bóng
        player_positions = player_mini_court_detections[start_frame]
        if player_positions and 1 in ball_mini_court_detections[start_frame]:
            player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                     ball_mini_court_detections[start_frame][1]))
        else:
            player_shot_ball = None

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        
        # Cập nhật số liệu cho người đánh và tốc độ di chuyển của những người còn lại
        if player_shot_ball and 1 <= player_shot_ball <= 4:
            current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
            current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
            current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

            for other_player_id in range(1, 5):
                if other_player_id != player_shot_ball and other_player_id in player_mini_court_detections[start_frame] and other_player_id in player_mini_court_detections[end_frame]:
                    dist_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][other_player_id],
                                                            player_mini_court_detections[end_frame][other_player_id])
                    dist_opponent_meters = convert_pixel_distance_to_meters(dist_opponent_pixels,
                                                                            constants.DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_mini_court()) 
                    speed_of_opponent = dist_opponent_meters / ball_shot_time_in_seconds * 3.6
                    
                    current_player_stats[f'player_{other_player_id}_total_player_speed'] += speed_of_opponent
                    current_player_stats[f'player_{other_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # ---------------------------------------------------------
    # 5. KẾT XUẤT VIDEO (RENDERING)
    # ---------------------------------------------------------
    # Vẽ Khung nhận diện (Bounding Boxes)
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Vẽ Khung sân 2D đè lên video
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Vẽ Sân Mini (Top-down)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))    

    # Vẽ Bảng điều khiển Thông số (ĐÃ ĐƯỢC ẨN ĐỂ TẬP TRUNG VÀO SÂN MINI)
    # output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Vẽ số khung hình ở góc trên bên trái
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Lưu kết quả
    save_video(output_video_frames, "output_videos/output_video.mp4")

if __name__ == "__main__":
    main()