import numpy as np
import cv2
import pandas as pd

def draw_player_stats(output_video_frames, player_stats):
    for index, row in player_stats.iterrows():
        # Đảm bảo khung hình tồn tại
        if index >= len(output_video_frames):
            break
            
        frame = output_video_frames[index]
        
        # Mở rộng bảng đen để chứa đủ 4 người chơi
        width = 580 
        height = 230

        # Vị trí góc dưới cùng bên phải
        start_x = frame.shape[1] - width - 30
        start_y = frame.shape[0] - height - 30
        end_x = start_x + width
        end_y = start_y + height

        # Tạo lớp nền mờ (overlay) đen
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Vẽ Tiêu đề
        cv2.putText(frame, "Match Statistics", (start_x + 200, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Vẽ các Tiêu đề Hàng (Thông số)
        cv2.putText(frame, "Shot Speed", (start_x + 10, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "Player Speed", (start_x + 10, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "Avg S. Speed", (start_x + 10, start_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "Avg P. Speed", (start_x + 10, start_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Vòng lặp lấy thông số và vẽ thành 4 cột cho 4 Player
        for player_id in range(1, 5):
            # Tính toán vị trí cột chữ (X_offset) cho từng người chơi
            col_x = start_x + 130 + (player_id - 1) * 110
            
            # Tên cột (P1, P2, P3, P4)
            cv2.putText(frame, f"Player {player_id}", (col_x, start_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            
            # Lấy dữ liệu an toàn (Dùng .get() để nếu file tính toán bị lỗi/thiếu thì trả về mặc định là 0)
            shot_spd = row.get(f'player_{player_id}_last_shot_speed', 0.0)
            plyr_spd = row.get(f'player_{player_id}_last_player_speed', 0.0)
            avg_shot_spd = row.get(f'player_{player_id}_average_shot_speed', 0.0)
            avg_plyr_spd = row.get(f'player_{player_id}_average_player_speed', 0.0)
            
            # Xử lý trường hợp không có dữ liệu (NaN)
            shot_spd_str = f"{shot_spd:.1f}" if pd.notna(shot_spd) and shot_spd > 0 else "N/A"
            plyr_spd_str = f"{plyr_spd:.1f}" if pd.notna(plyr_spd) and plyr_spd > 0 else "N/A"
            avg_shot_spd_str = f"{avg_shot_spd:.1f}" if pd.notna(avg_shot_spd) and avg_shot_spd > 0 else "N/A"
            avg_plyr_spd_str = f"{avg_plyr_spd:.1f}" if pd.notna(avg_plyr_spd) and avg_plyr_spd > 0 else "N/A"

            # Vẽ các thông số vào đúng hàng và đúng cột
            cv2.putText(frame, shot_spd_str, (col_x, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, plyr_spd_str, (col_x, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, avg_shot_spd_str, (col_x, start_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, avg_plyr_spd_str, (col_x, start_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        output_video_frames[index] = frame
        
    return output_video_frames