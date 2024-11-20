import cv2
import os
from datetime import datetime

def extract_frames(video_path, output_folder, frame_interval=1):
    # Membuka video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Tidak dapat membuka video {video_path}")
        return

    # Pastikan folder output ada
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    extracted_count = 0

    while True:
        # Membaca frame
        success, frame = video.read()
        if not success:
            print("Selesai mengekstrak frame.")
            break

        # Menyimpan frame setiap interval yang diinginkan
        if frame_count % frame_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_filename = os.path.join(output_folder, f"frame_{timestamp}.jpeg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    # Membersihkan
    video.release()
    print(f"Frame berhasil diekstrak: {extracted_count} frame disimpan di {output_folder}")

# Konfigurasi
video_path = "D:/gambarsawit/Ripe/v11.mp4"  # Ganti dengan path video Anda
output_folder = "D:/gambarsawit/Ripe"  # Folder untuk menyimpan frame
frame_interval = 30  # Ambil setiap 30 frame, sesuaikan dengan kebutuhan

# Jalankan fungsi
extract_frames(video_path, output_folder, frame_interval)
