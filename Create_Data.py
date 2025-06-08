import cv2
import mediapipe as mp
import csv
import os
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

current_label = "Squat"

save_now = False

def mouse_callback(event, x, y, flags, param):
    global save_now
    if event == cv2.EVENT_LBUTTONDOWN:
        save_now = True

cv2.namedWindow('Mediapipe Feed')
cv2.setMouseCallback('Mediapipe Feed', mouse_callback)


file_path = 'pose_data.csv'
file_exists = os.path.isfile(file_path)

csv_file = open(file_path, mode='a', newline='')
csv_writer = csv.writer(csv_file)

# Nếu file chưa có -> ghi header
if not file_exists:
    header = ['label']
    for i in range(33):
        header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
    csv_writer.writerow(header)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    key = cv2.waitKey(10) & 0xFF

    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Thực hiện phát hiện
    results = pose.process(image)
    # Tô lại màu hình ảnh thành BGR ban đầu
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.putText(image, current_label, (70, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),
                1, cv2.LINE_AA)

    cv2.imshow('Mediapipe Feed', image)
    try:

        if key == ord('1'):
            current_label = 'Squat'
        elif key == ord('2'):
            current_label = 'PushUp'
        elif key == ord('3'):
            current_label = 'Bicep Curl'
        elif save_now and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            if len(landmarks) == 33:
                print(f"Đã phát hiện {len(landmarks)} điểm")
                row = [current_label]
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y,lm.z, lm.visibility])
                if len(row) == 133:
                    csv_writer.writerow(row)
                    csv_file.flush()

                    df = pd.read_csv('pose_data.csv')
                    label_counts = df['label'].value_counts()
                    count = label_counts.get(current_label, 0)
                    print("Đã ghi 1 dòng dữ liệu")
                    print("Số mẫu {} hiện có: {} ".format(current_label, count))
                    print()
                else:
                    print("Không đủ 33 điểm.")
                save_now = False
        elif key == ord('q'):
            break
    except:
        pass

csv_file.close()
cap.release()
cv2.destroyAllWindows()

