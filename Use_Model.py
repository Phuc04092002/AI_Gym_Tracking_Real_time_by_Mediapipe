import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

model = joblib.load("pose_classifier.pkl")
encoder = joblib.load("label_encoder.pkl")

cv2.namedWindow('Mediapipe Feed')
cap = cv2.VideoCapture(0)

squat_count = pushup_count = bicepCurl_count = 0
squat_stage = pushup_stage = bicepCurl_stage = None

while cap.isOpened():
    key = cv2.waitKey(10) & 0xFF

    ret, frame = cap.read()
    height, width, _ = frame.shape

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

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        total_visibility = sum([lm.visibility for lm in landmarks])
        if total_visibility < 20:
            cv2.putText(image, "Khong du khop de nhan dien tu the!",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Mediapipe Feed', image)
            try:
                if key == ord('q'):
                    break
            except:
                pass
            continue
        if len(landmarks) == 33:
            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])

            columns = []
            for i in range(33):
                columns.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
            X_input_df = pd.DataFrame([row], columns=columns)

            prediction = model.predict(X_input_df)[0]
            predicted_label = encoder.inverse_transform([prediction])[0]

            cv2.putText(image, "Tu the hien tai la: {}".format(predicted_label),
                        (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

            if predicted_label == "Squat":
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                cv2.putText(image, str(left_knee_angle),
                            tuple(np.multiply(left_knee, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                if left_knee_angle < 90 and right_knee_angle < 90:
                    squat_stage = "down"
                if squat_stage == "down" and left_knee_angle > 160 and right_knee_angle >160:
                    squat_stage = "up"
                    squat_count += 1
            elif predicted_label == "PushUp":
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                cv2.putText(image, str(left_elbow_angle),
                            tuple(np.multiply(left_elbow, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                if left_elbow_angle < 90 and right_elbow_angle < 90:
                    pushup_stage = "down"
                if pushup_stage == "down" and left_elbow_angle > 160 and right_elbow_angle > 160:
                    pushup_stage = "up"
                    pushup_count += 1
            elif predicted_label == "Bicep Curl":
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                cv2.putText(image, str(left_elbow_angle),
                            tuple(np.multiply(left_elbow, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                if left_elbow_angle < 70 and right_elbow_angle < 70:
                    bicepCurl_stage = "down"
                if bicepCurl_stage == "down" and left_elbow_angle > 160 and right_elbow_angle > 160:
                    bicepCurl_stage = "up"
                    bicepCurl_count += 1


    cv2.putText(image, "Squat: {}".format(squat_count), (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(image, "PushUp: {}".format(pushup_count), (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.putText(image, "Bicep Curl: {}".format(bicepCurl_count), (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    image = cv2.resize(image, (640, 480))
    cv2.imshow('Mediapipe Feed', image)
    try:
        if key == ord('q'):
            break
    except:
        pass

cap.release()
cv2.destroyAllWindows()

