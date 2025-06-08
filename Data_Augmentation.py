import pandas as pd
import numpy as np
from Create_Data import mp_pose

df = pd.read_csv("pose_data.csv")

def jitter_pose(row, amount=0.03):
    noisy_row = row.copy()
    for i in range(1, len(row)):
        noisy_row[i] += np.random.uniform(-amount, amount)
    return noisy_row

def mirror_pose(row):
    mirrored = row.copy()
    left_indices = [i for i in range(33) if 'LEFT' in mp_pose.PoseLandmark(i).name]
    right_indices = [i for i in range(33) if 'RIGHT' in mp_pose.PoseLandmark(i).name]

    for li, ri in zip(left_indices, right_indices):
        for d in range(4):  # x, y, z, v
            mirrored[1 + 4 * li + d], mirrored[1 + 4 * ri + d] = (
                row[1 + 4 * ri + d],
                row[1 + 4 * li + d]
            )
    return mirrored

augmented_data = []

for _, row in df.iterrows():
    row = row.tolist()
    augmented_data.append(mirror_pose(row))
    augmented_data.append(jitter_pose(row))

df_aug = pd.DataFrame(augmented_data, columns=df.columns)
df_total = pd.concat([df, df_aug], ignore_index=True)
df_total.to_csv("pose_data_augmented.csv", index=False)

print("Đã tăng số mẫu từ {} lên {}".format(len(df), len(df_total)))
df_total = pd.read_csv("pose_data_augmented.csv")
print(df_total['label'].value_counts())