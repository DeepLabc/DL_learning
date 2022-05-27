"""
This script to create dataset and labels by clean off some NaN, do a normalization,
label smoothing and label weights by scores.

"""
import numpy as np
import pandas as pd
import pickle


class_names = ['Stand', 'Walk', 'Run', "Lay", "Eat"]

main_parts = [ 'Left_eye_x', 'Left_eye_y', 'Left_eye_s',
            'Right_eye_x', 'Right_eye_y', 'Right_eye_s',
            'Nose_x', 'Nose_y', 'Nose_s',
            'Neck_x', 'Neck_y', 'Neck_s',
            'Root_of_Tail_x', 'Root_of_Tail_y', 'Root_of_Tail_s',
            'Left_Shoulder_x', 'Left_Shoulder_y', 'Left_Shoulder_s',
            'Left_Elbow_x', 'Left_Elbow_y', 'Left_Elbow_s',
            'Left_Front_Paw_x', 'Left_Front_Paw_y', 'Left_Front_Paw_s',
            'Right_Shoulder_x', 'Right_Shoulder_y', 'Right_Shoulder_s',
            'Rgiht_Elbow_x', 'Rgiht_Elbow_y', 'Rgiht_Elbow_s',
            'Right_Front_Paw_x', 'Right_Front_Paw_y', 'Right_Front_Paw_s',
            'Left_Hip_x', 'Left_Hip_y', 'Left_Hip_s',
            'Left_Knee_x', 'Left_Knee_y', 'Left_Knee_s',
            'Left_Back_Paw_x', 'Left_Back_Paw_y', 'Left_Back_Paw_s',
            'Right_Hip_x', 'Right_Hip_y', 'Right_Hip_s',
            'Right_Knee_x', 'Right_Knee_y', 'Right_Knee_s',
            'Right_Back_Paw_x', 'Right_Back_Paw_y', 'Right_Back_Paw_s']

main_idx_parts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # 1.5

csv_pose_file = './animal/pose_train.csv'
save_path = './animal/pose_action_all.pkl'

# Params.
smooth_labels_step = 8
n_frames = 10
skip_frame = 1

annot = pd.read_csv(csv_pose_file)

# Remove NaN.
idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
idx = np.where(idx)[0]
annot = annot.drop(idx)
# One-Hot Labels.
label_onehot = pd.get_dummies(annot['label'])

annot = annot.drop('label', axis=1).join(label_onehot)
cols = label_onehot.columns.values  # 0123



def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def seq_label_smoothing(labels, max_step=10):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
   
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue
        # print(np.argmax(labels[i]))
        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels


feature_set = np.empty((0, n_frames, 18, 3))  # 此处改动  0,10,18,3

labels_set = np.empty((0, len(cols)))
vid_list = annot['video'].unique()
for vid in vid_list:
    print(f'Process on: {vid}')
    data = annot[annot['video'] == vid].reset_index(drop=True).drop(columns='video')
    # Label Smoothing.
    esp = 0.1
    data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1)
    data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)

    # Separate continuous frames.
    frames = data['frame'].values
    frames_set = []
    fs = [0]
    for i in range(1, len(frames)):
        if frames[i] < frames[i-1] + 10:
            fs.append(i)
        else:
            frames_set.append(fs)
            fs = [i]
    frames_set.append(fs)

    for fs in frames_set:
        xys = data.iloc[fs, 1:-len(cols)].values.reshape(-1, 17, 3)     # 此处改动
        
        # Scale pose normalize.
        xys[:, :, :2] = scale_pose(xys[:, :, :2])
        # Add center point.
        xys = np.concatenate((xys, np.expand_dims((xys[:, 3, :] + xys[:, 4, :]) / 2, 1)), axis=1)  # 此处改动

        # Weighting main parts score.
        scr = xys[:, :, -1].copy()
        
        scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
        # Mean score.
        scr = scr.mean(1)
        
        # Targets.
        lb = data.iloc[fs, -len(cols):].values
        # Apply points score mean to all labels.
        lb = lb * scr[:, None]
    
        for i in range(xys.shape[0] - n_frames):
            feature_set = np.append(feature_set, xys[i:i+n_frames][None, ...], axis=0)
            labels_set = np.append(labels_set, lb[i:i+n_frames].mean(0)[None, ...], axis=0)
        # print(feature_set.shape)
            




with open(save_path, 'wb') as f:
    pickle.dump((feature_set, labels_set), f)
