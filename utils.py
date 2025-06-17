import torch.nn.functional as F
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as T
import os
import json
import torch
from tqdm import tqdm
import shutil
import numpy as np

def time_to_seconds(t):
    try:
        parts = list(map(int, t.strip().split(":")))
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            return None
        return h * 3600 + m * 60 + s
    except:
        return None

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return int(frame_count / fps)
    return None

def extract_dino_features(video_path, fps=1):
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2.eval().cuda()

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    true_fps = cap.get(cv2.CAP_PROP_FPS)
    if true_fps == 0:
        return None

    frame_interval = int(true_fps // fps)
    frame_id = 0

    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img).unsqueeze(0).cuda()

            with torch.no_grad():
                feat = dinov2(img_tensor)
                features.append(feat.squeeze(0).cpu())

        frame_id += 1

    cap.release()

    if not features:
        return None

    return torch.stack(features)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = self.fc(unpacked).squeeze(-1)
        return logits

@torch.no_grad()
def predict_on_video(model, feats, threshold=0.5):
    model.eval()
    lengths = torch.tensor([feats.shape[1]])
    
    logits = model(feats, lengths)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    pred_mask = (probs >= threshold).astype(int)

    return probs, pred_mask

def mask_to_intervals(mask, fps=1):
    intervals = []
    start = None
    for i, val in enumerate(mask):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            intervals.append((start, i - 1))
            start = None            
    if start is not None:
        intervals.append((start, len(mask) - 1))
    
    return [(s / fps, e / fps) for s, e in intervals]

def get_main_interval_from_mask(mask, fps=1):
    intervals = mask_to_intervals(mask, fps)
    if not intervals:
        return None
    return max(intervals, key=lambda x: x[1] - x[0])

def get_opening_interval(video_path, threshold=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BiLSTMClassifier().to(device)
    model.load_state_dict(torch.load("zastavka_model.pt"))
    model.eval()
    feats = extract_dino_features(video_path).unsqueeze(0).to(device)
    _, pred_mask = predict_on_video(model, feats, threshold=0.5)
    return get_main_interval_from_mask(pred_mask)

def seconds_to_time(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02}:{s:02}"

def print_opening_interval(video_path, threshold=0.5):
    start, end = get_opening_interval(video_path, threshold)
    return f"{seconds_to_time(start)} -- {seconds_to_time(end)}"