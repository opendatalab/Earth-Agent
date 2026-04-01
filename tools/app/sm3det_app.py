# env: RemoteSAM
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, "models/SM3Det")

import asyncio
import torch
import mmcv
import mmrotate
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from mmdet.apis import inference_detector, init_detector


app = FastAPI(title="SM3Det Service")


class SM3DetRequest(BaseModel):
    image_path: str


class SM3DetEngine:
    def __init__(
        self,
        checkpoint: str,
        config_path: str,
        score_thr: float = 0.3,
        max_concurrency: int = 1,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = init_detector(config_path, checkpoint, device=self.device)

        self.score_thr = score_thr
        self.classes = [
            'helipad', 'helicopter', 'person', 'container-crane', 'vehicle',
            'airport', 'small-vehicle', 'large-vehicle', 'plane', 'ship',
            'harbor', 'tennis-court', 'soccer-ball-field', 'ground-track-field', 'baseball-diamond',
            'swimming-pool', 'roundabout', 'basketball-court', 'storage-tank', 'bridge',
            'building', 'truck', 'car', 'bus', 'tank', 'excavator'
        ]
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _infer(self, image_path: str):
        image = cv2.imread(image_path)
        img_h, img_w = image.shape[:2]

        result = inference_detector(self.model, image)

        detections = []
        all_bboxes = []
        all_scores = []
        all_labels = []

        for class_idx, class_result in enumerate(result):
            if len(class_result) > 0:
                for detection in class_result:
                    if len(detection) >= 6:
                        x, y, w, h, angle, score = detection[:6]

                        if score >= self.score_thr:
                            cos_a = np.cos(np.deg2rad(angle))
                            sin_a = np.sin(np.deg2rad(angle))

                            corners = np.array([
                                [-w/2, -h/2], [w/2, -h/2],
                                [w/2, h/2], [-w/2, h/2]
                            ])

                            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                            rotated_corners = corners @ R.T + np.array([x, y])

                            area = float(w * h)
                            aspect_ratio = float(w / h) if h > 0 else 0

                            detections.append({
                                'class': self.classes[class_idx],
                                'class_id': class_idx,
                                'confidence': float(score),
                                'bbox': {
                                    'center_x': float(x),
                                    'center_y': float(y),
                                    'width': float(w),
                                    'height': float(h),
                                    'angle': float(angle),
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'corners': rotated_corners.tolist()
                                }
                            })

                            all_bboxes.append([x, y, w, h, angle])
                            all_scores.append(score)
                            all_labels.append(class_idx)

        detections.sort(key=lambda x: x['confidence'], reverse=True)

        if detections:
            confidences = [d['confidence'] for d in detections]
            areas = [d['bbox']['area'] for d in detections]
            class_counts = {}
            for d in detections:
                class_name = d['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            stats = {
                'max_confidence': float(np.max(confidences)),
                'min_confidence': float(np.min(confidences)),
                'avg_confidence': float(np.mean(confidences)),
                'max_area': float(np.max(areas)),
                'min_area': float(np.min(areas)),
                'avg_area': float(np.mean(areas)),
                'class_counts': class_counts
            }
        else:
            stats = {}

        result_data = {
            'image_path': image_path,
            'image_size': {'width': img_w, 'height': img_h},
            'num_detections': len(detections),
            'detections': detections,
            'statistics': stats
        }

        return {
            "status": "success",
            "result": result_data
        }

    async def infer(self, image_path: str):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._infer, image_path
            )


# Initialize engine
CHECKPOINT = "./models/SM3Det/checkpoint/iter_33468.pth"
CONFIG_PATH = "./models/SM3Det/configs/SM3Det/SM3Det_convnext_b.py"

engine = SM3DetEngine(
    checkpoint=CHECKPOINT,
    config_path=CONFIG_PATH,
    score_thr=0.3,
    max_concurrency=1
)


@app.post("/sm3det/predict")
async def sm3det_predict(req: SM3DetRequest):
    output = await engine.infer(req.image_path)

    return output