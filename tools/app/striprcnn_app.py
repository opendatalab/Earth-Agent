# env: RemoteSAM
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, "models/Strip-R-CNN")

import asyncio
import torch
import mmcv
import mmrotate
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from mmdet.apis import inference_detector, init_detector


app = FastAPI(title="StripRCNN Service")


class StripRCNNRequest(BaseModel):
    image_path: str


class StripRCNNEngine:
    def __init__(
        self,
        checkpoint: str,
        config_path: str,
        max_concurrency: int = 1,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = init_detector(config_path, checkpoint, device=self.device)

        self.semaphore = asyncio.Semaphore(max_concurrency)

    def convert_results_to_json(self, results, top_k=5):
        boxes = []

        dets = results[0] if isinstance(results, list) else results

        dets = dets[:top_k]

        for det in dets:
            box = {
                "cx": float(det[0]),
                "cy": float(det[1]),
                "w": float(det[2]),
                "h": float(det[3]),
                "angle": float(det[4])
            }
            boxes.append(box)

        return {"boxes": boxes}

    def _infer(self, image_path: str):
        result = inference_detector(self.model, image_path)

        output = self.convert_results_to_json(result)

        return {
            "status": "success",
            "result": output
        }

    async def infer(self, image_path: str):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._infer, image_path
            )


# Initialize engine
CHECKPOINT = "./models/Strip-R-CNN/checkpoint/stripnet_s.pth"
CONFIG_PATH = "./models/Strip-R-CNN/configs/strip_rcnn/strip_rcnn_s_fpn_3x_hrsc_le90.py"

engine = StripRCNNEngine(
    checkpoint=CHECKPOINT,
    config_path=CONFIG_PATH,
    max_concurrency=1
)


@app.post("/striprcnn/predict")
async def striprcnn_predict(req: StripRCNNRequest):
    output = await engine.infer(req.image_path)

    return output