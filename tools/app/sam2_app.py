# env: fastapi
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import asyncio
import torch
import numpy as np
import uuid
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


app = FastAPI(title="SAM2 Service")


class Sam2Request(BaseModel):
    image_path: str
    bbox: Optional[list] = None
    output_path: Optional[str] = None


class SAM2Engine:
    def __init__(
        self,
        checkpoint: str,
        model_cfg: str,
        max_concurrency: int = 1,
        output_dir: str = "tmp"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_sam2(model_cfg, checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.predictor = SAM2ImagePredictor(self.model)
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _apply_color_map(self, masks):
        if len(masks) == 0:
            return None

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        h, w = masks.shape[-2:]
        segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(len(masks)):
            mask = masks[i]
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            segmentation_map[mask > 0] = color

        return segmentation_map

    def _infer(self, image_path: str, bbox: list = None, output_path: str = None):
        # Load image from file path
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "message": f"Failed to read image: {image_path}"}
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            self.predictor.set_image(image)

            if bbox is not None:
                # Use bbox for segmentation
                x1, y1, x2, y2 = bbox
                input_box = np.array([x1, y1, x2, y2])
                masks, scores, logits = self.predictor.predict(
                    box=input_box[None, :],
                    multimask_output=False
                )
            else:
                masks, scores, logits = self.predictor.predict()

        seg_map = self._apply_color_map(masks)

        if seg_map is not None:
            if output_path:
                file_path = os.path.abspath(output_path)
            else:
                file_name = f"seg_{uuid.uuid4().hex}.png"
                file_path = os.path.abspath(os.path.join(self.output_dir, file_name))
            Image.fromarray(seg_map).save(file_path)

            return {
                "status": "success",
                "mask_path": file_path,
                "scores": scores.tolist() if hasattr(scores, 'tolist') else scores
            }

        return {"status": "error", "message": "No masks generated"}

    async def infer(self, image_path: str, bbox: list = None, output_path: str = None):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._infer, image_path, bbox, output_path
            )


# Initialize engine
CHECKPOINT = "models/sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

engine = SAM2Engine(
    checkpoint=CHECKPOINT,
    model_cfg=MODEL_CFG,
    max_concurrency=1
)


@app.post("/sam2/predict")
async def sam2_predict(req: Sam2Request):
    output = await engine.infer(req.image_path, req.bbox, req.output_path)

    return output