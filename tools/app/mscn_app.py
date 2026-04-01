# env: RemoteSAM
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, "models/MSCN")

import asyncio
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from models.MSCN.network.MSCN import MSCN


app = FastAPI(title="MSCN Service")


class MSCNRequest(BaseModel):
    image_path: str
    top_k: Optional[int] = 5


class MSCNEngine:
    CLASSES = [
        'Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church',
        'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial',
        'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground',
        'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential',
        'Square', 'Stadium', 'StorageTanks', 'Viaduct'
    ]

    def __init__(
        self,
        checkpoint: str,
        num_classes: int = 30,
        res: int = 50,
        k1: int = 2,
        k2: int = 3,
        g: int = 8,
        max_concurrency: int = 1
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        self.res = res
        self.k1 = k1
        self.k2 = k2
        self.g = g

        self.model = MSCN(
            num_classes=num_classes,
            res=res,
            k1=k1,
            k2=k2,
            g=g
        )

        checkpoint = torch.load(checkpoint, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _infer(self, image_path: str, top_k: int = 5):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).cuda()

            # Inference
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]

            # Build output
            output = ''
            for prob, idx in zip(top_probs, top_indices):
                class_name = self.CLASSES[idx] if idx < len(self.CLASSES) else f"class_{idx}"
                output += f"{class_name:<40} {prob * 100:5.1f}%\n"

            return {
                "status": "success",
                "output": output,
                "predictions": [
                    {"class": self.CLASSES[idx] if idx < len(self.CLASSES) else f"class_{idx}",
                     "confidence": float(prob)}
                    for prob, idx in zip(top_probs, top_indices)
                ]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def infer(self, image_path: str, text_queries=None):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            top_k = 5
            if text_queries is not None:
                if isinstance(text_queries, int):
                    top_k = text_queries
                elif isinstance(text_queries, str):
                    try:
                        top_k = int(text_queries)
                    except:
                        pass
            return await loop.run_in_executor(None, self._infer, image_path, top_k)


# Initialize engine
CHECKPOINT = "./models/MSCN/ckpt/AID_0_97.14.pth"

engine = MSCNEngine(
    checkpoint=CHECKPOINT,
    num_classes=30,
    res=50,
    k1=2,
    k2=3,
    g=8,
    max_concurrency=1
)


@app.post("/mscn/predict")
async def mscn_predict(req: MSCNRequest):
    output = await engine.infer(req.image_path, req.top_k)

    return output