# env: insam
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="RemoteCLIP Service")


class RemoteCLIPRequest(BaseModel):
    image_path: str
    text_queries: Optional[str] = None


class RemoteCLIPEngine:
    def __init__(
        self,
        checkpoint: str,
        max_concurrency: int = 1
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes = [
            "Airport", "Beach", "Bridge", "Commercial", "Desert", "Farmland",
            "footballField", "Forest", "Industrial", "Meadow", "Mountain",
            "Park", "Parking", "Pond", "Port", "railwayStation", "Residential",
            "River", "Viaduct"
        ]

        model_name = os.path.basename(checkpoint).removesuffix('.pt').removeprefix('RemoteCLIP-')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.model = self.model.cuda().eval()
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _infer(self, image_path: str, text_queries: list | str | None=None):
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        if text_queries is None:
            text_queries = self.classes

        text = self.tokenizer(text_queries)
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image.cuda())
            text_features = self.model.encode_text(text.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

        output = ''
        for query, prob in zip(text_queries, text_probs):
            output += f"{query:<40} {prob * 100:5.1f}%\n"

        return {
            "status": "success",
            "output": output
        }

    async def infer(self, image_path: str, text_queries: list | str | None=None):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._infer, image_path, text_queries
            )


# Initialize engine
CHECKPOINT = "./models/RemoteCLIP/checkpoint/RemoteCLIP-ViT-L-14.pt"

engine = RemoteCLIPEngine(
    checkpoint=CHECKPOINT,
    max_concurrency=1
)


@app.post("/remoteclip/predict")
async def remoteclip_predict(req: RemoteCLIPRequest):
    output = await engine.infer(req.image_path, req.text_queries)

    return output