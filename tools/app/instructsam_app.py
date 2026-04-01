# env: insam
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.InstructSAM.instruct_sam.counting import encode_image_as_base64, predict, extract_dict_from_string


app = FastAPI(title="InstructSAM Service")


class InstructSAMRequest(BaseModel):
    image_path: str
    query_text: str


class InstructSAMEngine:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = 'gpt-4o',
        max_concurrency: int = 1,
    ):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.prompt = """
        {{
            "Persona": "You are an advanced AI model capable of understanding and analyzing aerial images.",
            "Task": "Given an input satellite imagery, count the number of objects from specific categories. Provide the results in JSON format where the keys are the category names and the values are the corresponding counts.",
            "Instructions": [
                "The categories to count are: {target_objects}",
                "Count all visible instances of these objects in the image.",
                "If an object type is not visible, return 0 for that category.",
                "Be precise and thorough in your counting."],
            "Output format": "{{ \\"category1\\": count1, \\"category2\\": count2, ... }}",
            "Examples": [
                {{{example1}}},
                {{{example2}}}
            ]
        }}
        """

    def _infer(self, image_path: str, text_prompt: str):
        try:
            target_objects = [text_prompt]
            example1 = ", ".join([f'"{obj}": 0' for obj in target_objects])
            example2 = ", ".join(
                [f'"{obj}": 2' for obj in target_objects[:2]] +
                [f'"{obj}": 0' for obj in target_objects[2:]]
            )

            prompt = self.prompt.format(
                target_objects=target_objects,
                example1=example1,
                example2=example2
            )
            base64_image = encode_image_as_base64(image_path)
            completion = predict(prompt, base64_image,
                                top_p=1, temperature=0.01,
                                model=self.model, json_output=False)
            response = completion.choices[0].message.content
            pred_counts = extract_dict_from_string(response)

            return {
                "status": "success",
                "counts": pred_counts[text_prompt]
            }
        except:
            return {"status": "error", "message": "Error"}

    async def infer(self, image_path: str, query_text: str):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._infer, image_path, query_text
            )


# Initialize engine
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

engine = InstructSAMEngine(
    api_key=API_KEY,
    base_url=BASE_URL,
    model='gpt-4o',
    max_concurrency=1
)


@app.post("/instructsam/predict")
async def instructsam_predict(req: InstructSAMRequest):
    output = await engine.infer(req.image_path, req.query_text)

    return output