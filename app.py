from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import tempfile
from pipeline import analyze_image
import uvicorn


class ImageRequest(BaseModel):
    image_urls: List[str]


app = FastAPI()


@app.post("/analyze")
async def analyze(req: ImageRequest):
    results = []

    for url in req.image_urls:
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Save to temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(response.content)
        tmp.close()

        # Run pipeline
        result = analyze_image(tmp.name)
        results.append({
            "url": url,
            "result": result
        })

    return {"results": results}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
