import base64
import os
import shutil
import uuid

import easyocr
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference import load_model, run_inference

app = FastAPI()

HF_REPO = "Asuranosuke/Detect-Info"

print("Downloading model weights...")
predictor = load_model(HF_REPO)
print("Model downloaded!")

print("Loading EasyOCR...")
reader = easyocr.Reader(["en", "vi"], gpu=False)
print("All ready!")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    output_dir = f"/tmp/result_{session_id}"

    image_bytes = await file.read()

    json_result, vis_path = run_inference(
        image_bytes, predictor, reader, output_dir
    )

    with open(vis_path, "rb") as f:
        vis_b64 = base64.b64encode(f.read()).decode()

    shutil.make_archive(f"/tmp/result_{session_id}", "zip", output_dir)

    return JSONResponse({
        "session_id": session_id,
        "result":     json_result,
        "vis_image":  vis_b64,
    })

@app.get("/download/{session_id}")
def download(session_id: str):
    zip_path = f"/tmp/result_{session_id}.zip"
    return FileResponse(zip_path,
                        filename=f"result_{session_id}.zip",
                        media_type="application/zip")