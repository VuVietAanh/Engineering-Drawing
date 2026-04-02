import base64
import json
import os
import shutil
import uuid

import cv2
import easyocr
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

from inference import load_model, run_inference

app = FastAPI()

# ── Load model khi khởi động ───────────────────────────────
HF_REPO = "Asuranosuke/Detect-Info"

print("Downloading model weights...")
weights_path = hf_hub_download(repo_id=HF_REPO, filename="model_final.pth")
ann_path     = hf_hub_download(repo_id=HF_REPO, filename="annotations_train.json")
print("Model downloaded!")

# Register dataset
if "drawing_demo" not in DatasetCatalog:
    register_coco_instances("drawing_demo", {}, ann_path, "")
    MetadataCatalog.get("drawing_demo").thing_classes = ["PartDrawing","Note","Table"]

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.MASK_ON                        = False
cfg.MODEL.ROI_HEADS.NUM_CLASSES          = 3
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = 0.5
cfg.MODEL.WEIGHTS                        = weights_path
cfg.MODEL.DEVICE                         = "cpu"

print("Loading predictor...")
predictor = DefaultPredictor(cfg)
META      = MetadataCatalog.get("drawing_demo")

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
    os.makedirs(f"{output_dir}/PartDrawing", exist_ok=True)
    os.makedirs(f"{output_dir}/Note",        exist_ok=True)
    os.makedirs(f"{output_dir}/Table",       exist_ok=True)

    # Đọc ảnh
    img_bytes = await file.read()
    nparr     = np.frombuffer(img_bytes, np.uint8)
    img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Inference
    outputs   = predictor(img)
    instances = outputs["instances"].to("cpu")
    boxes     = instances.pred_boxes.tensor.numpy()
    classes   = instances.pred_classes.numpy()
    scores    = instances.scores.numpy()

    # Visualize
    v       = Visualizer(img[:,:,::-1], metadata=META, scale=1.0,
                         instance_mode=ColorMode.SEGMENTATION)
    vis_img = v.draw_instance_predictions(instances).get_image()[:,:,::-1]
    vis_b64 = base64.b64encode(cv2.imencode(".jpg", vis_img)[1]).decode()

    # Pipeline
    json_result = run_pipeline(img, boxes, classes, scores,
                               output_dir, reader)

    # Zip output
    zip_path = f"/tmp/result_{session_id}.zip"
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