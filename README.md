# Engineering Drawing Object Detection & OCR System

> Hệ thống tự động phát hiện và trích xuất nội dung từ bản vẽ kỹ thuật  
> **Web Demo:** https://asuranosuke-engineering-drawing-demo.hf.space  
> **Model Weights:** https://huggingface.co/Asuranosuke/Detect-Info

---

## Tổng quan

Hệ thống nhận diện và trích xuất 3 loại đối tượng trong bản vẽ kỹ thuật:

| Class | Mô tả | Output |
|-------|-------|--------|
| `PartDrawing` | Vùng bản vẽ chi tiết kỹ thuật | `.png` |
| `Note` | Vùng ghi chú, chú thích | `.png` + `.txt` |
| `Table` | Vùng bảng dữ liệu kỹ thuật | `.png` + `.pdf` có cấu trúc |

---

## Cài đặt môi trường

### Yêu cầu
- Python 3.10
- CUDA (khuyến nghị để train, inference chạy được cả CPU)

### Cài đặt

```bash
git clone https://github.com/YOUR_USERNAME/engineering-drawing-demo
cd engineering-drawing-demo

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài torch CPU
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Cài Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Cài các thư viện còn lại
pip install -r requirements.txt
```

---

## Train Model

### Chuẩn bị dataset

1. Annotation bằng **LabelMe** với 3 class: `PartDrawing`, `Note`, `Table`
2. Convert sang COCO format:

```bash
python convert_labelme_to_coco.py \
    --image_dir data/images \
    --output_dir data/annotations \
    --val_ratio 0.15
```

Cấu trúc thư mục sau khi chuẩn bị:
```
Dữ liệu được upload lên Google Drive
data/
├── images/                # 58 ảnh bản vẽ kỹ thuật và file .json chứa thông tin gắn nhãn của sản phẩm
└── annotations/ <Code sẽ tự sinh ra folder>
    ├── annotations_train.json
    └── annotations_val.json
```

### Chạy train (khuyến nghị Google Colab T4 GPU)

```python
# Mở file engineering_drawing.ipynb trên Google Colab
`` Kết nối với T4 GPU và kết nối Google Drive

**Config train tốt nhất (v7):**

```python
cfg.MODEL.WEIGHTS  = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml" 
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 8000
cfg.SOLVER.STEPS   = (5600, 7200)
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
```

---
## Lưu ý: 
Nhập token của HuggingFace của bản thân khi chạy để có thể upload ngược lên hệ thống

## Download Model Weights

Model weights được lưu trên Hugging Face Hub, tự động download khi chạy inference:

```python
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(
    repo_id="Asuranosuke/Detect-Info",
    filename="model_final.pth"
)
```

Hoặc download thủ công:
- **HuggingFace:** https://huggingface.co/Asuranosuke/Detect-Info
- **File:** `model_final.pth` (553 MB, Cascade R-CNN ResNet-50 FPN)

---

## Chạy Inference Pipeline

### Inference 1 ảnh

```python
from inference import load_model, run_inference
import easyocr

# Load model
predictor = load_model("Asuranosuke/Detect-Info")
reader    = easyocr.Reader(["en", "vi"], gpu=False)

# Chạy inference
with open("drawing.jpg", "rb") as f:
    image_bytes = f.read()

json_result, vis_path = run_inference(
    image_bytes, predictor, reader,
    output_dir="./output"
)
print(json_result)
```

### Chạy Web Demo local

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
# Truy cập: http://localhost:8000
```

### Output: Hiện lên màn hình có các bbox gắn nhãn đi kèm với confidence score. Có hỗ trợ download kết quả và được Zip khi tải xuống 

```
output/
├── PartDrawing/
│   ├── partdrawing1.png
│   └── partdrawing2.png
├── Note/
│   ├── note1.png
│   └── note1.txt          # nội dung OCR
├── Table/
│   ├── table1.png
│   └── extract_table1.pdf # bảng có cấu trúc rows/cols
└── visualized.jpg         # ảnh gốc với bbox
```

JSON output mẫu:
```json
{
  "objects": [
    {
      "id": 1,
      "class": "Table",
      "confidence": 0.97,
      "bbox": {"x1": 120, "y1": 340, "x2": 680, "y2": 520},
      "ocr_content": "ITEM | QTY | DESCRIPTION\n1 | 2 | BUCKSTAY"
    }
  ]
}
```

---

## Kết quả đạt được

### Detection (val set — 8 ảnh)

| Metric | Kết quả |
|--------|---------|
| AP@[0.5:0.95] | **72.10%** |
| AP@0.50 | **85.11%** |
| AP@0.75 | **77.03%** |
| AP — PartDrawing | 65.07% |
| AP — Note | 60.94% |   
| AP — Table | **90.30%** |

# AP-Note còn thấp do số lượng data còn ít, nhiễu dữ liệu
---

## Approach & Phương pháp

### 1. Detection Model

Sau nhiều thử nghiệm, chọn **Cascade R-CNN + ResNet-50 + FPN** (Detectron2) vì:
- Cascade R-CNN dùng 3 tầng detector với IoU threshold tăng dần (0.5 → 0.6 → 0.7), bbox chính xác hơn Faster R-CNN ~4–6% AP@75
- Giấy phép thương mại compatible (Apache 2.0), không dùng YOLO

**Các cải tiến chính:**
- `RepeatFactorTrainingSampler` để oversample class `Note` (ít instance nhất)
- Aspect ratio `[0.25, 0.5, 1.0, 2.0, 4.0]` — bắt được Note dạng ngang rộng
- Augmentation: RandomFlip, RandomRotation ±10°, RandomBrightness/Contrast, multi-scale resize
- 8000 iterations với LR decay tại 70% và 90%

### 2. OCR Pipeline

**Note:** PaddleOCR v4 / EasyOCR với preprocessing:
- Upscale lên 1500px width (INTER_LANCZOS4)
- Denoise → CLAHE → Otsu threshold → Dilation

**Table:** img2table (detect cấu trúc) + EasyOCR (OCR từng cell):
- img2table tự nhận diện rows/columns từ đường kẻ bảng
- Fallback sang morphology detection nếu img2table thất bại
- Output: PDF giữ nguyên cấu trúc rows/columns

---

## Các thử nghiệm

| Lần | Model | Iter | Note AP | Table AP | AP tổng |
|-----|-------|------|---------|----------|---------|
| 1 | Faster R-CNN | 3000 | 49.6% | 78.7% | 63.0% |
| 2 | Faster R-CNN + augmentation | 5000 | 57.1% | 79.2% | 66.4% |
| 3 | Faster R-CNN + RepeatFactor | 3000 | 46.5% | 86.2% | 64.8% |
| **4** | **Cascade R-CNN + RepeatFactor** | **8000** | **60.9%** | **90.3%** | **72.1%** |

**Bài học rút ra:**
- Annotation chất lượng quan trọng hơn model phức tạp — Note AP thấp một phần do bbox annotation không nhất quán
- Cascade R-CNN tăng AP tổng ~5% so với Faster R-CNN cùng điều kiện
- `RepeatFactorTrainingSampler` hiệu quả hơn custom loss weight cho class imbalance

---

## Hướng cải thiện

**Detection:**
- Re-annotate lại toàn bộ Note với bbox đầy đủ hơn → kỳ vọng Note AP > 75%
- Tăng dataset lên 200+ ảnh bằng cách thu thập thêm hoặc synthetic augmentation
- Thử Cascade R-CNN với ResNet-101 backbone (AP +2–3% nhưng chậm hơn)

**OCR:**
- Tích hợp PaddleOCR v4 thay EasyOCR cho tiếng Anh — chính xác hơn ~10%
- Dùng Google Vision API làm fallback cho vùng có confidence thấp
- Post-processing: spell check với từ điển kỹ thuật (đơn vị đo, vật liệu...)

**System:**
- Deploy trên GPU instance để giảm inference time từ ~30s xuống ~3s
- Batch processing: xử lý nhiều ảnh cùng lúc
- Cache model weights trong container thay vì download mỗi lần khởi động

---

## Cấu trúc project

```
engineering-drawing-demo/
├── app.py              # FastAPI backend
├── inference.py        # Detection + OCR pipeline
├── requirements.txt
├── Dockerfile
├── static/
│   └── index.html      # Frontend UI
└── README.md
```

---

## Liên hệ

- **Web Demo:** https://asuranosuke-engineering-drawing-demo.hf.space
- **Model:** https://huggingface.co/Asuranosuke/Detect-Info
- **Email:** vanhvu1903vn@gmail.com
