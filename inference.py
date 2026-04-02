import json
import os

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from huggingface_hub import hf_hub_download
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.platypus import Table as RLTable
from reportlab.platypus import TableStyle

CLASS_NAMES = ["PartDrawing", "Note", "Table"]

def load_model(repo_id: str, filename: str = "model_final.pth"):
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.MASK_ON                        = False
    cfg.MODEL.WEIGHTS                        = weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = 0.5
    cfg.MODEL.DEVICE                         = "cpu"
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
    return DefaultPredictor(cfg)

def preprocess_note(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1500:
        gray = cv2.resize(gray, None, fx=1500/w, fy=1500/w,
                          interpolation=cv2.INTER_LANCZOS4)
    gray  = cv2.fastNlMeansDenoising(gray, h=20,
                                     templateWindowSize=7,
                                     searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray  = clahe.apply(gray)
    _, b  = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
    return cv2.dilate(b, k, iterations=1)

def preprocess_table(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 2000:
        gray = cv2.resize(gray, None, fx=2000/w, fy=2000/w,
                          interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray  = clahe.apply(gray)
    gray  = cv2.bilateralFilter(gray, 9, 75, 75)
    _, b  = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)

def detect_cells(binary):
    h, w = binary.shape
    inv  = cv2.bitwise_not(binary)
    hk   = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w//10,60),1))
    vk   = cv2.getStructuringElement(cv2.MORPH_RECT, (1,max(h//10,30)))
    grid = cv2.add(
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk, iterations=3),
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk, iterations=3)
    )
    grid = cv2.dilate(grid,
                      cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),
                      iterations=2)
    cnts,_ = cv2.findContours(grid, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
    cells  = [(x,y,x+cw,y+ch) for c in cnts
              for x,y,cw,ch in [cv2.boundingRect(c)]
              if cw*ch > h*w*0.001 and cw>40 and ch>15]
    if len(cells) < 2:
        return None
    cells = sorted(cells, key=lambda c:(c[1],c[0]))
    rows, cur, tol = [], [cells[0]], h*0.025
    for cell in cells[1:]:
        if abs(cell[1]-cur[0][1]) < tol:
            cur.append(cell)
        else:
            rows.append(sorted(cur, key=lambda c:c[0]))
            cur = [cell]
    rows.append(sorted(cur, key=lambda c:c[0]))
    return rows

def extract_table_img2table(crop_img, reader):
    """Dùng img2table detect cấu trúc + OCR từng cell."""
    import tempfile

    from img2table.document import Image as Img2TableImage
    from img2table.ocr import EasyOCR as Img2TableEasyOCR

    try:
        # Lưu crop ra file tạm
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, crop_img)
        tmp.close()

        ocr_engine = Img2TableEasyOCR(reader=reader)
        doc        = Img2TableImage(src=tmp.name)
        tables     = doc.extract_tables(
            ocr=ocr_engine,
            implicit_rows=True,
            implicit_columns=False,
            borderless_tables=False,
            min_confidence=40,
        )
        os.unlink(tmp.name)

        if not tables:
            return None

        # Lấy bảng lớn nhất
        best = max(tables, key=lambda t: t.df.shape[0] * t.df.shape[1])
        df   = best.df

        rows = []
        # Header
        if list(df.columns) != list(range(len(df.columns))):
            rows.append([str(c) for c in df.columns])
        # Data
        for _, row in df.iterrows():
            rows.append([str(v) if v is not None else "" for v in row])
        return rows

    except Exception as e:
        print(f"img2table error: {e}")
        return None

def save_table_pdf(table_data, out_path, title="Table"):
    if not table_data:
        return
    doc    = SimpleDocTemplate(str(out_path), pagesize=A4,
                               leftMargin=0.8*cm, rightMargin=0.8*cm,
                               topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    n_cols = max(len(r) for r in table_data)
    norm   = [r + [""]*(n_cols-len(r)) for r in table_data]
    pw     = A4[0] - 1.6*cm
    wrap   = [[Paragraph(str(c), styles["Normal"]) for c in row]
              for row in norm]
    tbl    = RLTable(wrap, colWidths=[pw/n_cols]*n_cols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#1F4E79")),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white, colors.HexColor("#EBF3FB")]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#888888")),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("RIGHTPADDING",  (0,0),(-1,-1), 5),
    ]))
    doc.build([Paragraph(f"<b>{title}</b>", styles["Heading2"]),
               Spacer(1, 0.3*cm), tbl])

def run_inference(image_bytes: bytes, predictor, reader,
                  output_dir: str):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    os.makedirs(f"{output_dir}/PartDrawing", exist_ok=True)
    os.makedirs(f"{output_dir}/Note",        exist_ok=True)
    os.makedirs(f"{output_dir}/Table",       exist_ok=True)

    outputs   = predictor(img)
    instances = outputs["instances"].to("cpu")
    boxes     = instances.pred_boxes.tensor.numpy()
    classes   = instances.pred_classes.numpy()
    scores    = instances.scores.numpy()

    # Visualize
    try:
        meta = MetadataCatalog.get("drawing_demo")
        meta.thing_classes = CLASS_NAMES  # ← luôn set lại
    except Exception:
        DatasetCatalog.register("drawing_demo", lambda: [])
        meta = MetadataCatalog.get("drawing_demo")
        meta.thing_classes = CLASS_NAMES

    v       = Visualizer(img[:,:,::-1], metadata=meta, scale=1.0,
                         instance_mode=ColorMode.SEGMENTATION)
    vis_img = v.draw_instance_predictions(instances).get_image()[:,:,::-1]
    vis_path = f"{output_dir}/visualized.jpg"
    cv2.imwrite(vis_path, vis_img)

    counters    = {c: 0 for c in CLASS_NAMES}
    json_result = {"objects": []}

    for i, (box, cls_id, score) in enumerate(
            zip(boxes, classes, scores)):
        x1,y1,x2,y2 = map(int, box)
        cls  = CLASS_NAMES[cls_id]
        crop = img[y1:y2, x1:x2]
        counters[cls] += 1
        idx = counters[cls]
        ocr_content = None

        if cls == "PartDrawing":
            cv2.imwrite(
                f"{output_dir}/PartDrawing/partdrawing{idx}.png", crop)

        elif cls == "Note":
            cv2.imwrite(f"{output_dir}/Note/note{idx}.png", crop)
            proc    = preprocess_note(crop)
            results = sorted(reader.readtext(proc, detail=1),
                             key=lambda r: r[0][0][1])
            lines   = [t for (_,t,c) in results if c > 0.3]
            ocr_content = "\n".join(lines)
            with open(f"{output_dir}/Note/note{idx}.txt",
                      "w", encoding="utf-8") as f:
                f.write(ocr_content)

        elif cls == "Table":
            cv2.imwrite(f"{output_dir}/Table/table{idx}.png", crop)
            proc = preprocess_table(crop)

            # Thử img2table trước
            table_data = extract_table_img2table(crop, reader)

            # Fallback morphology nếu img2table thất bại
            if not table_data or len(table_data) < 2:
                rows = detect_cells(proc)
                if rows:
                    table_data = ocr_table(proc, rows, reader)
                else:
                    res        = sorted(reader.readtext(proc, detail=1),
                                        key=lambda r: r[0][0][1])
                    table_data = [[t] for (_,t,c) in res if c > 0.3]

            ocr_content = "\n".join(" | ".join(r) for r in table_data)
            save_table_pdf(table_data,
                        f"{output_dir}/Table/extract_table{idx}.pdf",
                        title=f"Table {idx}")

        json_result["objects"].append({
            "id":          i + 1,
            "class":       cls,
            "confidence":  round(float(score), 4),
            "bbox":        {"x1":x1,"y1":y1,"x2":x2,"y2":y2},
            "ocr_content": ocr_content,
        })

    return json_result, vis_path