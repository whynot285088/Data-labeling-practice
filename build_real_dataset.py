"""
build_real_dataset.py
----------------------
Kaggle 'Car License Plate Detection' 데이터셋(433장)을 이용해
실제 하이파킹 업무와 동일한 구조의 검증용 데이터를 만든다.

1) XML(PASCAL VOC) 파싱 -> 실제(ground truth) 번호판 박스 추출
2) 실제 박스를 크롭해서 별도 이미지로 저장
3) 크롭 이미지에 진짜 OCR(tesseract) 실행 -> "AI 인식 결과" (recognized_plate)
   * 실제 번호판 "정답 텍스트"는 원래 데이터셋에 없음 -> 사람이 크롭 이미지를 보고
     직접 입력해야 하는 부분. 이게 바로 여러분이 실무에서 하게 될 일.
4) 정답 박스를 무작위로 변형(jitter/누락/오탐 추가)해서 "AI 탐지 결과"를 시뮬레이션
   -> IoU 계산 -> TP/FP/FN 판정 (실제 이미지 크기 기준의 진짜 IoU)
5) 사람이 채워야 할 "라벨링용 CSV" (actual_plate 빈 칸) 생성
6) GT박스(초록) vs 시뮬레이션 예측박스(빨강)를 그린 예시 이미지 8장 저장 (IoU 값 표기)
"""
import os
import argparse
import random
import xml.etree.ElementTree as ET
import pandas as pd
import pytesseract
from PIL import Image, ImageDraw, ImageFont

random.seed(7)

parser = argparse.ArgumentParser(description="Kaggle Car License Plate Detection 데이터로 검증용 워크시트 생성")
parser.add_argument("--raw_dir", default="./data/raw", help="Kaggle 데이터셋 압축 해제 경로 (images/, annotations/ 포함)")
parser.add_argument("--out_dir", default="./output", help="결과물(크롭, 워크시트 등) 저장 경로")
args = parser.parse_args()

RAW_DIR = args.raw_dir
IMG_DIR = os.path.join(RAW_DIR, "images")
ANN_DIR = os.path.join(RAW_DIR, "annotations")
OUT_DIR = args.out_dir
CROP_DIR = os.path.join(OUT_DIR, "plate_crops")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlay_examples")
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)


def iou(box_a, box_b):
    """두 박스(xmin,ymin,xmax,ymax)의 IoU(Intersection over Union) 계산"""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append((
            int(float(b.find("xmin").text)), int(float(b.find("ymin").text)),
            int(float(b.find("xmax").text)), int(float(b.find("ymax").text)),
        ))
    return filename, w, h, boxes


def jitter_box(box, w, h, scale=0.06):
    """실제 박스를 살짝 흔들어서 '완벽하진 않은 AI 예측'을 흉내낸다"""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx1 = random.uniform(-scale, scale) * bw
    dy1 = random.uniform(-scale, scale) * bh
    dx2 = random.uniform(-scale, scale) * bw
    dy2 = random.uniform(-scale, scale) * bh
    nx1 = max(0, min(w, x1 + dx1))
    ny1 = max(0, min(h, y1 + dy1))
    nx2 = max(0, min(w, x2 + dx2))
    ny2 = max(0, min(h, y2 + dy2))
    return (int(nx1), int(ny1), int(nx2), int(ny2))


def random_fp_box(w, h):
    """실제로는 없는데 AI가 엉뚱한 곳을 번호판으로 오검출한 상황을 흉내"""
    bw, bh = random.randint(40, 100), random.randint(15, 35)
    x1 = random.randint(0, max(1, w - bw))
    y1 = random.randint(0, max(1, h - bh))
    return (x1, y1, x1 + bw, y1 + bh)


records = []
event_id = 1
xml_files = sorted(os.listdir(ANN_DIR))

for xml_name in xml_files:
    filename, w, h, gt_boxes = parse_xml(os.path.join(ANN_DIR, xml_name))
    img_path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")

    # --- AI 탐지 결과 시뮬레이션: 90% 정상 검출(약간의 위치오차), 10% 미검출 ---
    pred_boxes = []
    for gb in gt_boxes:
        if random.random() < 0.90:
            pred_boxes.append(jitter_box(gb, w, h, scale=0.05))
        # 10%는 미검출(FN) -> pred 없음

    # 이미지 12장 중 1장 꼴로 엉뚱한 오검출(FP) 박스 하나 추가
    extra_fp = random.random() < 0.08
    if extra_fp:
        pred_boxes.append(random_fp_box(w, h))

    # --- 매칭: 각 GT를 IoU 최대인 예측박스와 매칭 (IoU>=0.5 => TP) ---
    matched_pred_idx = set()
    for gi, gb in enumerate(gt_boxes):
        best_iou, best_j = 0.0, -1
        for pj, pb in enumerate(pred_boxes):
            if pj in matched_pred_idx:
                continue
            v = iou(gb, pb)
            if v > best_iou:
                best_iou, best_j = v, pj

        if best_j >= 0 and best_iou >= 0.5:
            status = "TP"
            matched_pred_idx.add(best_j)
            pred_box_used = pred_boxes[best_j]
        else:
            status = "FN"  # 실제 있는데 AI가 못 찾음(또는 겹침이 너무 적어 인정 안 됨)
            pred_box_used = None

        # --- 번호판 크롭 + 실제 OCR 실행 (AI 인식 결과) ---
        crop = img.crop(gb)
        crop_name = f"{filename.replace('.png','')}_{gi}.png"
        crop_path = os.path.join(CROP_DIR, crop_name)
        crop_upscaled = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
        crop_upscaled.save(crop_path)

        ocr_raw = pytesseract.image_to_string(
            crop_upscaled, config="--psm 7"
        ).strip().replace(" ", "").replace("\n", "")

        records.append({
            "event_id": event_id,
            "source_image": filename,
            "crop_file": crop_name,
            "img_w": w, "img_h": h,
            "gt_box": gb,
            "pred_box": pred_box_used,
            "iou": round(best_iou, 3) if best_j >= 0 else 0.0,
            "detection_status": status,
            "ai_recognized_plate": ocr_raw,
            "actual_plate": "",  # <- 사람이 크롭 이미지를 보고 직접 채워야 하는 칸
        })
        event_id += 1

    # --- FP: 매칭되지 않은 예측 박스(오검출) ---
    for pj, pb in enumerate(pred_boxes):
        if pj in matched_pred_idx:
            continue
        records.append({
            "event_id": event_id,
            "source_image": filename,
            "crop_file": "",
            "img_w": w, "img_h": h,
            "gt_box": None,
            "pred_box": pb,
            "iou": 0.0,
            "detection_status": "FP",
            "ai_recognized_plate": "",
            "actual_plate": "",
        })
        event_id += 1

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "labeling_worksheet.csv"), index=False, encoding="utf-8-sig")

print(f"총 {len(df)}건 생성 (GT 이미지 {len(xml_files)}장 기준)")
print(df["detection_status"].value_counts())
print(f"\n크롭 이미지: {CROP_DIR}")
print(f"라벨링 워크시트: {OUT_DIR}/labeling_worksheet.csv")

# ---- 시각화 예시 8장: 초록=실제(GT), 빨강=AI예측, IoU 표기 ----
sample_files = random.sample([x for x in xml_files if x], 8)
for xml_name in sample_files:
    filename, w, h, gt_boxes = parse_xml(os.path.join(ANN_DIR, xml_name))
    img = Image.open(os.path.join(IMG_DIR, filename)).convert("RGB")
    draw = ImageDraw.Draw(img)
    sub = df[df["source_image"] == filename]
    for _, row in sub.iterrows():
        if row["gt_box"] is not None and isinstance(row["gt_box"], tuple):
            draw.rectangle(row["gt_box"], outline=(0, 220, 0), width=3)
            if row["detection_status"] == "FN":
                draw.text((row["gt_box"][0], max(0, row["gt_box"][1] - 14)), "FN (미검출)", fill=(255, 220, 0))
        if row["pred_box"] is not None and isinstance(row["pred_box"], tuple):
            draw.rectangle(row["pred_box"], outline=(230, 40, 40), width=2)
            label = f'{row["detection_status"]} IoU={row["iou"]}'
            draw.text((row["pred_box"][0], max(0, row["pred_box"][1] - 14)), label, fill=(255, 220, 0))
    img.save(os.path.join(OVERLAY_DIR, filename))

print(f"시각화 예시 8장: {OVERLAY_DIR}")
