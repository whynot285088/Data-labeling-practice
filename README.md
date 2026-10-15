# 주차장 AI 검출·인식 검증 실습

주차 관리 AI 시스템에서 사람이 하는 "AI 결과 vs 실제값" 검증 업무를 연습한 프로젝트입니다.

## 데이터셋
[Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) (Kaggle, CC0)
433장의 차량 사진 + 번호판 위치(bounding box) 라벨. 저작권 문제로 원본 이미지는 포함하지 않았습니다.
`build_real_dataset.py` 실행 전 위 링크에서 다운로드해 `data/raw/`에 압축을 푸세요.

## 무엇을 진짜로 했고, 무엇을 시뮬레이션했는지
| 항목 | 실제 AI인가? |
|---|---|
| 번호판 텍스트 인식 (`ai_recognized_plate`) | ✅ 실제 OCR 엔진(Tesseract)을 코드로 실행한 결과 |
| 차량 검출 위치 (`pred_box`, TP/FP/FN) | ❌ 실제 탐지 모델이 없어 정답 박스를 임의로 변형해 시뮬레이션 |

## 과정
1. `build_real_dataset.py` — XML 정답 라벨 파싱 → 번호판 크롭 → OCR 실행 → 검출 시뮬레이션(IoU 계산) → `labeling_worksheet.csv` 생성
2. 크롭 이미지를 눈으로 직접 보고 실제 번호판 값을 `actual_plate` 컬럼에 라벨링 (32건)
3. `verify_real.py` — 사람이 입력한 정답과 OCR 결과를 비교해 정인식/오인식/미인식 집계

## 결과 (32건 기준, 진행 중)
- 검출 성능: Precision 91.7% / Recall 89.0% / F1 90.3%
- 번호판 인식 성능: 정인식률 17.9% (범용 OCR이라 낮음 — 전용 ANPR 모델의 필요성을 보여줌)

## 검수 중 발견한 데이터 이슈
`issues_log.md` 참고 — 정답 라벨 누락, 원본 이미지 중복(433장 중 실제 고유 302장) 등 발견