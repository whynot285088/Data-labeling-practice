"""
verify_real.py
---------------
labeling_worksheet.csv (여러분이 plate_crops 이미지를 보고 actual_plate 칸을 채운 파일)를
읽어서 최종 집계를 낸다.

사용법:
    python3 verify_real.py --input labeling_worksheet.csv --output real_summary.json
"""
import argparse
import json
import pandas as pd


def char_error_rate(actual: str, recognized: str):
    if actual == "" or pd.isna(actual):
        return None
    if recognized == "" or pd.isna(recognized):
        return 1.0
    n, m = len(actual), len(recognized)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if actual[i - 1] == recognized[j - 1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m] / n


def normalize_plate(text: str) -> str:
    """
    비교용 정규화: 공백/하이픈/점 등 장식용 구분기호를 제거하고 순수 영숫자만 남긴다.
    (실제 번호판 인식 업계에서도 구분기호 차이만으로 오인식 처리하지 않기 위해 흔히 쓰는 방식)
    """
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def judge_plate(actual, recognized):
    if pd.isna(actual) or actual == "":
        return "미검증"  # 아직 사람이 actual_plate를 안 채운 행
    if str(actual).strip().upper() == "UNREADABLE":
        return "확인불가"  # 사람이 봤지만 판독 자체가 불가능했던 행 (미검증과는 다름)
    if pd.isna(recognized) or recognized == "":
        return "미인식"
    return "정인식" if normalize_plate(actual) == normalize_plate(recognized) else "오인식"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="real_summary.json")
    args = parser.parse_args()

    df = pd.read_excel(args.input, dtype=str, keep_default_na=False) if args.input.lower().endswith((".xlsx", ".xls")) \
        else pd.read_csv(args.input, dtype=str, keep_default_na=False)

    # --- 1) 검출(Detection) 성능: detection_status 컬럼 그대로 집계 ---
    tp = (df["detection_status"] == "TP").sum()
    fp = (df["detection_status"] == "FP").sum()
    fn = (df["detection_status"] == "FN").sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None

    print(f"[검출 성능] TP={tp} FP={fp} FN={fn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}\n")

    # --- 2) 번호판 인식 성능: actual_plate가 채워진 행만 대상 ---
    df["plate_judgement"] = df.apply(lambda r: judge_plate(r["actual_plate"], r["ai_recognized_plate"]), axis=1)
    df["cer"] = df.apply(
        lambda r: char_error_rate(normalize_plate(r["actual_plate"]), normalize_plate(r["ai_recognized_plate"]))
        if r["plate_judgement"] in ("정인식", "오인식", "미인식") else None,
        axis=1,
    )

    labeled = df[df["plate_judgement"] != "미검증"]
    n_labeled = len(labeled)
    n_total_plates = ((df["detection_status"] == "TP") | (df["detection_status"] == "FN")).sum()

    print(f"[번호판 라벨링 진행률] {n_labeled} / {n_total_plates} 건 완료 (확인불가 포함)")

    # 정인식률/CER은 '확인불가'를 뺀, 실제 판단 가능했던 행만 대상으로 계산
    judged = labeled[labeled["plate_judgement"] != "확인불가"]
    n_unreadable = (labeled["plate_judgement"] == "확인불가").sum()

    if len(judged) > 0:
        counts = judged["plate_judgement"].value_counts().to_dict()
        correct_rate = counts.get("정인식", 0) / len(judged)
        mean_cer = judged["cer"].dropna().astype(float).mean()
        print(f"정인식={counts.get('정인식',0)}  오인식={counts.get('오인식',0)}  미인식={counts.get('미인식',0)}  확인불가={n_unreadable}(집계 제외)")
        print(f"정인식률={correct_rate*100:.1f}%   평균 CER={mean_cer:.3f}")
    else:
        counts, correct_rate, mean_cer = {}, None, None
        print(f"확인불가={n_unreadable}건 있으나, 판단 가능한 행이 아직 없습니다.")

    summary = {
        "detection": {"TP": int(tp), "FP": int(fp), "FN": int(fn),
                       "precision": precision, "recall": recall, "f1": f1},
        "plate": {"labeled": int(n_labeled), "total": int(n_total_plates),
                  "unreadable": int(n_unreadable), "judged": int(len(judged)),
                  "counts": counts, "correct_rate": correct_rate, "mean_cer": mean_cer},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {args.output}")


if __name__ == "__main__":
    main()
