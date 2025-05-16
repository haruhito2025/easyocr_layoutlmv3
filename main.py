from pdf2image import convert_from_path
from pathlib import Path
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import easyocr
import numpy as np
import json
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EasyOCRのReader初期化
reader = easyocr.Reader(['ja', 'en'])

# モデル読み込み
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# PDF読み込み
pdf_folder = Path("pdf_folder")
pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    logger.error("PDFフォルダにPDFファイルが見つかりません")
    exit(1)

for pdf_path in pdf_files:
    logger.info(f"処理開始: {pdf_path.name}")
    images = convert_from_path(pdf_path, dpi=300)

    # 出力フォルダ（PDFファイル名ごとに作成）
    output_dir = Path("layoutlm_ocr_output") / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_ocr_boxes_and_text(image: Image.Image):
        """EasyOCRでOCRを実行し、テキストとボックスを取得"""
        try:
            results = reader.readtext(np.array(image))
            words, boxes = [], []
            w, h = image.size
            for bbox, text, conf in results:
                if conf < 0.3 or text.strip() == "":
                    continue
                words.append(text.strip())
                x_min = min(p[0] for p in bbox)
                y_min = min(p[1] for p in bbox)
                x_max = max(p[0] for p in bbox)
                y_max = max(p[1] for p in bbox)
                boxes.append([
                    int(1000 * x_min / w),
                    int(1000 * y_min / h),
                    int(1000 * x_max / w),
                    int(1000 * y_max / h)
                ])
            return words, boxes
        except Exception as e:
            logger.error(f"OCR処理中にエラーが発生: {str(e)}")
            raise

    def save_text_output(words, boxes, output_file):
        """OCR結果をテキストファイルに保存"""
        try:
            sorted_items = sorted(zip(words, boxes), key=lambda x: x[1][1])
            with open(output_file, 'w', encoding='utf-8') as f:
                current_y = -1
                line_buffer = []
                for word, box in sorted_items:
                    if current_y == -1 or abs(box[1] - current_y) > 20:
                        if line_buffer:
                            f.write(' '.join(line_buffer) + '\n')
                            line_buffer = []
                        current_y = box[1]
                    line_buffer.append(word)
                if line_buffer:
                    f.write(' '.join(line_buffer) + '\n')
        except Exception as e:
            logger.error(f"テキスト保存エラー: {str(e)}")
            raise

    # ページごとの処理
    for idx, image in enumerate(images):
        try:
            logger.info(f"ページ {idx+1} の処理開始")
            image_rgb = image.convert("RGB")
            words, boxes = get_ocr_boxes_and_text(image_rgb)

            if not words:
                logger.warning(f"ページ {idx+1}：テキスト検出なし")
                continue

            encoding = processor(image_rgb, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
            for k in encoding:
                encoding[k] = encoding[k].to(device)

            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                predictions = logits.argmax(-1).squeeze().tolist()

            tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())
            results = []
            for word, box, pred, token in zip(words, boxes, predictions[1:1+len(words)], tokens[1:1+len(words)]):
                results.append({
                    "word": word,
                    "bbox": box,
                    "token": token,
                    "label": int(pred)
                })

            # JSON & テキストファイル出力
            with open(output_dir / f"page_{idx+1}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            save_text_output(words, boxes, output_dir / f"page_{idx+1}.txt")

            logger.info(f"ページ {idx+1} の処理完了")

            # メモリ解放
            del encoding, outputs, logits, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"ページ {idx+1} の処理中にエラー: {str(e)}")
            continue
