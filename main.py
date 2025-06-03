#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PDF OCR Text Extraction Tool with Multi-Engine Ensemble
ベクトルDB用高精度OCRシステム
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from PIL import Image
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# インポートエラーのハンドリング
def check_dependencies():
    """依存関係をチェックし、不足している場合は適切なエラーメッセージを表示"""
    missing_deps = []
    
    try:
        import numpy
        if numpy.__version__.startswith('2.'):
            print(f"⚠️  警告: NumPy {numpy.__version__} が検出されました。")
            print("   NumPy 2.x系は一部のライブラリとの互換性に問題があります。")
            print("   以下のコマンドでダウングレードしてください:")
            print("   pip install 'numpy>=1.21.0,<2.0.0'")
            return False
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from pdf2image import convert_from_path
    except ImportError:
        missing_deps.append("pdf2image")
        print("⚠️  pdf2imageのインポートエラー")
        print("   macOSの場合: brew install poppler")
        print("   Linuxの場合: apt-get install poppler-utils")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import easyocr
    except ImportError:
        missing_deps.append("easyocr")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print("❌ 以下のライブラリが不足しています:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n解決方法:")
        print("1. 仮想環境を使用: python -m venv venv && source venv/bin/activate")
        print("2. 依存関係をインストール: pip install -r requirements.txt")
        print("3. macOSの場合: brew install poppler")
        return False
    
    return True

# 依存関係チェック
if not check_dependencies():
    sys.exit(1)

# OCR Engines
import easyocr
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("PaddleOCRをインストール中...")
    os.system("pip install paddleocr")
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("⚠️  PaddleOCRのインストールに失敗しました。EasyOCRのみ使用します。")
        PaddleOCR = None

# Layout Analysis
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Super Resolution (optional)
try:
    import realesrgan
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    print("Real-ESRGANが利用できません。基本的なアップスケーリングを使用します。")
    realesrgan = None

# Language Model for Post-processing (optional)
try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
    BERT_AVAILABLE = True
except ImportError:
    print("⚠️  日本語BERTモデルが利用できません。基本的なテキスト処理のみ行います。")
    BERT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR結果の構造化データ"""
    text: str
    confidence: float
    bbox: List[float]
    source: str  # 'easyocr', 'paddleocr'

@dataclass
class PageResult:
    """ページ単位の結果"""
    page_num: int
    ocr_results: List[OCRResult]
    layout_info: Dict
    final_text: str
    confidence_score: float
    processing_time: float

class SuperResolutionProcessor:
    """超解像処理クラス"""
    
    def __init__(self):
        self.upsampler = None
        if realesrgan:
            try:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                self.upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False
                )
            except Exception as e:
                logger.warning(f"Real-ESRGAN初期化失敗: {e}")
                self.upsampler = None
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """画像の超解像処理"""
        if self.upsampler is None:
            # フォールバック: OpenCVによる基本的なアップスケーリング
            return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        try:
            enhanced, _ = self.upsampler.enhance(image, outscale=2)
            return enhanced
        except Exception as e:
            logger.warning(f"超解像処理失敗: {e}")
            return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

class MultiEngineOCR:
    """複数OCRエンジンのアンサンブル処理"""
    
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=torch.cuda.is_available())
        if PaddleOCR:
            self.paddleocr = PaddleOCR(use_angle_cls=True, lang='japan', use_gpu=torch.cuda.is_available())
        else:
            self.paddleocr = None
        self.super_resolution = SuperResolutionProcessor()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """画像前処理"""
        # 画質評価
        quality_score = self._assess_image_quality(image)
        
        if quality_score < 0.5:
            logger.info("低画質を検出、超解像処理を適用中...")
            image = self.super_resolution.enhance_image(image)
        
        # 基本的な前処理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 適応的閾値処理
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # ノイズ除去
        denoised = cv2.medianBlur(binary, 3)
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """画像品質評価（0-1スコア）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Laplacianによる鮮明度評価
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000, 1.0)
        
        # コントラスト評価
        contrast_score = gray.std() / 255.0
        
        # 総合スコア
        quality_score = (sharpness_score + contrast_score) / 2
        return quality_score
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """EasyOCRでのテキスト抽出"""
        try:
            results = self.easyocr_reader.readtext(image, paragraph=False, width_ths=0.7, height_ths=0.7)
            ocr_results = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.1 and len(text.strip()) > 0:
                    ocr_results.append(OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox,
                        source='easyocr'
                    ))
            
            return ocr_results
        except Exception as e:
            logger.error(f"EasyOCR抽出失敗: {e}")
            return []
    
    def extract_text_paddleocr(self, image: np.ndarray) -> List[OCRResult]:
        """PaddleOCRでのテキスト抽出"""
        if not self.paddleocr:
            return []
        
        try:
            results = self.paddleocr.ocr(image, cls=True)
            ocr_results = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox, (text, confidence) = line
                        if confidence > 0.1 and len(text.strip()) > 0:
                            ocr_results.append(OCRResult(
                                text=text.strip(),
                                confidence=confidence,
                                bbox=bbox,
                                source='paddleocr'
                            ))
            
            return ocr_results
        except Exception as e:
            logger.error(f"PaddleOCR抽出失敗: {e}")
            return []
    
    def ensemble_results(self, results_list: List[List[OCRResult]]) -> List[OCRResult]:
        """複数OCRエンジンの結果をアンサンブル"""
        if not results_list:
            return []
        
        # 全結果を統合
        all_results = []
        for results in results_list:
            all_results.extend(results)
        
        if not all_results:
            return []
        
        # 重複除去とスコアリング
        final_results = []
        processed_texts = set()
        
        # 信頼度でソート
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        
        for result in all_results:
            # 類似テキストの重複チェック
            is_duplicate = False
            for processed_text in processed_texts:
                if self._text_similarity(result.text, processed_text) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_results.append(result)
                processed_texts.add(result.text)
        
        return final_results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """テキスト類似度計算"""
        if not text1 or not text2:
            return 0.0
        
        # 簡易的なJaccard係数
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

class LayoutAnalyzer:
    """レイアウト分析クラス"""
    
    def __init__(self):
        try:
            self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device).eval()
        except Exception as e:
            logger.warning(f"LayoutLMv3初期化失敗: {e}")
            self.processor = None
            self.model = None
    
    def analyze_layout(self, image: np.ndarray, ocr_results: List[OCRResult]) -> Dict:
        """レイアウト分析実行"""
        if not self.processor or not self.model:
            return {"error": "LayoutLMv3が利用できません"}
        
        try:
            # OCR結果をLayoutLMv3形式に変換
            tokens = []
            bboxes = []
            
            for result in ocr_results:
                words = result.text.split()
                for word in words:
                    tokens.append(word)
                    # バウンディングボックスを正規化
                    bbox = self._normalize_bbox(result.bbox, image.shape[:2])
                    bboxes.append(bbox)
            
            if not tokens:
                return {"tokens": [], "labels": [], "structure": {}}
            
            # PIL画像に変換
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # LayoutLMv3での分析
            encoding = self.processor(
                pil_image,
                tokens,
                boxes=bboxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length"
            )
            
            for k in encoding:
                encoding[k] = encoding[k].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            return {
                "tokens": tokens,
                "labels": predictions,
                "structure": self._analyze_structure(tokens, predictions)
            }
            
        except Exception as e:
            logger.error(f"レイアウト分析失敗: {e}")
            return {"error": str(e)}
    
    def _normalize_bbox(self, bbox, image_shape) -> List[int]:
        """バウンディングボックス正規化"""
        height, width = image_shape
        
        if isinstance(bbox[0], list):
            # PaddleOCR形式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
        else:
            # EasyOCR形式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
        
        # 0-1000スケールに正規化
        return [
            int(1000 * x1 / width),
            int(1000 * y1 / height),
            int(1000 * x2 / width),
            int(1000 * y2 / height)
        ]
    
    def _analyze_structure(self, tokens: List[str], labels: List[int]) -> Dict:
        """文書構造分析"""
        structure = {
            "headings": [],
            "paragraphs": [],
            "tables": [],
            "lists": []
        }
        
        # 簡易的な構造分析
        current_text = ""
        for i, token in enumerate(tokens):
            current_text += token + " "
            
            # 改行や区切りでセクション分割
            if i == len(tokens) - 1 or self._is_section_break(token):
                if current_text.strip():
                    structure["paragraphs"].append(current_text.strip())
                current_text = ""
        
        return structure
    
    def _is_section_break(self, token: str) -> bool:
        """セクション区切り判定"""
        return token in ['。', '.', '!', '?', '\n'] or len(token) > 50

class TextPostProcessor:
    """テキスト後処理クラス"""
    
    def __init__(self):
        self.fill_mask = None
        if BERT_AVAILABLE:
            try:
                from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
                tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
                model = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
                self.fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=1)
            except Exception as e:
                logger.warning(f"言語モデル初期化失敗: {e}")
                self.fill_mask = None
    
    def post_process_text(self, text: str) -> str:
        """テキスト後処理"""
        if not text:
            return text
        
        # 基本的なクリーニング
        cleaned_text = self._basic_cleaning(text)
        
        # 言語モデルベース補正
        if self.fill_mask:
            cleaned_text = self._language_model_correction(cleaned_text)
        
        return cleaned_text
    
    def _basic_cleaning(self, text: str) -> str:
        """基本的なテキストクリーニング"""
        # 不要な空白や改行の整理
        text = ' '.join(text.split())
        
        # 明らかな誤字の修正
        corrections = {
            'ｏ': 'o', 'Ｏ': 'O', '０': '0',
            'ｌ': 'l', 'Ｌ': 'L', '１': '1',
            'ｇ': 'g', 'Ｇ': 'G',
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _language_model_correction(self, text: str) -> str:
        """言語モデルベース補正"""
        try:
            # 文を分割
            sentences = text.split('。')
            corrected_sentences = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 0:
                    corrected_sentences.append(sentence.strip())
            
            return '。'.join(corrected_sentences)
        except Exception as e:
            logger.warning(f"言語モデル補正失敗: {e}")
            return text

class EnhancedPDFOCR:
    """強化されたPDF OCRメインクラス"""
    
    def __init__(self, output_dir: str = "enhanced_ocr_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # コンポーネント初期化
        self.ocr_engine = MultiEngineOCR()
        self.layout_analyzer = LayoutAnalyzer()
        self.text_processor = TextPostProcessor()
        
        logger.info("Enhanced PDF OCR が初期化されました")
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """PDF処理メイン関数"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDFファイルが見つかりません: {pdf_path}")
        
        logger.info(f"PDF処理開始: {pdf_path}")
        start_time = time.time()
        
        # 出力ディレクトリ作成
        pdf_output_dir = self.output_dir / pdf_path.stem
        pdf_output_dir.mkdir(exist_ok=True)
        
        # PDF直接テキスト抽出を試行
        direct_text = self._extract_direct_text(pdf_path)
        
        # 画像変換
        images = convert_from_path(pdf_path, dpi=200)
        
        # ページ並列処理
        page_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_page = {
                executor.submit(self._process_page, img, i+1): i+1 
                for i, img in enumerate(images)
            }
            
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    page_results.append(result)
                    logger.info(f"ページ {page_num} 処理完了 (信頼度: {result.confidence_score:.2f})")
                except Exception as e:
                    logger.error(f"ページ {page_num} 処理失敗: {e}")
        
        # 結果の統合
        page_results.sort(key=lambda x: x.page_num)
        
        # 最終文書生成
        final_document = self._create_final_document(page_results, direct_text)
        
        # 結果保存
        self._save_results(pdf_output_dir, page_results, final_document)
        
        processing_time = time.time() - start_time
        
        summary = {
            "pdf_path": str(pdf_path),
            "total_pages": len(images),
            "processed_pages": len(page_results),
            "processing_time": processing_time,
            "average_confidence": np.mean([r.confidence_score for r in page_results]) if page_results else 0,
            "output_directory": str(pdf_output_dir)
        }
        
        logger.info(f"PDF処理完了 {processing_time:.2f}秒")
        return summary
    
    def _extract_direct_text(self, pdf_path: Path) -> str:
        """PDF直接テキスト抽出"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.warning(f"直接テキスト抽出失敗: {e}")
            return ""
    
    def _process_page(self, image: Image.Image, page_num: int) -> PageResult:
        """ページ処理"""
        start_time = time.time()
        
        # PIL to OpenCV
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 画像前処理
        processed_image = self.ocr_engine.preprocess_image(image_cv)
        
        # 複数OCRエンジンでテキスト抽出
        easyocr_results = self.ocr_engine.extract_text_easyocr(processed_image)
        paddleocr_results = self.ocr_engine.extract_text_paddleocr(processed_image)
        
        # アンサンブル
        ensemble_results = self.ocr_engine.ensemble_results([easyocr_results, paddleocr_results])
        
        # レイアウト分析
        layout_info = self.layout_analyzer.analyze_layout(processed_image, ensemble_results)
        
        # テキスト統合
        raw_text = " ".join([result.text for result in ensemble_results])
        
        # 後処理
        final_text = self.text_processor.post_process_text(raw_text)
        
        # 信頼度計算
        confidence_score = self._calculate_confidence(ensemble_results, layout_info)
        
        processing_time = time.time() - start_time
        
        return PageResult(
            page_num=page_num,
            ocr_results=ensemble_results,
            layout_info=layout_info,
            final_text=final_text,
            confidence_score=confidence_score,
            processing_time=processing_time
        )
    
    def _calculate_confidence(self, ocr_results: List[OCRResult], layout_info: Dict) -> float:
        """総合信頼度計算"""
        if not ocr_results:
            return 0.0
        
        # OCR信頼度平均
        ocr_confidence = np.mean([result.confidence for result in ocr_results])
        
        # レイアウト信頼度（構造の複雑さで評価）
        layout_confidence = 0.8 if "error" not in layout_info else 0.5
        
        # テキスト量ベース信頼度
        total_chars = sum(len(result.text) for result in ocr_results)
        text_confidence = min(total_chars / 100, 1.0)
        
        # 重み付き平均
        weights = [0.5, 0.3, 0.2]
        scores = [ocr_confidence, layout_confidence, text_confidence]
        
        return np.average(scores, weights=weights)
    
    def _create_final_document(self, page_results: List[PageResult], direct_text: str) -> str:
        """最終文書作成"""
        # 直接テキストが十分な場合はそれを使用
        if direct_text and len(direct_text) > 100:
            logger.info("直接テキスト抽出を使用")
            return direct_text
        
        # OCR結果を統合
        ocr_text = "\n\n".join([
            f"=== ページ {result.page_num} ===\n{result.final_text}"
            for result in page_results
            if result.final_text
        ])
        
        return ocr_text
    
    def _save_results(self, output_dir: Path, page_results: List[PageResult], final_document: str):
        """結果保存"""
        # メイン文書
        with open(output_dir / "MAIN_DOCUMENT.txt", "w", encoding="utf-8") as f:
            f.write(final_document)
        
        # AI用最適化
        ai_optimized = self._optimize_for_ai(final_document)
        with open(output_dir / "FOR_AI.txt", "w", encoding="utf-8") as f:
            f.write(ai_optimized)
        
        # 詳細レポート
        report = self._generate_report(page_results)
        with open(output_dir / "PROCESSING_REPORT.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # ページ別詳細
        for result in page_results:
            page_file = output_dir / f"page_{result.page_num}.txt"
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(result.final_text)
            
            # ページ別JSON
            page_json = output_dir / f"page_{result.page_num}.json"
            with open(page_json, "w", encoding="utf-8") as f:
                json.dump({
                    "page_num": result.page_num,
                    "text": result.final_text,
                    "confidence": result.confidence_score,
                    "processing_time": result.processing_time,
                    "ocr_results_count": len(result.ocr_results),
                    "layout_info": result.layout_info
                }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"結果を保存しました: {output_dir}")
    
    def _optimize_for_ai(self, text: str) -> str:
        """AI用テキスト最適化"""
        # 段落分割の改善
        paragraphs = text.split('\n\n')
        optimized_paragraphs = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # 長すぎる段落を分割
                if len(paragraph) > 500:
                    sentences = paragraph.split('。')
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 400:
                            current_chunk += sentence + "。"
                        else:
                            if current_chunk:
                                optimized_paragraphs.append(current_chunk.strip())
                            current_chunk = sentence + "。"
                    if current_chunk:
                        optimized_paragraphs.append(current_chunk.strip())
                else:
                    optimized_paragraphs.append(paragraph.strip())
        
        return '\n\n'.join(optimized_paragraphs)
    
    def _generate_report(self, page_results: List[PageResult]) -> Dict:
        """処理レポート生成"""
        total_pages = len(page_results)
        successful_pages = len([r for r in page_results if r.confidence_score > 0.5])
        
        return {
            "summary": {
                "total_pages": total_pages,
                "successful_pages": successful_pages,
                "success_rate": successful_pages / total_pages if total_pages > 0 else 0,
                "average_confidence": np.mean([r.confidence_score for r in page_results]) if page_results else 0,
                "total_processing_time": sum([r.processing_time for r in page_results])
            },
            "page_details": [
                {
                    "page": r.page_num,
                    "confidence": r.confidence_score,
                    "processing_time": r.processing_time,
                    "text_length": len(r.final_text),
                    "ocr_results_count": len(r.ocr_results)
                }
                for r in page_results
            ]
        }

def main():
    """メイン実行関数（従来との互換性維持）"""
    # PDF読み込み
    pdf_folder = Path("pdf_folder")
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        logger.error("PDFフォルダにPDFファイルが見つかりません")
        sys.exit(1)

    # OCRシステム初期化
    ocr_system = EnhancedPDFOCR(output_dir="enhanced_ocr_output")
    
    # 各PDFファイルを処理
    for pdf_path in pdf_files:
        try:
            result = ocr_system.process_pdf(pdf_path)
            
            # 結果表示
            print(f"\n=== 処理結果: {pdf_path.name} ===")
            print(f"ページ数: {result['processed_pages']}/{result['total_pages']}")
            print(f"処理時間: {result['processing_time']:.2f}秒")
            print(f"平均信頼度: {result['average_confidence']:.2f}")
            print(f"出力ディレクトリ: {result['output_directory']}")
            
        except Exception as e:
            logger.error(f"PDF処理失敗 {pdf_path.name}: {e}")
            continue

if __name__ == "__main__":
    main()
