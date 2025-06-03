# Enhanced PDF OCR with Layout Analysis

高精度なPDF OCRシステムで、複数のOCRエンジンとレイアウト分析を組み合わせて、より正確なテキスト抽出を実現します。

## 🚀 利用可能なバージョン

### 1. 完全版 (`main.py`)
- **EasyOCR** + **PaddleOCR** のアンサンブル
- **Real-ESRGAN** 超解像処理
- 日本語BERT後処理
- 並列処理

### 2. 簡易版 (`main_simple.py`) - **推奨**
- **EasyOCR** のみ使用
- 基本的な画像前処理
- LayoutLMv3 構造分析
- 軽量で安定した動作

## 🚀 主要機能

### 複数OCRエンジンアンサンブル（完全版）
- **EasyOCR** + **PaddleOCR** の組み合わせによる高精度テキスト抽出
- 重複除去と信頼度ベースの結果統合
- テキスト類似度計算による最適化

### 高度な画像処理
- **Real-ESRGAN** による超解像処理（低画質画像の品質向上）
- 画質自動評価と適応的前処理
- OpenCVベースのノイズ除去・二値化処理

### LayoutLMv3による文書構造分析
- Microsoft LayoutLMv3を使用した高精度レイアウト分析
- 見出し、段落、表、リストの自動識別
- 文書構造を考慮したテキスト配置

### テキスト後処理
- 日本語BERTモデルによる誤字脱字補正（完全版）
- 全角・半角文字の正規化
- AI用最適化テキストの自動生成

### 並列処理とパフォーマンス
- ThreadPoolExecutorによるページ並列処理（完全版）
- CUDA GPU サポート（自動検出）
- メモリ効率的な処理

## 📋 必要条件

- Python 3.8以上
- CUDA対応GPU（推奨）
- macOSの場合: `brew install poppler`
- Linuxの場合: `apt-get install poppler-utils`

## 🛠 セットアップ

### 1. 段階的インストール（推奨）

```bash
# 1. システム依存関係
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# 2. 仮想環境作成
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# または
venv\Scripts\activate     # Windows

# 3. 段階的インストールスクリプト実行
python install_dependencies.py
```

### 2. 手動インストール

#### 基本パッケージ
```bash
pip install --upgrade pip
pip install wheel
pip install 'numpy>=1.21.0,<2.0.0'
pip install Pillow>=9.0.0
pip install opencv-python-headless>=4.7.0
```

#### コア依存関係
```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install pdf2image>=1.16.3
pip install easyocr>=1.7.0
pip install PyMuPDF>=1.23.0
```

#### オプション依存関係（完全版用）
```bash
pip install scipy  # 最新版
pip install paddleocr>=2.7.0  # 時間がかかる場合があります
```

### 3. インストール確認

```bash
python -c "import fitz, easyocr, torch, transformers; print('✅ 必須ライブラリ正常')"
```

## 📖 使用方法

### クイックスタート（簡易版推奨）

1. PDFファイルを `pdf_folder/` ディレクトリに配置
2. 簡易版スクリプトを実行:
```bash
python main_simple.py
```

### 完全版の実行

```bash
# 全ての依存関係がインストール済みの場合
python main.py
```

### 出力ファイル

結果は以下に保存されます：

#### 簡易版出力 (`simple_ocr_output/[PDF名]/`)
- `MAIN_DOCUMENT.txt` - メイン抽出テキスト
- `PROCESSING_REPORT.json` - 処理レポート
- `page_X.txt` - ページ別テキスト

#### 完全版出力 (`enhanced_ocr_output/[PDF名]/`)
- `MAIN_DOCUMENT.txt` - メイン抽出テキスト
- `FOR_AI.txt` - AI用最適化テキスト
- `PROCESSING_REPORT.json` - 詳細処理レポート
- `page_X.txt` - ページ別テキスト
- `page_X.json` - ページ別詳細データ

### プログラムでの使用

```python
# 簡易版
from main_simple import SimplePDFOCR
ocr_system = SimplePDFOCR(output_dir="custom_output")
result = ocr_system.process_pdf("document.pdf")

# 完全版
from main import EnhancedPDFOCR
ocr_system = EnhancedPDFOCR(output_dir="custom_output")
result = ocr_system.process_pdf("document.pdf")
```

## 📊 処理結果の詳細

### 信頼度スコア
- OCR信頼度（重み: 50%）
- レイアウト信頼度（重み: 30%）
- テキスト量信頼度（重み: 20%）

### パフォーマンス指標
- ページ処理時間
- 総合処理時間
- 成功率（信頼度0.5以上）
- 抽出文字数

## 🐛 トラブルシューティング

### よくあるエラーと解決方法

#### Python バージョン互換性
```
Your Python version is too new. SciPy 1.9 supports Python 3.8-3.11
```
**解決方法**: 
```bash
pip install scipy  # 最新版を使用
```

#### PyMuPDF (fitz) インポートエラー
```bash
pip install PyMuPDF  # 正しいパッケージ名
```

#### PaddleOCR インストール時間が長い
- 簡易版 (`main_simple.py`) の使用を推奨
- バックグラウンドでの完了を待つ

#### CUDA/GPU関連警告
- GPU利用不可の場合、自動的にCPU実行に切り替わります
- 処理時間は長くなりますが、品質に影響はありません

#### メモリ不足
```python
# DPI を下げて軽量化
convert_from_path(pdf_path, dpi=150)  # デフォルト: 200
```

### 段階的デバッグ

1. **依存関係確認**
```bash
python install_dependencies.py
```

2. **簡易版テスト**
```bash
python main_simple.py
```

3. **完全版テスト**（依存関係完備後）
```bash
python main.py
```

## 🔧 開発・カスタマイズ

### 設定変更

#### 信頼度閾値調整
```python
# main_simple.py 内
if confidence > 0.1:  # 0.1 → 0.3 に変更で厳格化
```

#### DPI設定
```python
images = convert_from_path(pdf_path, dpi=300)  # 高品質
images = convert_from_path(pdf_path, dpi=150)  # 軽量
```

### 新機能追加

#### カスタム前処理
```python
class SimpleOCR:
    def preprocess_image(self, image):
        # カスタム処理を追加
        return processed_image
```

## 📄 ファイル構成

```
easyocr_layoutlmv3/
├── main.py                    # 完全版OCRシステム
├── main_simple.py             # 簡易版OCRシステム（推奨）
├── install_dependencies.py    # 段階的インストーラー
├── requirements.txt           # 依存関係リスト
├── README.md                  # このファイル
├── setup.sh                   # 自動セットアップ（macOS）
├── pdf_folder/                # PDF入力フォルダ
├── simple_ocr_output/         # 簡易版出力
└── enhanced_ocr_output/       # 完全版出力
```

## 📄 ライセンス

MIT License

## 🤝 貢献

Issue や Pull Request をお待ちしています。

## 📞 サポート

問題が発生した場合は、以下の情報を含めてIssueを作成してください：
- OS とバージョン
- Python バージョン
- 使用したスクリプト（`main.py` or `main_simple.py`）
- エラーメッセージ全文
- 処理しようとしたPDFの特徴（言語、レイアウトなど）

## 特徴

- EasyOCRとPaddleOCRのアンサンブル処理
- LayoutLMv3による文書構造分析
- 超解像処理による低画質文書の改善
- 日本語BERTモデルによる後処理
- マルチスレッド処理による高速化

## 必要条件

- Python 3.8以上
- CUDA対応GPU（推奨）
- macOSの場合: `brew install poppler`
- Linuxの場合: `apt-get install poppler-utils`

## インストール

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/enhanced-pdf-ocr.git
cd enhanced-pdf-ocr
```

2. 仮想環境を作成して有効化:
```bash
python -m venv venv
source venv/bin/activate  # Linuxの場合
# または
.\venv\Scripts\activate  # Windowsの場合
```

3. 依存関係をインストール:
```bash
pip install -r requirements.txt
```

## 使用方法

1. PDFファイルを`pdf_folder`ディレクトリに配置します。

2. スクリプトを実行:
```bash
python main.py
```

3. 処理結果は`enhanced_ocr_output`ディレクトリに保存されます。

## 出力ファイル

- `MAIN_DOCUMENT.txt`: 抽出されたテキスト全体
- `FOR_AI.txt`: AI処理用に最適化されたテキスト
- `PROCESSING_REPORT.json`: 処理の詳細レポート
- `page_*.txt`: ページごとの抽出テキスト
- `page_*.json`: ページごとの詳細情報

## ライセンス

MIT License

## 注意事項

- このプロジェクトは研究・開発目的で作成されています
- 商用利用の場合は、各ライブラリのライセンスを確認してください
- 処理速度は使用するハードウェアに依存します 