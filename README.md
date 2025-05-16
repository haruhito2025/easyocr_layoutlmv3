# EasyOCR LayoutLMv3 PDF Processor

このプロジェクトは、PDFファイルからテキストを抽出し、LayoutLMv3モデルを使用して文書の構造を分析するツールです。

## 機能

- PDFファイルからのテキスト抽出（EasyOCR使用）
- 文書構造の分析（LayoutLMv3使用）
- 日本語と英語の両方に対応
- 複数PDFファイルの一括処理

## 必要条件

- Python 3.8以上
- CUDA対応GPU（推奨）

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

1. PDFファイルを`pdf_folder`ディレクトリに配置
2. 以下のコマンドを実行：

```bash
python main.py
```

3. 結果は`layoutlm_ocr_output`ディレクトリに保存されます

## 出力形式

各PDFファイルに対して以下のファイルが生成されます：

- `page_X.json`: テキストとレイアウト情報を含むJSONファイル
- `page_X.txt`: 抽出されたテキストのみを含むテキストファイル

## ライセンス

MIT License 