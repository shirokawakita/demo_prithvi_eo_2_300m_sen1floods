# 🌧️ Prithvi-EO-2.0 Sen1Floods11 洪水検出システム

IBMとNASAが開発した第2世代地球観測基盤モデル「Prithvi-EO-2.0」を使用したSentinel-2画像からの洪水検出システムです。

## 📋 概要

このプロジェクトは、Sentinel-2衛星画像を使用してリアルタイムで洪水エリアを検出するWebアプリケーションです。Prithvi-EO-2.0-300M-TL-Sen1Floods11モデルを使用して、高精度な洪水セマンティックセグメンテーションを実行します。

### 主な機能

- 🖼️ **Sentinel-2画像のアップロード**: TIFFファイルのドラッグ&ドロップ対応
- 🔄 **自動画像前処理**: サイズ調整とデータ型変換
- 🧠 **AI洪水検出**: Prithvi-EO-2.0モデルによる高精度予測
- 📊 **3種類の結果表示**: 入力画像、予測結果、オーバーレイ表示
- 💾 **結果ダウンロード**: PNG形式での結果保存
- 🌍 **サンプルデータ**: India、Spain、USAのサンプル画像

## 🛠️ セットアップ

### 1. 環境構築

```bash
# Anaconda環境の作成
conda create -n prithvi python=3.10
conda activate prithvi

# 必要なパッケージのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install streamlit rasterio pillow pyyaml numpy scikit-image
pip install terratorch  # Prithvi-EO-2.0モデル用
```

### 2. モデルとデータのダウンロード

#### 方法1: 直接ダウンロード（推奨）

[Hugging Face Model Repository](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/tree/main)から必要なファイルをダウンロードしてください：

**必須ファイル:**
- `Prithvi-EO-V2-300M-TL-Sen1Floods11.pt` (1.28 GB) - 学習済みモデル
- `config.yaml` (4.33 kB) - モデル設定ファイル
- `inference.py` (10.3 kB) - 推論スクリプト（参考用）

**ダウンロード手順:**
1. [モデルページ](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/tree/main)にアクセス
2. 各ファイルをクリックして「Download」ボタンでダウンロード
3. プロジェクトフォルダに配置

#### 方法2: wgetコマンド

```bash
# モデルファイルのダウンロード
wget https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt

# 設定ファイルのダウンロード
wget https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/config.yaml
```

#### 方法3: Hugging Face Hub（オプション）

```bash
# Hugging Face Hubのインストール
pip install huggingface_hub

# Pythonでダウンロード
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11', filename='Prithvi-EO-V2-300M-TL-Sen1Floods11.pt')
hf_hub_download(repo_id='ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11', filename='config.yaml')
"
```

#### サンプルデータ（オプション）

```bash
# サンプルデータフォルダの作成
mkdir data
# サンプル画像をdataフォルダに配置（別途入手）
```

### 3. ファイル構成

```
prithivi-eo-2-300m-tl-sen1floods11/
├── main.py                                    # Streamlit Webアプリケーション
├── inference.py                               # 推論スクリプト
├── config.yaml                                # モデル設定ファイル
├── Prithvi-EO-V2-300M-TL-Sen1Floods11.pt    # 学習済みモデル
├── data/                                      # サンプルデータフォルダ
│   ├── India_900498_S2Hand.tif
│   ├── Spain_7370579_S2Hand.tif
│   └── USA_430764_S2Hand.tif
└── README.md
```

## 🚀 使用方法

### Webアプリケーション（推奨）

```bash
conda activate prithvi
streamlit run main.py
```

ブラウザで `http://localhost:8501` にアクセスして、直感的なWebインターフェースを使用できます。

#### Webアプリの機能：

1. **ファイルアップロード**: Sentinel-2 TIFFファイルをドラッグ&ドロップ
2. **自動前処理**: 画像サイズとデータ型の自動調整
3. **推論実行**: ワンクリックで洪水検出を実行
4. **結果表示**: 3つの画像を並べて表示
   - **Input image**: 入力RGB画像（バンド3,2,1）
   - **Prediction**: 予測結果（白=洪水、黒=非洪水）
   - **Overlay**: 入力画像に洪水エリアを赤色で重畳
5. **ダウンロード**: 各結果をPNG形式で保存
6. **サンプル実行**: プリセットされたサンプル画像で即座にテスト

### コマンドライン実行

```bash
conda activate prithvi
python inference.py \
  --data_file data/India_900498_S2Hand.tif \
  --config config.yaml \
  --checkpoint Prithvi-EO-V2-300M-TL-Sen1Floods11.pt \
  --output_dir output \
  --rgb_outputs
```

## 📊 対応画像形式

### 入力要件

- **ファイル形式**: GeoTIFF (.tif, .tiff)
- **バンド数**: 13バンド（Sentinel-2 L1C）または6バンド（Prithvi対応バンド）
- **対応バンド**: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2
- **画像サイズ**: 任意（自動的に512×512にリサイズ）
- **データ型**: uint16またはint16（自動変換）

### 自動前処理機能

システムは以下の前処理を自動実行します：

1. **サイズ正規化**: 任意サイズ → 512×512ピクセル
2. **データ型変換**: uint16 → int16
3. **値域正規化**: 訓練データ範囲（1000-3000）に調整
4. **バンド選択**: 13バンドから6バンドを自動選択

## 🔧 技術仕様

### モデル詳細

- **ベースモデル**: Prithvi-EO-2.0-300M
- **ファインチューニング**: Sen1Floods11データセット
- **アーキテクチャ**: Vision Transformer + UperNet Decoder
- **入力サイズ**: 512×512ピクセル
- **出力**: 2クラス（洪水/非洪水）セマンティックセグメンテーション
- **データセット**: 446個のラベル付き512×512チップ（14バイオーム、357エコリージョン、6大陸、11洪水イベント）
- **対象バンド**: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2（6バンド）
- **クラス定義**: 
  - クラス0: 非水域
  - クラス1: 水域/洪水
  - クラス-1: データなし/雲

### 性能指標

テストデータセットでの性能（100エポック学習後）：

| **クラス** | **IoU** | **Acc** |
|-----------|---------|---------|
| 非水域     | 96.90%  | 98.11%  |
| 水域/洪水  | 80.46%  | 90.54%  |

| **aAcc** | **mIoU** | **mAcc** |
|----------|----------|----------|
| 97.25%   | 88.68%   | 94.37%   |

### 使用技術

- **深層学習フレームワーク**: PyTorch
- **Webフレームワーク**: Streamlit
- **画像処理**: rasterio, scikit-image, PIL
- **地理空間処理**: rasterio, GDAL

## 📈 性能と制限

### 推奨仕様

- **画像サイズ**: 500-1000ピクセル（高速処理のため）
- **メモリ**: 8GB以上のRAM
- **処理時間**: 512×512画像で約30-60秒（CPU）

### 制限事項

- 256×256ピクセルより大きな画像では、パッチ間でアーティファクトが発生する可能性
- CPUでの実行のため、大きな画像の処理には時間がかかる場合がある

## 🌍 サンプルデータ

プロジェクトには以下のサンプルデータが含まれています：

1. **India** (data/India_900498_S2Hand.tif): インドの洪水エリア
2. **Spain** (data/Spain_7370579_S2Hand.tif): スペインの洪水エリア  
3. **USA** (data/USA_430764_S2Hand.tif): アメリカの洪水エリア

## 🐛 トラブルシューティング

### よくある問題

1. **conda activateエラー**:
   ```bash
   conda init bash
   # ターミナルを再起動後、再度実行
   ```

2. **メモリ不足エラー**:
   - より小さな画像を使用
   - 他のアプリケーションを終了

3. **モジュールインポートエラー**:
   ```bash
   pip install --upgrade terratorch
   ```

### ログの確認

Streamlitアプリケーションでは、処理状況とエラーメッセージがリアルタイムで表示されます。

## 📚 参考文献

- [Prithvi-EO-2.0 Model](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)
- [Prithvi-EO-1.0 Model](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11)
- [Sen1Floods11 Dataset](https://github.com/cloudtostreet/Sen1Floods11)
- [Terratorch Framework](https://github.com/IBM/terratorch)

## 📖 Citation

このモデルが研究に役立った場合は、以下の論文を引用してください：

### Prithvi-EO-2.0

```bibtex
@article{Prithvi-EO-V2-preprint,    
    author          = {Szwarcman, Daniela and Roy, Sujit and Fraccaro, Paolo and Gíslason, Þorsteinn Elí and Blumenstiel, Benedikt and Ghosal, Rinki and de Oliveira, Pedro Henrique and de Sousa Almeida, João Lucas and Sedona, Rocco and Kang, Yanghui and Chakraborty, Srija and Wang, Sizhe and Kumar, Ankur and Truong, Myscon and Godwin, Denys and Lee, Hyunho and Hsu, Chia-Yu and Akbari Asanjan, Ata and Mujeci, Besart and Keenan, Trevor and Arévolo, Paulo and Li, Wenwen and Alemohammad, Hamed and Olofsson, Pontus and Hain, Christopher and Kennedy, Robert and Zadrozny, Bianca and Cavallaro, Gabriele and Watson, Campbell and Maskey, Manil and Ramachandran, Rahul and Bernabe Moreno, Juan},
    title           = {{Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications}},
    journal         = {arXiv preprint arXiv:2412.02732},
    year            = {2024}
}
```

### Prithvi-EO-1.0 洪水マッピング

```bibtex
@misc{Prithvi-100M-flood-mapping,
    author          = {Jakubik, Johannes and Fraccaro, Paolo and Oliveira Borges, Dario and Muszynski, Michal and Weldemariam, Kommy and Zadrozny, Bianca and Ganti, Raghu and Mukkavilli, Karthik},
    month           = aug,
    doi             = { 10.57967/hf/0973 },
    title           = {{Prithvi 100M flood mapping}},
    repository-code = {https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-sen1floods11},
    year            = {2023}
}
```

## 📄 ライセンス

このプロジェクトは、元のPrithvi-EO-2.0モデルのライセンスに従います。

## 🤝 貢献

バグ報告や機能改善の提案は、GitHubのIssueまでお願いします。

---

**開発者**: IBM & NASA Geospatial Team  
**最終更新**: 2025年1月
