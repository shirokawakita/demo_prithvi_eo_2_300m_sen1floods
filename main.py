import streamlit as st
import numpy as np
import torch
import yaml
import os
import tempfile
from PIL import Image
import rasterio
from datetime import datetime
from skimage.transform import resize

# Import functions from inference.py
from inference import (
    SemanticSegmentationTask,
    Sen1Floods11NonGeoDataModule,
    load_example,
    run_model,
    save_prediction
)

def preprocess_image(file_path, target_size=(512, 512), target_dtype=np.int16):
    """
    Preprocess image to match the expected format:
    - Resize to target size
    - Convert data type
    - Normalize data range
    """
    st.info(f"画像を前処理中... (目標サイズ: {target_size}, データ型: {target_dtype})")
    
    with rasterio.open(file_path) as src:
        # Read all bands
        img = src.read()  # Shape: (bands, height, width)
        profile = src.profile.copy()
        
        st.info(f"元画像: バンド数={img.shape[0]}, サイズ={img.shape[1]}x{img.shape[2]}, データ型={img.dtype}")
        
        # Resize each band if necessary
        if img.shape[1:] != target_size:
            st.info(f"画像をリサイズ中: {img.shape[1]}x{img.shape[2]} → {target_size[0]}x{target_size[1]}")
            resized_bands = []
            for i in range(img.shape[0]):
                # Resize each band individually
                resized_band = resize(
                    img[i], 
                    target_size, 
                    preserve_range=True,
                    anti_aliasing=True
                ).astype(img.dtype)
                resized_bands.append(resized_band)
            img = np.stack(resized_bands, axis=0)
        
        # Convert data type if necessary
        if img.dtype != target_dtype:
            st.info(f"データ型を変換中: {img.dtype} → {target_dtype}")
            
            # Normalize to target data type range
            if img.dtype == np.uint16 and target_dtype == np.int16:
                # Convert uint16 to int16 range
                # uint16: 0-65535 → int16: -32768 to 32767
                # But we'll map to positive range similar to training data (1000-3000)
                img_min, img_max = img.min(), img.max()
                # Normalize to 0-1 range
                img_normalized = (img.astype(np.float32) - img_min) / (img_max - img_min)
                # Scale to target range (similar to training data: 1000-3000)
                img = (img_normalized * 2000 + 1000).astype(target_dtype)
            else:
                # General conversion
                img = img.astype(target_dtype)
        
        st.success(f"前処理完了: バンド数={img.shape[0]}, サイズ={img.shape[1]}x{img.shape[2]}, データ型={img.dtype}")
        
        # Save preprocessed image to temporary file
        output_path = file_path.replace('.tif', '_preprocessed.tif')
        
        # Update profile for the new image
        profile.update({
            'height': target_size[0],
            'width': target_size[1],
            'dtype': target_dtype,
            'count': img.shape[0]
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(img)
        
        return output_path

def load_model_and_config():
    """Load model and configuration"""
    if 'model' not in st.session_state:
        with st.spinner('モデルを読み込み中...'):
            # Load config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Create model
            model = SemanticSegmentationTask(
                model_args={
                    "backbone_pretrained": True,
                    "backbone": "prithvi_eo_v2_300_tl",
                    "decoder": "UperNetDecoder",
                    "decoder_channels": 256,
                    "decoder_scale_modules": True,
                    "num_classes": 2,
                    "rescale": True,
                    "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
                    "head_dropout": 0.1,
                    "necks": [
                        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                        {"name": "ReshapeTokensToImage"},
                    ],
                },
                model_factory="EncoderDecoderFactory",
                loss="ce",
                ignore_index=-1,
                lr=0.001,
                freeze_backbone=False,
                freeze_decoder=False,
                plot_on_val=10,
            )
            
            # Load checkpoint
            checkpoint_dict = torch.load('Prithvi-EO-V2-300M-TL-Sen1Floods11.pt', map_location=torch.device('cpu'))["state_dict"]
            new_state_dict = {}
            for k, v in checkpoint_dict.items():
                if k.startswith("model.encoder._timm_module."):
                    new_key = k.replace("model.encoder._timm_module.", "model.encoder.")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            model.eval()
            
            # Load datamodule
            datamodule = Sen1Floods11NonGeoDataModule(config)
            
            st.session_state.model = model
            st.session_state.datamodule = datamodule
            st.session_state.config = config

def process_image(uploaded_file):
    """Process uploaded image and run inference"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Preprocess image to match training data format
        preprocessed_path = preprocess_image(tmp_path, target_size=(512, 512), target_dtype=np.int16)
        
        # Load data
        imgs, temporal_coords, location_coords = load_example(
            preprocessed_path,
            input_indices=[1, 2, 3, 8, 11, 12],  # Sentinel-2の6バンド
        )
        
        # Run model
        pred = run_model(
            imgs,
            temporal_coords,
            location_coords,
            st.session_state.model,
            st.session_state.datamodule,
        )
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        output_file = os.path.join(output_dir, 'prediction.tif')
        
        # Save predictions
        save_prediction(pred, output_file, rgb_outputs=True, input_image=imgs)
        
        # Load generated images
        input_rgb_path = output_file.replace('.tif', '_input_rgb.png')
        prediction_path = output_file.replace('.tif', '_prediction.png')
        overlay_path = output_file.replace('.tif', '_rgb.png')
        
        input_rgb = Image.open(input_rgb_path) if os.path.exists(input_rgb_path) else None
        prediction = Image.open(prediction_path) if os.path.exists(prediction_path) else None
        overlay = Image.open(overlay_path) if os.path.exists(overlay_path) else None
        
        return input_rgb, prediction, overlay
        
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if 'preprocessed_path' in locals() and os.path.exists(preprocessed_path):
            os.unlink(preprocessed_path)

def process_sample_image(sample_file):
    """Process sample image with preprocessing"""
    try:
        # Check if preprocessing is needed
        with rasterio.open(sample_file) as src:
            needs_preprocessing = (
                src.shape != (512, 512) or 
                src.dtypes[0] != np.int16
            )
        
        if needs_preprocessing:
            st.info(f"サンプル画像を前処理中...")
            preprocessed_path = preprocess_image(sample_file, target_size=(512, 512), target_dtype=np.int16)
            processing_path = preprocessed_path
        else:
            processing_path = sample_file
        
        # Load data
        imgs, temporal_coords, location_coords = load_example(
            processing_path,
            input_indices=[1, 2, 3, 8, 11, 12],
        )
        
        # Run model
        pred = run_model(
            imgs,
            temporal_coords,
            location_coords,
            st.session_state.model,
            st.session_state.datamodule,
        )
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        country = os.path.basename(sample_file).split('_')[0]
        output_file = os.path.join(output_dir, f'{country}_prediction.tif')
        
        # Save predictions
        save_prediction(pred, output_file, rgb_outputs=True, input_image=imgs)
        
        # Load generated images
        input_rgb_path = output_file.replace('.tif', '_input_rgb.png')
        prediction_path = output_file.replace('.tif', '_prediction.png')
        overlay_path = output_file.replace('.tif', '_rgb.png')
        
        input_rgb = Image.open(input_rgb_path)
        prediction = Image.open(prediction_path)
        overlay = Image.open(overlay_path)
        
        return input_rgb, prediction, overlay, country
        
    finally:
        # Clean up preprocessed file if created
        if needs_preprocessing and 'preprocessed_path' in locals() and os.path.exists(preprocessed_path):
            os.unlink(preprocessed_path)

def main():
    st.set_page_config(
        page_title="Prithvi-EO-2.0 Sen1Floods11 Demo",
        page_icon="🌧️",
        layout="wide"
    )
    
    st.title("🌧️ Prithvi-EO-2.0 Sen1Floods11 Demo")
    
    st.markdown("""
    **Prithvi-EO-2.0**は、IBMとNASAチームによって開発された第2世代EO基盤モデルです。
    このデモでは、Sentinel-2画像を使用した洪水検出のためにファインチューニングされたPrithvi-EO-2.0-300M-TLモデルを紹介します。
    
    **使用方法:**
    - Sentinel-2 L1C画像（13バンドまたは6つのPrithviバンド：Blue, Green, Red, Narrow NIR, SWIR, SWIR 2）をアップロードしてください
    - より高速な処理のために、500〜1000ピクセルの画像を推奨します
    - 256x256より大きな画像は、パッチ間でアーティファクトが発生する可能性があるスライディングウィンドウアプローチを使用して処理されます
    """)
    
    # Load model
    load_model_and_config()
    
    # File upload
    st.subheader("📁 ファイルアップロード")
    uploaded_file = st.file_uploader(
        "Sentinel-2 TIFFファイルをアップロードしてください",
        type=['tif', 'tiff'],
        help="Sentinel-2 L1C画像（.tif形式）をアップロードしてください"
    )
    
    if uploaded_file is not None:
        st.success(f"ファイル '{uploaded_file.name}' がアップロードされました")
        
        # Process button
        if st.button("🚀 推論実行", type="primary"):
            with st.spinner('画像を処理中...'):
                try:
                    input_rgb, prediction, overlay = process_image(uploaded_file)
                    
                    # Display results
                    st.subheader("📊 結果")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Input image**")
                        if input_rgb:
                            st.image(input_rgb, use_column_width=True)
                        else:
                            st.error("入力画像の生成に失敗しました")
                    
                    with col2:
                        st.markdown("**Prediction***")
                        if prediction:
                            st.image(prediction, use_column_width=True)
                        else:
                            st.error("予測画像の生成に失敗しました")
                    
                    with col3:
                        st.markdown("**Overlay**")
                        if overlay:
                            st.image(overlay, use_column_width=True)
                        else:
                            st.error("オーバーレイ画像の生成に失敗しました")
                    
                    st.markdown("*White = flood; Black = no flood")
                    
                    # Download buttons
                    st.subheader("💾 ダウンロード")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if input_rgb:
                            buf = tempfile.NamedTemporaryFile(suffix='.png')
                            input_rgb.save(buf.name)
                            with open(buf.name, 'rb') as f:
                                st.download_button(
                                    label="入力画像をダウンロード",
                                    data=f.read(),
                                    file_name=f"{uploaded_file.name.split('.')[0]}_input_rgb.png",
                                    mime="image/png"
                                )
                    
                    with col2:
                        if prediction:
                            buf = tempfile.NamedTemporaryFile(suffix='.png')
                            prediction.save(buf.name)
                            with open(buf.name, 'rb') as f:
                                st.download_button(
                                    label="予測結果をダウンロード",
                                    data=f.read(),
                                    file_name=f"{uploaded_file.name.split('.')[0]}_prediction.png",
                                    mime="image/png"
                                )
                    
                    with col3:
                        if overlay:
                            buf = tempfile.NamedTemporaryFile(suffix='.png')
                            overlay.save(buf.name)
                            with open(buf.name, 'rb') as f:
                                st.download_button(
                                    label="オーバーレイをダウンロード",
                                    data=f.read(),
                                    file_name=f"{uploaded_file.name.split('.')[0]}_overlay.png",
                                    mime="image/png"
                                )
                    
                except Exception as e:
                    st.error(f"処理中にエラーが発生しました: {str(e)}")
                    st.exception(e)
    
    # Sample images section
    st.subheader("📸 サンプル画像")
    st.markdown("以下のサンプル画像を使用してデモを試すことができます:")
    
    sample_files = [
        "data/India_900498_S2Hand.tif",
        "data/Spain_7370579_S2Hand.tif", 
        "data/USA_430764_S2Hand.tif"
    ]
    
    # Add data2 folder if it exists
    data2_file = "data2/Sentinel2_L1C_20250406_1749828526135.tif"
    if os.path.exists(data2_file):
        sample_files.append(data2_file)
    
    cols = st.columns(len(sample_files))
    for i, sample_file in enumerate(sample_files):
        if os.path.exists(sample_file):
            with cols[i]:
                if "data2" in sample_file:
                    country = "Japan"
                else:
                    country = os.path.basename(sample_file).split('_')[0]
                st.markdown(f"**{country}**")
                
                if st.button(f"{country}を使用", key=f"sample_{i}"):
                    with st.spinner(f'{country}の画像を処理中...'):
                        try:
                            # Process sample image with preprocessing
                            input_rgb, prediction, overlay, processed_country = process_sample_image(sample_file)
                            
                            # Display results
                            st.subheader(f"📊 {country}の結果")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Input image**")
                                st.image(input_rgb, use_column_width=True)
                            
                            with col2:
                                st.markdown("**Prediction***")
                                st.image(prediction, use_column_width=True)
                            
                            with col3:
                                st.markdown("**Overlay**")
                                st.image(overlay, use_column_width=True)
                            
                            st.markdown("*White = flood; Black = no flood")
                            
                        except Exception as e:
                            st.error(f"処理中にエラーが発生しました: {str(e)}")
                            st.exception(e)

if __name__ == "__main__":
    main() 