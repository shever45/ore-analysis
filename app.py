import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image, ImageOps
import csv
import json
from datetime import datetime
import logging
import tempfile
from skimage import measure

# ============================================
# ЛОГИРОВАНИЕ
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# КОНСТАНТЫ
# ============================================

CLASSES_RU = {
    'otalkovannye': 'ОТАЛЬКОВАННАЯ РУДА',
    'ryadovye': 'РЯДОВАЯ РУДА',
    'trudnoobogatimye': 'ТРУДНООБОГАТИМАЯ РУДА'
}
TALC_THRESHOLD = 4
MODEL_DIR = 'models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# МОДЕЛИ
# ============================================

class OreClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.35), nn.Linear(in_f, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.25), nn.Linear(256, 3)
        )

    def forward(self, x): return self.backbone(x)


logger.info("📦 Загрузка моделей...")
cls_model = OreClassifier().to(DEVICE)
cls_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'classifier_b5_final.pth'), map_location=DEVICE))
cls_model.eval()

seg_models = []
model = smp.Unet(encoder_name='efficientnet-b5', encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'unet_b5_768px_v2_1.pth'), map_location=DEVICE))
model.eval()
seg_models.append(model)
logger.info("✅ Модели загружены")


# ============================================
# ПРЕДОБРАБОТКА
# ============================================

def preprocess_image(img_pil, size=456):
    img = img_pil.convert('RGB')
    img = ImageOps.autocontrast(img, cutoff=2)
    img = img.resize((size, size), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    return img


# ============================================
# GEOSON ЭКСПОРТ
# ============================================

def export_geojson(mask, metrics):
    """Экспорт маски в GeoJSON"""
    contours = measure.find_contours(mask.astype(float), 0.5)

    features = []
    for i, contour in enumerate(contours):
        if len(contour) < 10:
            continue
        coordinates = [[float(p[1]), float(p[0])] for p in contour]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            },
            "properties": {
                "id": i,
                "type": "talc",
                "area_pixels": int(len(contour)),
                "talc_percent": metrics.get('Содержание_талька_%', 0)
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {k: str(v) for k, v in metrics.items()}
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.geojson', mode='w', encoding='utf-8')
    json.dump(geojson, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name


# ============================================
# АНАЛИЗ
# ============================================

def classify_and_analyze(image_pil):
    img_cls = preprocess_image(image_pil, 456).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cls_out = cls_model(img_cls)
        cls_probs = torch.softmax(cls_out, 1).cpu().numpy()[0]

    cls_idx = np.argmax(cls_probs)
    cls_name_en = ['otalkovannye', 'ryadovye', 'trudnoobogatimye'][cls_idx]
    cls_name_ru = CLASSES_RU[cls_name_en]
    cls_confidence = cls_probs[cls_idx]

    img_seg = preprocess_image(image_pil, 768).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        seg_raw = seg_models[0](img_seg)
        seg_prob = torch.sigmoid(seg_raw).squeeze().cpu().numpy()
    talc_bin = (seg_prob > 0.5).astype(np.uint8)
    talc_pct = talc_bin.sum() / talc_bin.size * 100

    if talc_pct > TALC_THRESHOLD:
        final_class = 'ОТАЛЬКОВАННАЯ РУДА'
        reason = f"Содержание талька ({talc_pct:.1f}%) превышает порог {TALC_THRESHOLD}%"
    else:
        final_class = cls_name_ru
        reason = f"Тальк ≤ {TALC_THRESHOLD}%, классификация по срастаниям"

    original = np.array(image_pil.resize((768, 768), Image.BILINEAR))

    # Маска
    overlay = original.copy()
    overlay[talc_bin == 1] = [0, 0, 255]
    result = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

    # Heatmap
    heatmap = cv2.applyColorMap((seg_prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

    metrics = {
        'Класс_руды': final_class,
        'Содержание_талька_%': round(talc_pct, 2),
        'Уверенность_классификатора_%': round(cls_confidence * 100, 1),
        'Вероятность_оталькованная_%': round(cls_probs[0] * 100, 1),
        'Вероятность_рядовая_%': round(cls_probs[1] * 100, 1),
        'Вероятность_труднообогатимая_%': round(cls_probs[2] * 100, 1),
        'Основание': reason,
        'Время_анализа': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    conclusion = f"""🔬 РЕЗУЛЬТАТ АНАЛИЗА
================================
📌 Класс руды: {final_class}
📌 Уверенность: {cls_confidence:.1%}
📌 Содержание талька: {talc_pct:.2f}%
📌 Основание: {reason}
================================
💡 {'Требуется доп. анализ' if talc_pct > 5 else 'Пригодна для переработки'}"""

    logger.info(f"Анализ: {final_class}, тальк={talc_pct:.1f}%")
    return Image.fromarray(result), Image.fromarray(heatmap_overlay), conclusion, metrics, talc_bin


# ============================================
# CSV
# ============================================

def create_csv(metrics_list):
    if not metrics_list:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', newline='', encoding='utf-8-sig')
    if isinstance(metrics_list, dict):
        metrics_list = [metrics_list]
    writer = csv.DictWriter(tmp, fieldnames=metrics_list[0].keys())
    writer.writeheader()
    writer.writerows(metrics_list)
    tmp.close()
    return tmp.name


# ============================================
# ПАКЕТНАЯ ОБРАБОТКА
# ============================================

def batch_process(files):
    if not files:
        return "Нет файлов", None, None
    results, progress = [], []
    all_geojson = {"type": "FeatureCollection", "features": []}

    for i, file in enumerate(files):
        try:
            img = Image.open(file.name).convert('RGB')
            _, _, _, metrics, mask = classify_and_analyze(img)
            results.append(metrics)

            # GeoJSON для каждого файла
            contours = measure.find_contours(mask.astype(float), 0.5)
            for j, contour in enumerate(contours):
                if len(contour) < 10:
                    continue
                coordinates = [[float(p[1]), float(p[0])] for p in contour]
                all_geojson["features"].append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                    "properties": {
                        "file": os.path.basename(file.name),
                        "type": "talc",
                        "area_pixels": int(len(contour))
                    }
                })

            progress.append(f"✅ {i + 1}/{len(files)}: {os.path.basename(file.name)}")
        except Exception as e:
            progress.append(f"❌ {i + 1}/{len(files)}: {e}")

    csv_file = create_csv(results)

    geojson_file = None
    if all_geojson["features"]:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.geojson', mode='w', encoding='utf-8')
        json.dump(all_geojson, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        geojson_file = tmp.name

    return "\n".join(progress), csv_file, geojson_file


# ============================================
# GRADIO
# ============================================

with gr.Blocks(title="🔬 Анализ руды по шлифам") as demo:
    gr.Markdown("# 🔬 Анализ руды по шлифам\n🔵 Синий — тальк | 🔥 Heatmap — уверенность | 🌍 GeoJSON — ГИС")

    with gr.Tabs():
        with gr.TabItem("🔍 Одиночный анализ"):
            with gr.Row():
                input_image = gr.Image(label="📤 Загрузите изображение", type="pil", height=400)
                output_mask = gr.Image(label="🎨 Маска (синий=тальк)", height=400)

            output_heatmap = gr.Image(label="🔥 Карта уверенности", height=400)
            analyze_btn = gr.Button("🔍 Анализировать", variant="primary", size="lg")

            with gr.Row():
                output_text = gr.Textbox(label="📝 Заключение", lines=12)
                output_metrics = gr.JSON(label="📊 Метрики")

            with gr.Row():
                csv_file_output = gr.File(label="📁 CSV")
                geojson_file_output = gr.File(label="🌍 GeoJSON")
            with gr.Row():
                export_csv_btn = gr.Button("📥 Скачать CSV")
                export_geojson_btn = gr.Button("🌍 Скачать GeoJSON")

        with gr.TabItem("📦 Пакетная обработка"):
            batch_files = gr.File(label="📁 Загрузите серию", file_count="multiple", file_types=["image"])
            batch_btn = gr.Button("🔄 Обработать все", variant="primary")
            batch_progress = gr.Textbox(label="📊 Прогресс", lines=10)
            with gr.Row():
                batch_download_csv = gr.File(label="📥 CSV")
                batch_download_geojson = gr.File(label="🌍 GeoJSON")

    current_metrics = gr.State({})
    current_mask = gr.State(None)


    def analyze_wrapper(img):
        mask, heatmap, conclusion, metrics, talc_mask = classify_and_analyze(img)
        return mask, heatmap, conclusion, metrics, metrics, talc_mask


    analyze_btn.click(
        fn=analyze_wrapper,
        inputs=[input_image],
        outputs=[output_mask, output_heatmap, output_text, output_metrics, current_metrics, current_mask]
    )

    export_csv_btn.click(
        fn=lambda m: create_csv([m]) if m else None,
        inputs=[current_metrics],
        outputs=[csv_file_output]
    )

    export_geojson_btn.click(
        fn=lambda m, mask: export_geojson(mask, m) if mask is not None and m else None,
        inputs=[current_metrics, current_mask],
        outputs=[geojson_file_output]
    )

    batch_btn.click(
        fn=batch_process,
        inputs=[batch_files],
        outputs=[batch_progress, batch_download_csv, batch_download_geojson]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7890)