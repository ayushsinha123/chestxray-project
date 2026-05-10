import os
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], 'GPU')

# ============================================================
# CONFIG
# ============================================================

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

THRESHOLDS = {
    'Atelectasis': 0.4729, 'Cardiomegaly': 0.4002, 'Effusion': 0.4951,
    'Infiltration': 0.5025, 'Mass': 0.4228, 'Nodule': 0.4243,
    'Pneumonia': 0.3641, 'Pneumothorax': 0.4236, 'Consolidation': 0.4092,
    'Edema': 0.3862, 'Emphysema': 0.3938, 'Fibrosis': 0.3751,
    'Pleural_Thickening': 0.4044, 'Hernia': 0.3415
}

IMG_SIZE = 320

# ============================================================
# ASYMMETRIC LOSS
# ============================================================

class AsymmetricLoss(tf.keras.losses.Loss):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05,
                 label_smoothing=0.02, name='AsymmetricLoss'):
        super().__init__(name=name)
        self.gamma_neg       = gamma_neg
        self.gamma_pos       = gamma_pos
        self.clip            = clip
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true = y_true * (1.0 - self.label_smoothing) \
               + 0.5    *         self.label_smoothing

        y_pred_neg = tf.clip_by_value(y_pred - self.clip, 0.0, 1.0)
        y_pred     = tf.clip_by_value(y_pred,             1e-7, 1.0 - 1e-7)
        y_pred_neg = tf.clip_by_value(y_pred_neg,         1e-7, 1.0 - 1e-7)

        log_pos = tf.math.log(y_pred)
        log_neg = tf.math.log(1.0 - y_pred_neg)

        p_t_pos = y_pred
        p_t_neg = 1.0 - y_pred_neg
        w_pos   = tf.pow(1.0 - p_t_pos, self.gamma_pos)
        w_neg   = tf.pow(1.0 - p_t_neg, self.gamma_neg)

        loss = -(
            y_true         * w_pos * log_pos +
            (1.0 - y_true) * w_neg * log_neg
        )
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'gamma_neg': self.gamma_neg, 'gamma_pos': self.gamma_pos,
            'clip': self.clip, 'label_smoothing': self.label_smoothing,
        })
        return cfg

# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    'outputs/weights/best_model_phaseC.keras',
    custom_objects={'AsymmetricLoss': AsymmetricLoss},
    compile=False
)

# ============================================================
# GRAD-CAM SETUP — auto-detect layer names
# ============================================================

print("\nTop-level model layers:")
for layer in model.layers:
    print(f"  {layer.name}")

# Auto-detect the DenseNet sub-model by name
densenet = None
for layer in model.layers:
    if 'densenet' in layer.name.lower():
        densenet = layer
        print(f"\nFound DenseNet sub-model: '{layer.name}'")
        break

if densenet is None:
    print("WARNING: No DenseNet sub-model found. Grad-CAM will be disabled.")

# Auto-detect the last Conv2D layer inside DenseNet
LAST_CONV = None
if densenet is not None:
    for layer in reversed(densenet.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            LAST_CONV = layer.name
            print(f"Auto-detected last conv layer: '{LAST_CONV}'")
            break

grad_model = None
if densenet is not None and LAST_CONV is not None:
    try:
        grad_model = tf.keras.Model(
            inputs=densenet.input,
            outputs=[
                densenet.get_layer(LAST_CONV).output,
                densenet.output
            ]
        )
        print(f"Grad-CAM model built successfully.")
    except Exception as e:
        print(f"WARNING: Could not build grad_model: {e}")

# Auto-detect head layer names
HEAD_LAYERS = {}
for name in ['gap', 'head_bn', 'drop1', 'fc1', 'fc_bn', 'drop2', 'predictions']:
    try:
        model.get_layer(name)
        HEAD_LAYERS[name] = name
        print(f"Head layer found: '{name}'")
    except Exception:
        print(f"WARNING: Head layer '{name}' not found in model.")

# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(image):
    img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.densenet.preprocess_input(img)
    return tf.expand_dims(img, 0)

# ============================================================
# GRAD-CAM
# ============================================================

def get_gradcam(img_tensor, class_idx):
    if grad_model is None:
        print("Grad-CAM skipped: grad_model not available.")
        return None

    required = ['gap', 'head_bn', 'drop1', 'fc1', 'fc_bn', 'drop2', 'predictions']
    if not all(k in HEAD_LAYERS for k in required):
        print("Grad-CAM skipped: one or more head layers missing.")
        return None

    try:
        with tf.GradientTape() as tape:
            conv_out, base_out = grad_model(img_tensor, training=False)
            tape.watch(conv_out)

            x = model.get_layer(HEAD_LAYERS['gap'])(base_out)
            x = model.get_layer(HEAD_LAYERS['head_bn'])(x, training=False)
            x = model.get_layer(HEAD_LAYERS['drop1'])(x, training=False)
            x = model.get_layer(HEAD_LAYERS['fc1'])(x)
            x = model.get_layer(HEAD_LAYERS['fc_bn'])(x, training=False)
            x = model.get_layer(HEAD_LAYERS['drop2'])(x, training=False)
            preds = model.get_layer(HEAD_LAYERS['predictions'])(x)
            loss  = preds[:, class_idx]

        grads  = tape.gradient(loss, conv_out)

        if grads is None:
            print("Grad-CAM: gradients are None. Skipping.")
            return None

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_np = conv_out[0].numpy()
        pool_np = pooled.numpy()

        heatmap = np.dot(conv_np, pool_np)
        heatmap = np.maximum(heatmap, 0)

        vmax = heatmap.max()
        if vmax < 1e-8:
            heatmap = np.ones_like(heatmap) * 0.5
        else:
            heatmap /= vmax

        heatmap = tf.image.resize(
            heatmap[..., np.newaxis],
            [IMG_SIZE, IMG_SIZE]
        ).numpy().squeeze()

        return heatmap.astype(np.float32)

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

# ============================================================
# PREDICT
# ============================================================

def predict(image):
    if image is None:
        return {"Error": 1.0}, None

    img_tensor = preprocess_image(image)
    preds      = model(img_tensor, training=False).numpy()[0]

    results = {}
    for i, cls in enumerate(CLASSES):
        prob      = float(preds[i])
        threshold = THRESHOLDS[cls]
        if prob >= threshold:
            results[f'✅ {cls}'] = prob
        else:
            results[f'{cls}'] = prob

    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    top_idx = int(np.argmax(preds))
    heatmap = get_gradcam(img_tensor, top_idx)

    img_display = tf.image.resize(
        image, [IMG_SIZE, IMG_SIZE]
    ).numpy().astype(np.uint8)

    # If Grad-CAM failed, return plain resized image
    if heatmap is None:
        return results, img_display

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        img_display, 0.6,
        heatmap_colored, 0.4,
        0
    )

    return results, overlay

# ============================================================
# UI
# ============================================================

with gr.Blocks(title="Chest X-Ray Disease Detector") as demo:

    gr.Markdown("# 🫁 Multi-Disease Chest X-Ray Detection")
    gr.Markdown(
        """
        **Model:** DenseNet121 + Asymmetric Loss + Grad-CAM  
        **Input:** 320×320  
        **Final Test ROC-AUC:** 0.8195  
        **Classes:** 14 chest pathologies from NIH ChestX-ray14
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Chest X-Ray", type="numpy")
            submit_btn  = gr.Button("Analyze", variant="primary")
        with gr.Column():
            label_output = gr.Label(
                label="Disease Predictions (✅ = above threshold)",
                num_top_classes=14
            )
            gradcam_output = gr.Image(label="Grad-CAM Visualization")

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, gradcam_output]
    )

    gr.Markdown("⚠️ For research purposes only. Not a certified medical device.")

demo.launch(server_name="0.0.0.0")