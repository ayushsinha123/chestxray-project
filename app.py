import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

IMG_SIZE = 224

# Load model
model = tf.keras.models.load_model(
    'outputs/weights/best_model_final.keras',
    custom_objects={'FocalLoss': lambda **kw: None}
)


def preprocess_image(image):
    img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return tf.expand_dims(img, 0)


def get_gradcam(img_tensor, class_idx):
    last_conv = None
    for layer in model.layers[1].layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads    = tape.gradient(loss, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap  = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def predict(image):
    img_tensor   = preprocess_image(image)
    predictions  = model(img_tensor, training=False).numpy()[0]
    top_idx      = int(np.argmax(predictions))

    # Grad-CAM for top prediction
    heatmap         = get_gradcam(img_tensor, top_idx)
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_display     = np.array(
        tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    ).astype(np.uint8)
    overlay = cv2.addWeighted(img_display, 0.6, heatmap_colored, 0.4, 0)

    results = {cls: float(predictions[i]) for i, cls in enumerate(CLASSES)}
    return results, overlay


with gr.Blocks(title="Chest X-Ray Disease Detector") as demo:
    gr.Markdown("# 🫁 Multi-Disease Chest X-Ray Detection")
    gr.Markdown("Upload a chest X-ray to detect 14 lung diseases using EfficientNetB4 + Grad-CAM")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Chest X-Ray", type="numpy")
            submit_btn  = gr.Button("Analyze", variant="primary")
        with gr.Column():
            label_output   = gr.Label(label="Disease Predictions", num_top_classes=5)
            gradcam_output = gr.Image(label="Grad-CAM Heatmap")

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, gradcam_output]
    )

    gr.Markdown("⚠️ For research purposes only. Not a certified medical device.")

demo.launch()