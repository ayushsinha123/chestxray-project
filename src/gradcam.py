import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

IMG_SIZE = 224


def get_gradcam(model, image_tensor, class_idx, layer_name):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        loss = predictions[:, class_idx]

    grads    = tape.gradient(loss, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap  = conv_out @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    original_uint8  = np.uint8(255 * original_img)
    overlay         = cv2.addWeighted(original_uint8, 0.6, heatmap_colored, 0.4, 0)
    return overlay


def get_last_conv_layer(model):
    for layer in model.layers[1].layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam_grid(model, df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    last_conv = get_last_conv_layer(model)
    print(f"Using conv layer: {last_conv}")

    sample_classes = ['Cardiomegaly', 'Effusion', 'Pneumothorax', 'Atelectasis']
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for row, cls_name in enumerate(sample_classes):
        idx      = df[df[cls_name] == 1].index[0]
        img_path = df.loc[idx, 'image_path']
        cls_idx  = CLASSES.index(cls_name)

        img_raw = tf.io.read_file(img_path)
        img_raw = tf.image.decode_png(img_raw, channels=3)
        img_raw = tf.image.resize(img_raw, [IMG_SIZE, IMG_SIZE])
        img_display   = img_raw.numpy() / 255.0
        img_processed = tf.cast(img_raw, tf.float32)
        img_processed = tf.keras.applications.efficientnet.preprocess_input(img_processed)
        img_tensor    = tf.expand_dims(img_processed, 0)

        pred = model(img_tensor, training=False).numpy()[0][cls_idx]

        try:
            heatmap = get_gradcam(model, img_tensor, cls_idx, layer_name=last_conv)
            overlay = overlay_gradcam(img_display, heatmap)

            axes[row, 0].imshow(img_display, cmap='gray')
            axes[row, 0].set_title(f'Original\n{cls_name}', fontsize=10)
            axes[row, 0].axis('off')

            axes[row, 1].imshow(heatmap, cmap='jet')
            axes[row, 1].set_title(f'Heatmap\nPred: {pred:.3f}', fontsize=10)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(overlay)
            axes[row, 2].set_title('Overlay', fontsize=10)
            axes[row, 2].axis('off')

        except Exception as e:
            print(f"Grad-CAM failed for {cls_name}: {e}")

    plt.suptitle('Grad-CAM Visualizations — Disease Localization', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gradcam_results.png', dpi=150)
    plt.show()
    print(f"Grad-CAM saved to {save_dir}/gradcam_results.png")