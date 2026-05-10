import tensorflow as tf
import numpy as np
import cv2

from src.config import IMG_SIZE


# ============================================================
# GRAD-CAM MODEL BUILDER
# ============================================================

def build_grad_model(model):
    """
    Finds the DenseNet sub-model inside the wrapper model,
    auto-detects the last Conv2D layer, and builds the
    Grad-CAM sub-model that outputs (conv_features, densenet_output).

    Returns: (grad_model, densenet_layer) or (None, None) on failure.
    """

    # Auto-detect DenseNet sub-model
    densenet = None
    for layer in model.layers:
        if 'densenet' in layer.name.lower():
            densenet = layer
            print(f"[GradCAM] Found DenseNet sub-model: '{layer.name}'")
            break

    if densenet is None:
        print("[GradCAM] WARNING: No DenseNet sub-model found. Grad-CAM disabled.")
        return None, None

    # Auto-detect last Conv2D inside DenseNet
    last_conv = None
    for layer in reversed(densenet.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            print(f"[GradCAM] Auto-detected last conv layer: '{last_conv}'")
            break

    if last_conv is None:
        print("[GradCAM] WARNING: No Conv2D layer found in DenseNet. Grad-CAM disabled.")
        return None, None

    # Build grad model
    try:
        grad_model = tf.keras.Model(
            inputs=densenet.input,
            outputs=[
                densenet.get_layer(last_conv).output,
                densenet.output
            ]
        )
        print(f"[GradCAM] Grad-CAM model built successfully.")
        return grad_model, densenet
    except Exception as e:
        print(f"[GradCAM] WARNING: Could not build grad_model: {e}")
        return None, None


# ============================================================
# HEAD LAYER VALIDATOR
# ============================================================

def get_head_layers(model):
    """
    Checks that all expected head layer names exist in the model.
    Returns a dict of {name: name} for found layers, and prints
    warnings for any that are missing.
    """
    expected = ['gap', 'head_bn', 'drop1', 'fc1', 'fc_bn', 'drop2', 'predictions']
    found = {}

    for name in expected:
        try:
            model.get_layer(name)
            found[name] = name
            print(f"[GradCAM] Head layer found: '{name}'")
        except Exception:
            print(f"[GradCAM] WARNING: Head layer '{name}' not found in model.")

    return found


# ============================================================
# GRAD-CAM COMPUTATION
# ============================================================

def get_gradcam(model, grad_model, head_layers, img_tensor, class_idx):
    """
    Computes the Grad-CAM heatmap for a given class index.

    Args:
        model:       Full wrapper model.
        grad_model:  Sub-model outputting (conv_features, densenet_output).
        head_layers: Dict of verified head layer names from get_head_layers().
        img_tensor:  Preprocessed image tensor of shape (1, H, W, 3).
        class_idx:   Index of the class to visualize.

    Returns:
        heatmap: np.ndarray of shape (IMG_SIZE, IMG_SIZE), float32, range [0, 1].
                 Returns None if computation fails.
    """
    if grad_model is None:
        print("[GradCAM] Skipped: grad_model not available.")
        return None

    required = ['gap', 'head_bn', 'drop1', 'fc1', 'fc_bn', 'drop2', 'predictions']
    if not all(k in head_layers for k in required):
        print("[GradCAM] Skipped: one or more head layers missing.")
        return None

    try:
        with tf.GradientTape() as tape:
            conv_out, base_out = grad_model(img_tensor, training=False)
            tape.watch(conv_out)

            x = model.get_layer(head_layers['gap'])(base_out)
            x = model.get_layer(head_layers['head_bn'])(x, training=False)
            x = model.get_layer(head_layers['drop1'])(x, training=False)
            x = model.get_layer(head_layers['fc1'])(x)
            x = model.get_layer(head_layers['fc_bn'])(x, training=False)
            x = model.get_layer(head_layers['drop2'])(x, training=False)
            preds = model.get_layer(head_layers['predictions'])(x)
            loss  = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)

        if grads is None:
            print("[GradCAM] Gradients are None — graph not connected. Skipping.")
            return None

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_np = conv_out[0].numpy()
        pool_np = pooled.numpy()

        heatmap = np.dot(conv_np, pool_np)
        heatmap = np.maximum(heatmap, 0)

        vmax = heatmap.max()
        if vmax < 1e-8:
            # Flat heatmap — no signal, return neutral
            heatmap = np.ones_like(heatmap) * 0.5
        else:
            heatmap /= vmax

        # Resize to IMG_SIZE x IMG_SIZE
        heatmap = tf.image.resize(
            heatmap[..., np.newaxis],
            [IMG_SIZE, IMG_SIZE]
        ).numpy().squeeze()

        return heatmap.astype(np.float32)

    except Exception as e:
        print(f"[GradCAM] Error during computation: {e}")
        return None


# ============================================================
# OVERLAY
# ============================================================

def overlay_gradcam(image, heatmap, alpha=0.4):
    """
    Blends the original image with the Grad-CAM heatmap.

    Args:
        image:   Original image as np.ndarray (H, W, 3), uint8.
        heatmap: Normalized heatmap from get_gradcam(), float32 [0, 1].
        alpha:   Heatmap blend weight (default 0.4).

    Returns:
        overlay: np.ndarray (IMG_SIZE, IMG_SIZE, 3), uint8.
    """
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_display = tf.image.resize(
        image, [IMG_SIZE, IMG_SIZE]
    ).numpy().astype(np.uint8)

    overlay = cv2.addWeighted(
        img_display,    1 - alpha,
        heatmap_colored, alpha,
        0
    )
    return overlay