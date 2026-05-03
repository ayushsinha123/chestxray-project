import tensorflow as tf

IMG_SIZE = 224


def build_model(num_classes=14):
    base = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.BatchNormalization()(x)
    x       = tf.keras.layers.Dropout(0.4)(x)
    x       = tf.keras.layers.Dense(256, activation='relu')(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base