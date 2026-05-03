import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, pos_weights=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weights = pos_weights

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce    = -y_true * tf.math.log(y_pred) \
                 - (1 - y_true) * tf.math.log(1 - y_pred)
        if self.pos_weights is not None:
            bce = bce * (y_true * self.pos_weights + (1 - y_true))
        p_t   = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = self.alpha * tf.pow(1 - p_t, self.gamma) * bce
        return tf.reduce_mean(focal)