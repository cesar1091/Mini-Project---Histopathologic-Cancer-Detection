import tensorflow as tf
from tensorflow.keras import backend as K

def f1_score(y_true, y_pred, beta=1):
    y_pred = tf.math.sigmoid(y_pred)
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + K.epsilon())
    return K.mean(f1)