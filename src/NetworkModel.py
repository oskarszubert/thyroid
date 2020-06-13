import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
import sklearn

class NetworkModel():
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_score(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))


    def create_model(self, layers, input_size, output_size , momentum):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(layers[0], activation='relu', input_shape=(input_size,)))

        for neuron in layers[1:]:
            model.add(tf.keras.layers.Dense(neuron, activation='relu'))

        model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

        model.summary()
        sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=momentum, nesterov=True)
        model.compile(optimizer=sgd,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', self.f1_score])

        return model
