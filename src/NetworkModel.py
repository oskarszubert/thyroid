import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import numpy as np
import seaborn as sn
import pandas as pd


class NetworkModel:
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

    def create_learning_plots(self, model_history, filename):
        path = '../gen/plots/'
        if not os.path.exists(path):
            os.makedirs(path)

        # plt.clf()
        # plt.plot(model_history.history['acc'])
        # plt.plot(model_history.history['val_acc'])
        # plt.title('model metric: accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig(path+filename+'_acc.png')

        plt.clf()

        plt.plot(model_history.history['f1_score'])
        plt.plot(model_history.history['val_f1_score'])
        plt.title('model metric: f1_score')
        plt.ylabel('f1_score')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(path+filename+'_f1score.png')

        plt.clf()

        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(path, filename+'_loss.png'))

        plt.clf()

    def create_confusion_matrix(self, model, X_train, y_train, filename):
        path = '../gen/confusion_matrix/'
        if not os.path.exists(path):
            os.makedirs(path)
        predictions_with_vector = model.predict(X_train)
        predictions = []
        for vect in predictions_with_vector:
            predictions.append(np.argmax(vect))

        matrix = confusion_matrix(predictions, y_train)
        matrix = np.round(matrix/len(predictions), 3)

        df_cm = pd.DataFrame(matrix)

        # sn.set(font_scale=1.4) # for bigger size, uncomment for ann_thyroid
        f, ax = plt.subplots(figsize=(16, 9))
        sn.heatmap(df_cm, annot=True, linewidths=1, ax=ax)

        ax.set( title="Confusion Matrix",
                xlabel="Predicted",
                ylabel="Actual")

        plt.savefig(os.path.join(path, filename+'.png'))
        plt.clf()