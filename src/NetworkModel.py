import tensorflow as tf

class NetworkModel():
    def create_model(self, layers, input_size, output_size , momentum):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(layers[0], activation='relu', input_shape=(input_size,)))

        for neuron in layers[1:]:
            model.add(tf.keras.layers.Dense(neuron, activation='relu'))

        model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

        model.summary()
        sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=momentum, nesterov=True)

        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
