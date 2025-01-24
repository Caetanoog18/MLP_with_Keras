import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


class Train:
    def __init__(self):
        self.fashion_mnist = keras.datasets.fashion_mnist

    def train(self):
        # Loading the database
        (X_train_full, y_train_full), (X_test, y_test) = self.fashion_mnist.load_data()

        # print(X_train_full.shape, X_train_full.dtype)
        # print(y_train_full.shape, y_train_full.dtype)
        # print(X_test.shape, X_test.dtype)
        # print(y_test.shape, y_test.dtype)


        # Creating a validate set and scaling input features
        X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:] / 255.0
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

        # Defining class names
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

        # print(class_names[y_train[0]])

        # Creating the neural network model
        model = Sequential([
            Flatten(input_shape=[28, 28]),
            Dense(300, activation="relu"),
            Dense(100, activation="relu"),
            Dense(10, activation="softmax"),
        ])

        model.summary()

        hidden_1 = model.layers[1]
        # print(hidden_1.name)

        # Checking if the layer named 'dense' is the same as hidden_1
        is_same_layer = model.get_layer('dense') is hidden_1
        print(f"Is the layer named 'dense' the same as 'hidden_1'? {is_same_layer}")

        # Verifying the model parameters
        weight, biases = hidden_1.get_weights()
        # print("Weights of the 'hidden_1' layer:")
        # print(weight)
        #
        # print("\nBiases of the 'hidden_1' layer:")
        # print(biases)

        # Show their shapes to get a summary
        print(f"\nWeights shape: {weight.shape}")
        print(f"Biases shape: {biases.shape}")

        # Compiling the model
        model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        # Training and evaluating the model
        history = model.fit(X_train, y_train, epochs=3,
                            validation_data=(X_valid, y_valid))







