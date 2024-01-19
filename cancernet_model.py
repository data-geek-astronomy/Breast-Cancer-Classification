# cancernet_model.py

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def build_cancernet(input_shape, classes):
    # Load the VGG16 network, ensuring the head FC layer sets are left off
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))

    # Construct the head of the model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(classes, activation="softmax")(headModel)

    # Place the head FC model on top of the base model (this will become the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # Loop over all layers in the base model and freeze them so they will not be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    return model

# Construct the model
model = build_cancernet((50, 50, 3), 2)  # width, height, depth, classes (2 for binary classification)
