# train_cancernet.py

from tensorflow.keras.optimizers import Adam
from cancernet_model import build_cancernet
from data_preprocessing import augmentation, X_train, X_val, y_train, y_val

# Model parameters
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (50, 50, 3)

# Compile model
model = build_cancernet(IMAGE_DIMS, 2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
H = model.fit(
    augmentation.flow(X_train, y_train, batch_size=BS),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // BS,
    epochs=EPOCHS, verbose=1)

# Save the model
model.save('cancernet_model.h5')
