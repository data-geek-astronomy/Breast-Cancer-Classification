# evaluate_model.py

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, recall_score
from data_preprocessing import X_test, y_test

# Load the model
model = load_model('cancernet_model.h5')

# Evaluate the model
predictions = model.predict(X_test, batch_size=32)
y_pred = np.argmax(predictions, axis=1)

# Calculate metrics
cm = confusion_matrix(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Confusion Matrix:\n{cm}\n")
print(f"Recall: {recall}")
