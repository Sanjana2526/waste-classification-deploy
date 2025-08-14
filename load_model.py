from keras.models import load_model

model = load_model("model/my_model.keras")

# Print model summary to confirm it loaded
model.summary()
