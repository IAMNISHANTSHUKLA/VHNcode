# Example using TensorFlow and Keras
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom output layer for book detection
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(4, activation='sigmoid')  # 4 for (x, y, width, height) of bounding box
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
