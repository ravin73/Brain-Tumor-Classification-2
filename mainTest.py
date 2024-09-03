import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.h5')
image = cv2.imread('//Users//ravinyadav//Downloads//Brain Tumor Classification 2//pred//pred1.jpg')

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)
result = model.predict(input_img)

# Get the index corresponding to the class with the highest probability
predicted_class = np.argmax(result, axis=1)

print("Predicted class:", predicted_class)
