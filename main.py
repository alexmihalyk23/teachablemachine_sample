import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

cap  = cv2.VideoCapture(0)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# image = Image.open('test3.jpg')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
# image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
#

while True:
    ret, frame = cap.read()
    image = Image.fromarray(frame)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    print("without mask: ", prediction[0][0])
    print("with mask: ", prediction[0][1])

    im_np = np.asarray(image)
    if prediction[0][0] >= 0.75:
        cv2.putText(im_np, "No Mask", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    elif prediction[0][1] >= 0.75:
        cv2.putText(im_np, "With Mask", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(im_np, "Not Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("test", im_np)
    keyCode = cv2.waitKey(1)

    if cv2.getWindowProperty("test", cv2.WND_PROP_VISIBLE) < 1:
        break
