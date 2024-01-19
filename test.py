import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
from cvzone import stackImages

model_a = load_model('A_model_7.h5')
model_b = load_model('B_model_7.h5')

test_path = 'Test_2'
files = os.listdir(test_path)

input_images = []
output_images = []


for file in files:
    img_path = os.path.join(test_path,file)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (128, 128))

    input_images.append(img)
    print(img.shape)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_layer = img_lab[:, :, 0]
    l_layer = np.expand_dims(l_layer, axis=0)

    a_layer = model_a.predict(l_layer)
    b_layer = model_b.predict(l_layer)

    a_layer = np.squeeze(a_layer, axis=0)
    b_layer = np.squeeze(b_layer, axis=0)

    l_layer = l_layer.reshape((128, 128))
    a_layer = a_layer.reshape((128, 128))
    b_layer = b_layer.reshape((128, 128))

    print(a_layer.shape, b_layer.shape, l_layer.shape)

    output_image = tf.stack([l_layer, b_layer, a_layer],axis=-1)
    output_image = np.array(output_image)

    output_image = cv2.cvtColor(output_image, cv2.COLOR_LAB2BGR)

    output_image = cv2.resize(output_image, (500,500))

    output_images.append(output_image)



prediction = stackImages(input_images+output_images, cols=len(input_images), scale=1)
if (not os.path.exists('Prediction')):
    os.mkdir('Prediction')
cv2.imwrite("Prediction/Prediction_6.jpg", prediction)
cv2.imshow("Prediction", prediction)
cv2.waitKey(0)
cv2.destroyAllWindows()