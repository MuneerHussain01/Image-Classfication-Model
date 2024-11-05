import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

st.header('Image Classification Model')

model = load_model('C:/Users/B.J COMP/Downloads/Fruits_Vegetables/Image_classify.keras')

data_category = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

image_height = 180
image_width = 180


image = st.text_input('Enter Image Name', 'Apple.jpg')


image_load = tf.keras.utils.load_img(image, target_size=(image_height, image_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr, 0)


predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Veg/Fruit in image is ' + data_category[np.argmax(score)])
st.write(f'with accuracy of {np.max(score)*100:.2f}%')


