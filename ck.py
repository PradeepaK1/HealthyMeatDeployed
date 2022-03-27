from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import streamlit as st
from PIL import Image

image = Image.open('imgs/ic.jpeg')
st.image(image,width=600)
st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)

i1='imgs/r1.jpeg'
i2='imgs/r2.jpeg'
i3='imgs/r3.jpeg'
i4='imgs/r4.jpeg'
i5='imgs/r5.jpeg'

l=[i1,i2,i3,i4,i5]
p=['imgs/chick.jpg']

st.image(p, width=200, caption=[""])
cb, mb = st.columns([.4,1])



img_width, img_height = 224, 224
train_data_dir = 'chicken_train2'
validation_data_dir = 'chicken_test2'
nb_train_samples =22
nb_validation_samples = 22
epochs = 3
batch_size = 5
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer='rmsprop',
metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
model.save('model_saved.h5')
import random
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import keras
import tensorflow
from keras import backend as K
from keras.models import model_from_config, Sequential
from keras.models import load_model

model = keras.models.load_model('model_saved.h5')


MAX_FRAMES = 2



ii = st.file_uploader("Choose an image...", type=".jpg")
if ii is not None:
    nn=ii.name
    try:
        image = load_img('testfile/'+str(nn), target_size=(227, 227))
        img = np.array(image)
        img = img / 255.0
        img = img.reshape(1,227,227,3)
        label = model.predict(img)
        st.write("Predicted Class (Towards 0% - Stale Meat, Towards 100% - Fresh Meat): ",abs(100-round((label[0][0])*100)), '%')
        st.markdown("<h3 style='text-align: center; color: black;'>Freshness Report</h3>", unsafe_allow_html=True)
        st.write("Chosen Meat: Chicken")

        if ((label[0][0])<.30):
            kk = ['imgs/chick.jpg','imgs/90p.jpg', 'imgs/80p.jpg', 'imgs/wb.jpg']
            st.image(kk, width=170, caption=["Chosen Meat","Color","Freshness","Decision"])
            st.markdown("<h3 style='text-align: center; color: black;'>Healthy Reciecipe of the day</h3>", unsafe_allow_html=True)

            j=random.choice(l)
            im = Image.open(j)
            st.image(im)
        elif (.30<(label[0][0])<.50):
            kk = ['imgs/chick.jpg','imgs/80p.jpg', 'imgs/70p.jpg', 'imgs/wb.jpg']
            st.image(kk, width=170, caption=["Chosen Meat","Color","Freshness","Decision"])
            st.markdown("<h3 style='text-align: center; color: black;'>Healthy Reciecipe of the day</h3>", unsafe_allow_html=True)
            j=random.choice(l)
            im = Image.open(j)
            st.image(im)
        elif (.50<(label[0][0])<.60):
            kk = ['imgs/chick.jpg','imgs/50p.jpg', 'imgs/40p.jpg', 'imgs/harm.jpg']
            st.image(kk, width=170, caption=["Chosen Meat","Color","Freshness","Decision"])
        elif(.60<(label[0][0])<1):
            kk = ['imgs/chick.jpg','imgs/40p.jpg', 'imgs/30p.jpg', 'imgs/harm.jpg']
            st.image(kk, width=170, caption=["Chosen Meat","Color","Freshness","Decision"])

        
    except:
        image = Image.open('imgs/index.jpg')
        st.image(image,width=200)
        st.write("Try with a meat image!")
