from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras import applications
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
import numpy as np
import glob,os
from scipy.misc import imread,imresize
from dates import classes, data_dir

from keras.layers.normalization import BatchNormalization

batch_size = 128

def load_VGG16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    print("Model loaded..!")
    print(base_model.summary())
    return base_model

def extract_features_and_store():
    train_data = np.load(data_dir/'train_images2.npy')
    train_labels = np.load(data_dir/'train_labels2.npy')
    train_data,train_labels = shuffle(train_data,train_labels)

    valid_data = np.load(data_dir/'valid_images2.npy')
    valid_labels = np.load(data_dir/'valid_labels2.npy')
    valid_data, valid_labels = shuffle(valid_data, valid_labels)

    test_data = np.load(data_dir/'test_images2.npy')
    test_labels = np.load(data_dir/'test_labels2.npy')
    test_data, test_labels = shuffle(test_data, test_labels)

    return train_data,train_labels,valid_data,valid_labels,test_data,test_labels

def train_model(train_data, train_labels, validation_data, validation_labels):
    model = Sequential()
    # model.add(Dense(32, input_shape=train_data.shape, activation='relu'))
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1))
    sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # model.load_weights('video_3_512_VGG_no_drop.h5')
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                 ModelCheckpoint('video_3_512_VGG_no_drop.h5', monitor='val_loss', save_best_only=True, verbose=0)]
    epochs = 500
    print(train_data.shape)
    print(valid_data.shape)
    model.fit(train_data, train_labels, validation_data = (validation_data,validation_labels),
              batch_size=batch_size, epochs=epochs, callbacks=callbacks, shuffle=True, verbose=1)
    return model

def test_on_whole_videos(test_data, test_labels, model):
    base_model = load_VGG16_model()
    x_features = base_model.predict(test_data)
    total_video = len(test_labels)

    correct = 0
    
    answer = model.predict(x_features)
    for i in range(total_video):
        if(test_labels[i] == np.argmax(answer[i])):
            correct+=1

    print("correct_video", correct, "total_video", total_video)
    print("The accuracy for video classification of ", total_video, " videos is ", (correct/total_video))

if __name__ == '__main__':
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = extract_features_and_store()
    model = train_model(train_data, train_labels, valid_data, valid_labels)
    test_on_whole_videos(test_data, test_labels, model)
  
