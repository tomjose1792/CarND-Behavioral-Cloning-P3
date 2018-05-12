import os
import cv2
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import argparse #for command line arguments
# Importing libraries for creating mdoel
from sklearn.model_selection import train_test_split #to split out training and testing data 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Reading images
def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

                #Image augmentation Functions

#Random choosing of center, left and right camera images and adusting the steering angles 
def choose_image(data_dir, center, left, right, steering_angle):
    
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle

# Flipping left and right images using cv2
def random_flip(image, steering_angle):
    # im = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    # cv2.imwrite("unflipped_image.png", im)
    # cv2.imwrite("flipped_image.png", cv2.flip(im, 1))
    
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

#Image augmentation funtion
def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)    
    return image, steering_angle

            #Preprocessing techniques

#Cropping the sky and the part of the lane which might have a car in front
def crop(image):
    return image[60:-25, :, :] 

#Resize image to 66,200 for model
def resize(image):
    return cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA)

#RGB to YUV for the model(NVIDIA)
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# Combined Preprocessing function
def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

#Generator Function
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    
    #Generate training image give image paths and associated steering angles
    
    images = np.empty([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    steer = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]

            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 

            images[i] = preprocess(image)
            steer[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steer

#Splitting into training and validation sets
def load_data(args):
    
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values #Input data
    y = data_df['steering'].values #Output data
    #now we can split the data into a training (80%), validation(20%).
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid

# NVIDIA model used (from Paper, End to end learning for self driving car)using Keras
def training_model(args):
      
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)) #Normalisation
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))) #Convolution layers
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu')) #Fully connected layers
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):

    #to have the best_model saved 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    # Python generator.
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

# string to boolean
def s2b(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

# Loading training and validation data and training the model
def main():
    
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    
    print('Parameters')
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    

    data = load_data(args)
    model = training_model(args) #building Training model
    train_model(model, args, *data) #train model, saved as model.h5


if __name__ == '__main__':
    main()

