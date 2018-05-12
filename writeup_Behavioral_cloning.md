# **Behavioral Cloning Project ** 

## Writeup


The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/Center_lane_image.jpg "Center_lane_image"
[image2]: ./Images/Recovering_image_1.jpg "Recovery Image_1"
[image3]: ./Images/Recovering_image_2.jpg "Recovery Image_2"
[image4]: ./Images/Normal_image.png "Normal_image"
[image5]: ./Images/Flipped_image.png "Flipped_image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_Behavioral_cloning.md

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
I did a small modification to the udacity provided drive.py file by including the preprocessing functions in the file.(code_line=65)
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used a NVIDIA model architecture illustrated in the paper, 'End to End learning for Self driving cars' by NVIDIA corporation. The model consists of 9 layers, a normalisation layer, 5 convolution layers, 3 fully connected layers (code_line:111-124). The Convolution neural network has the convolution layers with 3x3 and 5x5 filter sizes with 2x2 strides for first 3 and non-strided for last 2 layers. The model includes ELU activation functions to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer before introducing into the convolution layers. ELUs have improved learning characteristics compared to the units with other activation functions.  In contrast to ReLUs, ELUs have negative values which allows them to push mean unit activations closer to zero like batch normalization but with lower computational complexity.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer with a probability of 0.5 in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting using the train_model() and training_model() functions. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with the learning rate of 0.0001 with 10 epochs passed on by parser module.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used flipping of images from left to right for training. I used random choices of left, right and center camera images for data augmentation which also constitutes the training data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As I was going through the project tutorials on Udacity I came across the NVIDIA paper using a CNN for self driving cars. Since the model showed good results as mentioned in the paper, I thought of employing it for the project for training.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
The dropout layer in the model is to combat overfitting.
Then I added some preprocessing techniques already mentioned in the Udacity tutorials and found some online and then carried out data augmentation,i.e flipping and random choice of camera images and corresponding steering angle values.

I drove the car around the track one for five laps and, with data augmentaion, I collected enough data for training.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots like the hard turns where the vehicle strayed off the track. To improve the driving behavior in these cases, I had collect new training data with better car stability. The new training data helped the car to stay on track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The NVIDIA model architecture (model.py lines 111-124) consisted of a convolution neural network with the following layers. 
Input shape to model after preprocessing 66x200x3 YUV images.

| Layer         		|     Output Shape	        					| 
|:---------------------:|:---------------------------------------------:| 
| Normalisation, Lambda	| 66x200x3 YUV image   							| 
| Convolution 5x5     	|  31x98x24
| Convolution 5x5     	|  14x47x36
| Convolution 5x5     	|  5x22x48
| Convolution 5x5     	|  3x20x64
| Convolution 5x5     	|  1x18x64
| Dropout					|	Keep Prob=0.5
| Flatten     			|  1152
| Dense					| 100
| Dense					| 50
| Dense					| 10
| Dense					| 1



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded approx. 3 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded approx. 1 lap of certain points the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back if it strays off the track. These images show what a recovery looks like :

![alt text][image2]
![alt text][image3]

I had recorded 1 lap of the car driving clock-wise around the track one to avoid the predominance of steering towards to the left and hence right turns were also added to data.

I captured training data only from the track one provided in the simulator.

To augment the data set, I also flipped images and angles thinking that this would create diversity in the data. For example, here is an image that has then been flipped:

Normal Image

![alt text][image4]

Flipped Image

![alt text][image5]

and random choice of images from left, right and center camera images were sent to training of the model.

After the collection process,I preprocessed the data by cropping out the sky and the area in the lane which might have a car in front, resizing to (66x200x3) and later converting the images from RGB to YUV channel for the NVIDIA model.

I finally randomly shuffled the data set and put 20% of the data into a validation set and 80% in training set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the saturation of the loss value of the model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
