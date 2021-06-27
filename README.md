# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

TensorFlow 1.10.0, Python 3.5.6

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualize.jpg "Visualization"
[image2]: ./examples/training.jpg "training"
[image3]: ./examples/augment.jpg "Augment"
[image4]: ./examples/test_img.jpg "Traffic Sign"
[image5]: ./examples/top_softmax.jpg "softmax"
[image6]: ./examples/speed_limit.jpg "speed limit"
[image7]: ./examples/conv1.jpg "conv1"
[image8]: ./examples/conv2.jpg "conv2"

---
### Writeup

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the `numpy` library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

German traffic sign dataset is unbalanced. Some traffic signs have less samples in the dataset. We need to augment this dataset before using it to train our CNN model. Here is an exploratory visualization of the data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

I decided not to convert the images to grayscale. Because the color images contain more information compared with grayscale and the size of the color image is only (32,32,3). I also plan to develop our CNN based on LeNet. Since LeNet is using a color image as an input, I stick with color image.

I normalized the image data because we always want zero mean and equal invariance. It will help the optimizer to do its job and also converge quickly.

I decided to generate additional data because the German traffic sign dataset is unbalanced. If we use the original dataset to train CNN, our model will be more likely to predict an image as the images which have more samples in the training dataset.

To add more data to the data set, I used the following techniques [Jittering the image](https://github.com/vxy10/ImageAugmentation). By applying rotation, shear, and translation, we can generate new images from the original image.

When there are less than 300 sample images for a traffic sign category, we use the Jittering method to generate more images for that category. The difference between the original data set and the augmented data set is the following:

![alt text][image3]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

We first tried the original LeNet. But our accuracy is less than 0.9. Since a bigger CNN usually has more weights and bias to be adjusted, it usually less likely to be overfitting compared with a smaller network. We tested some bigger CNNs and decided on the following model architecture with a dropout rate 0.7.  My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x9 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x9 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x21 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x21 				|
| Fully connected		| Inputs: 525,  outputs: 216      			|
| Fully connected		| Inputs: 216,  outputs: 121      			|
| Softmax				| Inputs: 121,  outputs: 43       					|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used cross entopy as the loss function and stochastic gradient descent to reduce the loss. The batch size is 128, number of epochs are 31, learning rate is 0.001. We also apply a dropout rate 0.7 to prevent our model overfitting. But the model is still overfitting which is shown as the following image. The training accuracy reached almost 1.0 in an early stage and validation accuracy not improve so much with respect to the epochs.

![alt text][image2]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.958
* test set accuracy of 0.956

We choose to design our CNN based on LeNet. We tried the original LeNet, but the accuracy is less than 0.93 no matter how we fine tune the hyperparameters.

Since the dataset has been augmented, so we think the problem may be in the model itself. We tried different depth of convolutional layer and fully connected layer. In the end we settled down on the CNN model which we mentioned before. Our main criteria to select the depth for CNN is the validation accuracy. If the validation accuracy is less than 0.94, we will try another one.

Our model starts to begin overfitting very early, after 15 epochs. So we decided to apply dropout to the fully connected layers to alleviate overfitting. We tried from 0.1 to 0.9 and finally settled down on 0.7 dropout rate.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

They were all recognized perfectly by our model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The five traffic signs all have been recognized correctly by our trained CNN model. Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road      		| Priority road   									|
| Yield     			| Yield 										|
| Stop				| Stop										|
| No entry	      		| No entry					 				|
| Speed limit 60 km/h			| Speed limit 60 km/h      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the five images I downloaded, the model is quite good at predicting its category. Since the images I found on the web are very clear and in good lighting conditions, our model performed perfectly. I plan to download some less quality images in the future and test the performance again. I expect this model will function well, because there are a lot of low quality images in our training dataset.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Priority road   									|
| 1.00     				| Yield 										|
| 1.00					| Stop											|
| 1.00	      			| No entry					 				|
| 1.00				    | Speed limit 60 km/h      							|


The top five soft max probabilities were:

![alt text][image5]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The original traffic sign image:

![alt text][image6]

From the first layer, we can tell the edges of the 60 km/h speed limit sign have been detected. It is very similar to how a human recognize characters. People also depend on the edges to recognize characters. The first layer images:

![alt text][image7]

It is a little bit difficult for a person to extract some useful informations from the second layer. The second layer images:

![alt text][image8]
