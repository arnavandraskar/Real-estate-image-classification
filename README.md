# Real-estate-image-classification

This project is a Flask-based web application for image classification of real estate images using a pre-trained ResNet model. The application takes an image as input, preprocesses it and uses the pre-trained model to make a prediction about the category of the real estate image.

## Introduction:
Real estate agents spend considerable time sifting through huge amounts of image data in order to determine which ones to use in advertising a property .This is a real estate image classification problem where the objective is to classify images with respective classes with high accuracy.

## Business Problem:
Posting pictures is a necessary part of advertising a home for sale. Agents typically sort through dozens of images from which to pick the most complimentary ones. This is a manual effort involving annotating images accompanied by descriptions (bedroom, bathroom, attic, etc.). When volumes are small, manual annotation is not a problem, but there is a point where this becomes too burdensome and ultimately infeasible.

## Dataset:
The dataset contains 5859 images distributed across 6 classes: Bathroom, Bedroom, Living Room, Kitchen, Front Yard, and Backyard. The number of images varies across classes, but each class contains at least 700 images. Images are in .jpg, .jpeg and .PNG format.
source: https://drive.google.com/u/0/uc?id=0B761qYXle4lYZHlMQ01rTEtva3M&export=download

## Requirements:

Flask

TensorFlow

Keras

gevent

## Installation:

Clone the repository to your local machine.
Install the required dependencies by running the command pip install -r requirements.txt.
Run the application by running the command python rei_app.py.

## Usage:

Access the application by going to http://localhost:8080/ in your web browser.
Select an image to upload and click the "Predict" button to make a prediction about the category of the real estate image.

## File Descriptions:

rei_app.py: Contains the main Flask application code.
util.py: Contains utility functions for image preprocessing.
resnet_2_new: Contains the pre-trained ResNet model in JSON format.
resnet_new.h5: Contains the pre-trained weights for the ResNet model.
class_dicts: Contains the dictionary of class labels for the model.

## Author:

Arnav Andraskar

## License:

This project is licensed under the MIT License.
