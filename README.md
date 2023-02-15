# Real-estate-image-classification

This project is a Flask-based web application for image classification of real estate images using a pre-trained ResNet model. The application takes an image as input, preprocesses it and uses the pre-trained model to make a prediction about the category of the real estate image.

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
