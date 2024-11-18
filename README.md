# Burnout Predictor
This is an AI program trained on FER 2013 to analyze facial expressions and predict the user's risk for burnout.
## This data is the FER 2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
## What each file does
### burnout_predictor_model.pth
This is the pre-trained model.
### opencv_test.py
This is a test to check opencv functionality.
### trainingv2.py
This is the program to train the model.
### ~~version5.py~~ gui.py
This is the main program.
It got labeled as version5.py because when I started writing the program I was too dumb to learn Git properly first and just saved a bunch of checkpoints.
<<<<<<< HEAD
*face-palm* 
=======
*face-palm*

## Installation
1. Clone the repo
`https://github.com/sapphirethunderbird/burnout_predictor.git`
2. Install dependencies
    - This program requires Python 3.12.7
    - Python Libraries
    `pip install opencv-python pillow torch torchvision`
    - How you install `tkinter` will depend on your operating system. 
        - For OSX: `brew install python-tk`
        - For Windows, `tkinter` should be included in the Python installer
        - For Linux, consult the instructions for your distrobution
### Training the model 
- For training the model, download the FER 2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Make sure to match the path for your dataset in `trainingv2.py`
- This will require the MobileNet_V2 model

## Further Improvements
- Improve model accuracy
- Add the ability to train based on user data (train on facial expressions from the feed)
- Add an interactive chart
- Possible Raspberry Pi integration
- Localization


>>>>>>> e384675e155b974af08dc0e77bcef942eb4cc15e
