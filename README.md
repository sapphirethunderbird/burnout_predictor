# Burnout Predictor
This is an AI program trained on FER 2013 to analyze facial expressions and predict the user's risk for burnout.
## Background

I sit at a desk/computer for extended periods of time, and I am prone to stress. There are many known health risks associated with a sedentary lifestyle, such as obesity, osteoporosis, and depression. There are also risks associated with chronic stress, such as anxiety, depression, and problems with memory/focus. A sedentary lifestyle combined with stress is a recipe for disaster, not only for productivity but also overall well-being.<br><br>

There are currently several ways in which an employer can be held responsible for employee’s well-being. Depending on the employer and their country, an employer may be required to provide health insurance or mandate annual health exams for it’s employees. More recently, there has been a push for mental health care as well. This has resulted in, for example, the Japanese government mandating that employers conduct annual stress checks on their employees.<br><br>

However, these are not always as effective as they would seem to be. For example, I spoke to a high school teacher who was required to take a stress check annually. He said that the stress check was adding onto the teachers’ already heavy work load, with a looming deadline. He said that the annual stress check itself was in fact adding to teachers’ stress instead of helping reduce it in the long run.<br><br>

Looking at these issues with our current systems of promoting employee well-being, I decided to utilize AI technology in order to help remediate the issue of job burnout.

## Features

- Analyzes facial expressions in real-time and give predictions of burnout risk based on the live feed.
- Gives the user a notification when the program determines that there is a high risk of burnout
- Prompts the user to take a break when high risk of burnout is detected
- Gives the user choices of what happens to their data by storing past data locally

## Program Overview

### End User's Perspective

When the program is initiated, a window with the live camera feed from the computer is opened. The program will start to analyze the user’s facial expressions in real-time and give predictions of burnout. There are 3 categories of risk, low risk, moderate risk, and high risk. Every instance of high risk is recorded in a CSV file on the computer. In the current version of the program, the window turns red whenever a high risk of burnout is detected.

### Under the Hood

The MobileNet V2 model is trained locally on the [FER 2013 dataset from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). Using the MobileNet V2 model enabled training on consumer hardware as opposed to being limited to server hardware. This also enables the user to train the model on new data in the future.

## What each file does
### burnout_predictor_model.pth
This is the pre-trained model.
### opencv_test.py
This is a test to check opencv functionality.
### trainingv2.py
This is the program to train the model.
### ~~version5.py~~ gui.py
This is the main program.
It initially got labeled as version5.py because when I started writing the program I was too dumb to learn Git properly first and just saved a bunch of checkpoints.
*face-palm*
>`version5.py` can safely be deleted

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
        - For Linux, consult the instructions for your distro
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
