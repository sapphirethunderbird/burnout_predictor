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
### gui.py
This is the main program. (`version5.py` is an old checkpoint and can be safely deleted.)

## Installation
1. Clone the repo
`https://github.com/sapphirethunderbird/burnout_predictor.git`
2. Install dependencies
    - This program requires Python 3.10+
    - `pip install -r requirements.txt`
    - `tkinter` is a system package, not available via pip:
        - macOS: `brew install python-tk`
        - Windows: included in the Python installer
        - Linux: consult the instructions for your distro

### Training the model
- Download the [FER 2013 dataset from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and extract it so that `archive/train/` exists in the project root
- Run `python trainingv2.py` — training prints loss and validation accuracy per epoch
- A pre-trained model (`burnout_predictor_model.pth`) is included so you can run the app without retraining

## Modeling assumption
A single frame showing a sad or fearful expression is not burnout. Burnout is chronic — it shows up as a sustained pattern over time. This app reflects that: the on-screen risk indicator is based on the fraction of high-risk predictions over a rolling 60-frame window, not any individual frame. High-risk events are also logged to a CSV file and visualizable as a heatmap in the Chart Analysis tab.

## Further Improvements
- Improve model accuracy
- Add the ability to train based on user data (train on facial expressions from the feed)
- Add an interactive chart
- Possible Raspberry Pi integration
- Localization
