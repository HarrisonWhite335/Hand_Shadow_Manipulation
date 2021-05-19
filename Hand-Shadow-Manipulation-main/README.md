
# American Sign Language Detection using PyTorch

This project will allow you to recognize American Sign Language letters from static images and real time video. American Sign Language data is obtained from Kaggle. In order to
train your model you should download the dataset from https://www.kaggle.com/grassknoted/asl-alphabet.

Dataset consists of 26 letters from American Sign Language and space, delete and nothing classes. There is a total of 87,000 training images - 3,000 images from each class. The
image resolution is 200x200 pixels.

In order to run the program, please download all python files in this directory. Place dataset from Kaggle and python files inside the same input directory. Please consult 
to the file tree .png's for further questions on the directories. 

The model is already trained in our computer and corresponding model (model.pth), binarized labels (lb.pkl) and data file (data.csv) can be found in the github repository. If you want to use already trained model you can skip to Step 5.

The project uses pytorch, imutils, albumentations, opencv, pandas, numpy and matplotlib libraries. In order to run the codes aforementioned libraries should be installed.
- To install Pytorch: conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch (Refer to https://pytorch.org/ for MacOS and Linux)
- To install imultis: pip install imutils (https://pypi.org/project/imutils/)
- To install opencv: pip install opencv-python (https://pypi.org/project/opencv-python/)
- To install numpy: pip install numpy (https://numpy.org/install/)
- To install pandas: pip install pandas (https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- To install matplotlib: pip install matplotlib (https://pypi.org/project/matplotlib/)


___IF YOU WANT TO TRAIN ON YOUR COMPUTER___

1. Run preprocess_image.py to read, resize and save images.
Can be also run from the terminal python preprocess_image.py --num-images 1200
You can change the number of images that will be selected for the model by changing the last number.

2. Run create_csv.py file to convert images to map the image paths to the target classes.
First column corresponds to the image_path and second column corresponds to the label between 0-28 associated with each image.
This code will create data.csv.

3. Run cnn_models.py
Creates convolutional neural network with four 2D convolutional layers.

4. Run train.py to train the data
Train the dataset and then validate using 15% of the data. This step will give you model.pth, accuracy and loss plots.
You can run it from the terminal by python train.py --epochs 10
You can change the last number to train with less epochs. This will decrease the training time.

___IF YOU WANT TO USE ALREADY TRAINED MODEL___

5. Run test.py to test static hand images.
You can run the test.py from terminal using the following command: python test.py --img A_test.jpg
Change img A_test.jpg with any static image that you would like to test. 
Make sure that static hand image is located in the correct input directory.

6. Run cam_test_original.py to test American Sign Language letter in real time.
Original algorithm gives you corresponding 26 letter and 3 additional classes based on your real time hand input. Check test folder from Kaggle to make sure you are making the  
correct hand sign for the letters. This step requires a knowledge of American Sign Language. Your hand should be placed within the red square in the cameraview.

7. Run cam_test_modified.py to test emojis (Longhorn, Thumbs Up, Thumbs Down, I Love You, OK and Peace) trained using the same dataset.
Test the code by doing the aferomentioned emojis. 
I love you sign is a special type of hand language. Please check the wiki article to understand this sign: https://en.wikipedia.org/wiki/ILY_sign

NOTE: Heavily inspired from https://debuggercafe.com/american-sign-language-detection-using-deep-learning/ and edited. No licensing could be found on page or website.
