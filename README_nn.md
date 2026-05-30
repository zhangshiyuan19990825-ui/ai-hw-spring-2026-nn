# AI Homework 4.1 - MNIST Recognition by Neural Networks

This project is for Assignment #4.1: MNIST Recognition by NN.

The goal of this assignment is to build and test small neural network models for handwritten digit recognition using the MNIST dataset.

MNIST stands for Modified National Institute of Standards and Technology database. It contains handwritten digit images from 0 to 9. Each image is a 28 by 28 grayscale image.

## Models

I implemented three model options in `src/models.py`:

* Shallow MLP
* CNN
* Transformer Encoder

The MLP is the simplest model. It flattens the image into a vector and then uses fully connected layers for classification.

The CNN model uses convolution and pooling layers to extract local image features, such as edges, strokes, and digit shapes. Then it uses fully connected layers to classify the digit.

The Transformer Encoder model treats the image more like a sequence and uses attention to learn relationships between different parts of the image.

## Files

src/models.py          model definitions
src/train.py           train the selected model
src/test.py            test trained models
src/augment_test.py    test the CNN model with simple image transformations
results/               testing results
models/                saved trained models

## How to Run

Install dependencies:

python3 -m pip install -r requirements.txt


Train the models:

python3 src/train.py --model mlp --epochs 3
python3 src/train.py --model cnn --epochs 3
python3 src/train.py --model transformer --epochs 3


Test the models:

python3 src/test.py --models mlp cnn transformer

Run augmentation testing on CNN:

python3 src/augment_test.py --model cnn

## Results

The models were trained on the MNIST training set and tested on the MNIST test set.

| Model               | Test Accuracy |
| ------------------- | ------------: |
| MLP                 |        96.84% |
| CNN                 |        98.80% |
| Transformer Encoder |        96.07% |

The CNN model had the best accuracy in my experiment, so I used it as the main model for the augmentation test.

## Augmentation Test Results

| Test Type          | Accuracy |
| ------------------ | -------: |
| Clean images       |   98.80% |
| Rotate +10 degrees |   97.94% |
| Rotate -10 degrees |   97.23% |
| Translate 2 pixels |   95.12% |

## What I Learned

From this experiment, I learned that CNN works very well for MNIST image recognition. This makes sense because CNN is designed for image tasks and can learn local visual features.

The MLP also worked, but it loses the original image structure after flattening the image. The Transformer Encoder also worked, but for a small dataset like MNIST, it was more complex and did not perform better than CNN in my result.

The augmentation test also showed that the CNN is still strong under small rotations, but the accuracy drops more when the digit is shifted. This means the model is a little more sensitive to changes in position.
