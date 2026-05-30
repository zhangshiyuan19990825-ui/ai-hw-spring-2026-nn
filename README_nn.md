# AI Homework 4.1: MNIST Recognition by Neural Networks

## Project Overview

This project is for Assignment 4.1, MNIST Recognition by Neural Networks. The goal of this assignment is to build and evaluate neural network models for handwritten digit recognition.

The dataset used in this project is MNIST. MNIST stands for Modified National Institute of Standards and Technology database. It contains grayscale handwritten digit images from 0 to 9. Each image has a size of 28 by 28 pixels.

The models were trained on the MNIST training set and evaluated on the MNIST test set.

The main files in this project are:

| File                  | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| `src/models.py`       | Defines the neural network model architectures                              |
| `src/train.py`        | Trains the selected model on the MNIST training set                         |
| `src/test.py`         | Evaluates trained models on the MNIST test set                              |
| `src/augment_test.py` | Tests the CNN model using simple image transformations                      |
| `results/`            | Stores training results, testing results, augmentation results, and weights |

## Models Implemented

Three model options were implemented in this project: a shallow MLP, a CNN, and a Transformer Encoder.

The MLP model is the simplest model. It flattens the 28 by 28 image into a one-dimensional vector and uses fully connected layers for classification. This model is easy to understand, but it loses the original spatial structure of the image.

The CNN model stands for Convolutional Neural Network. It is well suited for image recognition because it uses convolution and pooling layers to learn local visual features, such as edges, strokes, and digit shapes. After feature extraction, the model uses fully connected layers to classify the image into one of 10 digit classes.

The Transformer Encoder model was also implemented as another option. It treats the image more like a sequence and uses attention to learn relationships between different parts of the image. Transformer-based models are powerful, but for a small 28 by 28 image dataset such as MNIST, the model is more complex and did not outperform the CNN in this experiment.

## Epoch Explanation

One epoch means one complete pass through the entire training dataset. In this experiment, each model was trained for 3 epochs, so each model processed the full MNIST training set three times.

The results are different across epochs because the model weights are updated after each epoch. At the beginning of training, the weights are initialized automatically by PyTorch. During training, the model makes predictions, calculates the loss, and updates the weights through backpropagation and the Adam optimizer. As the weights improve, the test accuracy usually increases.

## Model Training Results

The table below shows the test accuracy after each epoch.

| Model               | Epoch 1 Accuracy | Epoch 2 Accuracy | Epoch 3 Accuracy | Best Accuracy |
| ------------------- | ---------------: | ---------------: | ---------------: | ------------: |
| MLP                 |           94.69% |           96.17% |           96.84% |        96.84% |
| CNN                 |           97.79% |           98.69% |           98.80% |        98.80% |
| Transformer Encoder |           92.72% |           94.85% |           96.07% |        96.07% |

The final model comparison is shown below.

| Model               | Final Test Accuracy |
| ------------------- | ------------------: |
| MLP                 |              96.84% |
| CNN                 |              98.80% |
| Transformer Encoder |              96.07% |

Based on these results, the CNN model performed the best. The CNN reached 98.80% test accuracy after 3 epochs. This result is reasonable because MNIST is an image recognition task, and CNNs are designed to learn local image patterns.

## Weights Explanation

In a neural network, weights are the learnable parameters of the model. These weights were not manually assigned. They were initialized automatically by PyTorch and updated during training.

During training, the model compared its predictions with the correct labels, calculated the loss, and used backpropagation to compute how the weights should change. The Adam optimizer then updated the weights.

For the CNN model, the weights in the convolution layers work like filters. These filters help the model learn visual patterns such as edges, strokes, and digit shapes. The weights in the fully connected layers use these learned features to classify the image into one of the 10 digit classes.

The trained weight files are saved in the `results/weights/` folder. These files store the learned parameters after training, so the trained models can be loaded and tested again without retraining from the beginning.

| Weight File      | Description                                       |
| ---------------- | ------------------------------------------------- |
| `mlp.pt`         | Trained weights for the MLP model                 |
| `cnn.pt`         | Trained weights for the CNN model                 |
| `transformer.pt` | Trained weights for the Transformer Encoder model |

## Augmentation Test

Data augmentation means applying small transformations to the input images, such as rotation or translation. In this project, augmentation was used to test how stable the CNN model is when the input images are slightly changed.

Since the CNN model had the best result, it was selected for the augmentation test.

| Test Type          | Accuracy |
| ------------------ | -------: |
| Clean Images       |   98.80% |
| Rotate +10 Degrees |   97.94% |
| Rotate -10 Degrees |   97.23% |
| Translate 2 Pixels |   95.12% |

The results show that the CNN performed very well on clean images and remained strong under small rotations. However, when the image was translated by 2 pixels, the accuracy dropped more noticeably. This suggests that the model is more sensitive to changes in digit position than to small rotations.

## Assignment 4.1 Conclusion

The experiment shows that CNN was the strongest model for MNIST recognition among the three tested models. The MLP was simple and efficient but lost spatial information after flattening the image. The Transformer Encoder also worked, but it was more complex and did not perform better than the CNN for this small image dataset.

Overall, the CNN provided the best balance of accuracy, training efficiency, and interpretability for this MNIST recognition task.
