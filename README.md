# NeuralNetworkElixir
## Description
There are 2 neural networks implemented 100% percent in Elixir, Axon and Nx. The first one implements a deep learning model capable of recognizing handwritten numbers ranging from 0 to 9. To achieve this, we will acquire a curated image dataset consisting of 60,000 images from Scidata, each associated with its respective label, to serve as training data. Subsequently, we will utilize a separate set of 10 thousand images for testing the trained model's performance.

The Second one, will train a model for recognizing pneumonia from an x-ray image. We will consider only two possible target labels,  label "1" images showing pneumonia while labbel "0" labels images without pneumonia. The data set was obtained from Keggle and it consists of 5216 images for training and 624 for testing.

## Usage
To start the application and trigger the training and evaluation of the model, follow these steps:

**Clone the Repository:** Clone the repository to your local machine.

**Install Dependencies:** Ensure that you have Elixir installed on your machine. Then, navigate to the project directory and install the dependencies by running:

```elixir
mix deps.get
```
**Start the Application:** Start the application by running:


```elixir
iex -S mix
# Runs the digit detector 
Main.start_digit_detector(epochs: 5)

# Runs the x-ray classifier
Main.start_xray_detector(epochs: 5)
```
or 

```elixir
mix run -e "Main.start_digit_detector(epochs: 5)"

mix run -e "Main.start_xray_detector(epochs: 5)"
```

This will initiate the training and evaluation process of the neural network model using the MNIST dataset.

## Digit Detector Overview
The application consists of several modules:

- **Main:** The main module responsible for starting the application and triggering the training and evaluation process.

- **TrainingAndEvaluation:** Module for training and evaluating machine learning models.

- **TrainTensor:** Module for preprocessing training data.

- **TestTensor:** Module for preprocessing test data.

- **NeuralNetworkModel:** Module for building the neural network model architecture.

## The Training Set

The first tuple contains information about the test images. It consists of three elements:

- The binary data representing the images.
- The type of data, which is :u indicating unsigned integers with a size of 8 bits.
- The shape {60000, 1, 28, 28} , 60_000 thousand images, 1 channel (gray scale) and each image is 28 by 28 pixels.

The second tuple contains information about the corresponding labels for the test images. It also consists of three elements:
- The binary data representing the labels.
- The type of data, which is :u indicating unsigned integers with a size of 8 bits.
- There is no shape provided for the labels, but there are 60,000 labels corresponding to the 10,000 test images.

```elixir
{images, labels} = Scidata.MNIST.download()
```

```elixir
{{<<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...>>,
{:u, 8}, {60000, 1, 28, 28}}
{<<5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4, 0, 9, 1, 1,
2, 4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8, 7, 9, 3, 9, 8, ...>>,
```

## The Test Set

The first tuple contains information about the test images. It consists of three elements:

- The binary data representing the images.
- The type of data, which is :u indicating unsigned integers with a size of 8 bits.
- The shape of the data, which is {10000, 1, 28, 28} indicating there are 10,000 images, each with dimensions of 28x28 pixels.
- 
The second tuple contains information about the corresponding labels for the test images. It also consists of three elements:
- The binary data representing the labels.
- The type of data, which is :u indicating unsigned integers with a size of 8 bits.
- There is no shape provided for the labels, but there are 10,000 labels corresponding to the 10,000 test images.

```elixir
{test_images, test_labels} = Scidata.MNIST.download_test()
```

```elixir
{{<<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...>>,
{:u, 8}, {10000, 1, 28, 28}},
{<<7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5, 4,
0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 1, ...>>,
```

To see what you are working with, you can run the following to get a heatmap representing an image:

```elixir
{images, labels} = Scidata.MNIST.download()
{images_data, images_type, images_shape} = images
 
images_data
|> Nx.from_binary(images_type)
|> Nx.reshape(images_shape)
|> then(& &1[[0]])
|> Nx.to_heatmap()
```

## X-ray Detector Overview
todo