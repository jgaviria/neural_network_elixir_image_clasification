defmodule XRayClasifier.NeuralNetworkModel do
  @moduledoc """
  Module for building the neural network model.
  """

  @doc """
  Builds the neural network model architecture.

  ## Returns
  A model built using the Axon library.

  1. This defines the input layer of the neural network. It creates an input placeholder named "features" with a shape of {nil, 784}. Here, nil indicates that the first dimension (batch size) can vary, and 784 represents the number of features (or neurons) in the input layer. Each input sample is expected to have 784 features.
  First Dense Layer:

  2. This adds a dense (fully connected) layer to the neural network with 128 neurons. Each neuron in this layer will receive input from all neurons in the previous layer (the input layer in this case). This layer applies a linear transformation to the input data followed by an activation function.
  ReLU Activation:

  3. This applies the Rectified Linear Unit (ReLU) activation function element-wise to the output of the first dense layer. ReLU is a non-linear activation function commonly used in neural networks to introduce non-linearity into the model, helping it learn complex patterns in the data.
  Second Dense Layer:

  4. This adds another dense layer to the neural network with 10 neurons. Similar to the first dense layer, each neuron in this layer will receive input from all neurons in the previous layer (the ReLU activation layer). This layer will perform another linear transformation on the input data.
  Softmax Activation:

  5. This applies the softmax activation function to the output of the second dense layer. Softmax is commonly used as the output layer activation function in classification problems. It squashes the raw predicted values into probabilities that sum up to 1, representing the likelihood of each class. The name: "labels" parameter assigns the name "labels" to this layer for identification purposes.
  """
  @spec build_model() :: Axon.Model.t()
  def build_model do
    Axon.input("features", shape: {nil, 1, 128, 128})
    |> Axon.conv(16, kernel_size: {3, 3}, strides: 1, padding: :same, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2, padding: :same)
    |> Axon.conv(32, kernel_size: {3, 3}, strides: 1, padding: :same, activation: :relu)
    |> Axon.dropout(rate: 0.1)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2, padding: :same)
    |> Axon.conv(32, kernel_size: {3, 3}, strides: 1, padding: :same, activation: :relu)
    |> Axon.dropout(rate: 0.1)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2, padding: :same)
    |> Axon.conv(32, kernel_size: {3, 3}, strides: 1, padding: :same, activation: :relu)
    |> Axon.dropout(rate: 0.2)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2, padding: :same)
    |> Axon.flatten()
    |> Axon.dense(16, activation: :relu)
    |> Axon.dropout(rate: 0.1)
    |> Axon.dense(8, activation: :relu)
    |> Axon.dense(2, activation: :softmax)
  end
end
