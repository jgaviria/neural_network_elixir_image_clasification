defmodule DigitDetector.TrainTensor do
  @moduledoc """
  Module for preprocessing training data.
  """

  @doc """
  Preprocesses the training data.

  ## Returns
  A tuple containing the preprocessed images and labels tensors.

  ## HeatMap rep of single image
  images_data
  |> Nx.from_binary(images_type)
  |> Nx.reshape(images_shape)
  |> then(& &1[[0]])
  |> Nx.to_heatmap()
  """
  def preprocess_data() do
    # Download the MNIST training data
    {images, labels} = Scidata.MNIST.download()

    # Extract data, type, and shape information for images and labels
    {images_data, images_type, images_shape} = images
    {labels_data, labels_type, labels_shape} = labels

    # Convert binary image data into a normalized tensor
    images_tensor =
      images_data
      |> Nx.from_binary(images_type)
      |> Nx.reshape(images_shape)
      |> Nx.divide(255)
      |> Nx.reshape({60_000, :auto})

    # Add a new axis at the end of labels tensor
    labels_tensor =
      labels_data
      |> Nx.from_binary(labels_type)
      |> Nx.reshape(labels_shape)
      |> Nx.new_axis(-1)

    {images_tensor, labels_tensor}
  end
end
