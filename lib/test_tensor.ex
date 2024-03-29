defmodule TestTensor do
  @moduledoc """
  Module for preprocessing test tensors.
  """

  @doc """
  Preprocesses the test images and labels data.

  ## Returns
  A tuple containing preprocessed test images and labels tensors.
  """
  def preprocess_data() do
    # Download the MNIST test data
    {test_images, test_labels} = Scidata.MNIST.download_test()

    # Extract data, type, and shape information for test images and labels
    {test_images_data, test_images_type, test_images_shape} = test_images
    {test_labels_data, test_labels_type, test_labels_shape} = test_labels

    # Convert binary test image data into a normalized tensor
    test_images_tensor =
      test_images_data
      |> Nx.from_binary(test_images_type)
      |> Nx.reshape(test_images_shape)
      |> Nx.divide(255)
      |> Nx.reshape({10_000, :auto})

    # Add a new axis at the end of test labels tensor
    test_labels_tensor =
      test_labels_data
      |> Nx.from_binary(test_labels_type)
      |> Nx.reshape(test_labels_shape)
      |> Nx.new_axis(-1)
      |> Nx.equal(Nx.iota({10}))

    {test_images_tensor, test_labels_tensor}
  end
end
