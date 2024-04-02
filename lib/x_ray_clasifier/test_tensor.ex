defmodule XRayClasifier.TestTensor do
  @moduledoc """
  Module for preprocessing test tensors.
  """

  @doc """
  Preprocesses the test images and labels data.

  ## Returns
  A tuple containing preprocessed test images and labels tensors.
  """
  def preprocess_data() do
    {:ok, test_images_bin} = File.read("lib/x_ray_clasifier/data/x-ray-test-ubyte")
    <<
      _ :: 32,
      n_test_images :: 32,
      test_images_rows :: 32,
      test_images_cols :: 32,
      test_images :: binary
    >> = test_images_bin

    {:ok, test_labels_bin} = File.read("lib/x_ray_clasifier/data/x-ray-test-labels-ubyte")
    <<_ :: 32, n_test_labels :: 32, test_data :: binary>> = test_labels_bin

    # Convert binary test image data into a normalized tensor
    test_images_tensor =
      test_images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_test_images, 1, test_images_rows, test_images_cols})
      |> Nx.divide(255)

    # Add a new axis at the end of test labels tensor
    test_labels_tensor =
      test_data
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_test_labels, 1})
      |> Nx.equal(Nx.tensor(Enum.to_list(0..1)))

    {test_images_tensor, test_labels_tensor}
  end
end
