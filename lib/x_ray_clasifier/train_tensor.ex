defmodule XRayClasifier.TrainTensor do
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
    {:ok, train_images_bin} = File.read("lib/x_ray_clasifier/data/x-ray-train-ubyte")
    <<_ :: 32, n_images :: 32, n_rows :: 32, n_cols :: 32, images_data :: binary>> = train_images_bin

    {:ok, train_labels_bin} = File.read("lib/x_ray_clasifier/data/x-ray-train-labels-ubyte")
    <<_ :: 32, n_labels :: 32, labels_data :: binary>> = train_labels_bin

    images_tensor =
      images_data
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, 1, n_rows, n_cols})
      |> Nx.divide(255)


    # Add a new axis at the end of labels tensor
    labels_tensor =
      labels_data
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_labels, 1})
      |> Nx.equal(Nx.tensor(Enum.to_list(0..1)))

    {images_tensor, labels_tensor}
  end
end
