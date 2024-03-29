defmodule TrainingAndEvaluation do
  @moduledoc """
  Module for training and evaluating machine learning models.
  """

  import Axon.Loop

  @doc """
  Train and evaluate the machine learning model using MNIST dataset.

  ## Examples

      TrainingAndEvaluation.train_and_evaluate_model()
  """
  @spec train_and_evaluate_model() :: any()
  def train_and_evaluate_model do

    {images_tensor, labels_tensor} = TrainTensor.preprocess_data()
    {test_images_tensor, test_labels_tensor} = TestTensor.preprocess_data()

    train_data = Nx.to_batched(images_tensor, 32)
                 |> Stream.zip(Nx.to_batched(labels_tensor, 32))

    test_data = Nx.to_batched(test_images_tensor, 32)
                |> Stream.zip(Nx.to_batched(test_labels_tensor, 32))

    train_data = Enum.map(train_data, fn {images, labels} -> {images, Nx.equal(labels, Nx.iota({10}))} end)

    model = NeuralNetworkModel.build_model()

    trained_model_state =
      model
      |> trainer(:categorical_cross_entropy, :adam)
      |> run(train_data, %{}, compiler: EXLA, epochs: 5)

    model
    |> evaluator()
    |> metric(:accuracy)
    |> run(test_data, trained_model_state, compiler: EXLA)
  end

end
