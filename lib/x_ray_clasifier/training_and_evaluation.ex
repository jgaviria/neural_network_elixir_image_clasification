defmodule XRayClasifier.TrainingAndEvaluation do

  alias XRayClasifier.{TrainTensor, TestTensor, NeuralNetworkModel}

  import Axon.Loop

  @doc """
  Train and evaluate the machine learning model using MNIST dataset.

  ## Examples

      TrainingAndEvaluation.train_and_evaluate_model(epochs: 5)
  """
  @spec train_and_evaluate_model(epochs: integer()) :: any()
  def train_and_evaluate_model(epochs: epochs) do
    {images_tensor, labels_tensor} = TrainTensor.preprocess_data()
    {test_images_tensor, test_labels_tensor} = TestTensor.preprocess_data()

    x = Nx.to_batched(images_tensor, 32)
    y = Nx.to_batched(labels_tensor, 32)

    train_data = Stream.zip(x, y)

    model = NeuralNetworkModel.build_model()

    trained_model_state =
      model
      |> trainer(:categorical_cross_entropy, :adam)
      |> run(train_data, %{}, compiler: EXLA, epochs: epochs)

    predict = Axon.predict(model, trained_model_state, test_images_tensor, compiler: EXLA)

    result =
      predict
      |> Nx.argmax(axis: 1)
      |> Nx.reshape({624, 1})
      |> Nx.equal(Nx.tensor(Enum.to_list(0..1)))

    IO.inspect Axon.Metrics.accuracy(test_labels_tensor, result)

  end
end
