defmodule Main do
  @moduledoc """
  Main module to start the application.
  """
  alias DigitDetector.TrainingAndEvaluation

  @doc """
  Starts the application by triggering training and evaluation of the model.

  ## Returns
  :ok if successful.
  """
  @spec start_digit_detector(epochs :: integer()) :: :ok
  def start_digit_detector(epochs: epochs) do
    TrainingAndEvaluation.train_and_evaluate_model(epochs: epochs)
    :ok
  end
end
