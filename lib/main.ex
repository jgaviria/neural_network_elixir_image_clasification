defmodule Main do
  @moduledoc """
  Main module to start the application.
  """

  @doc """
  Starts the digit detector application by triggering training and evaluation of the model.

  ## Parameters
    * `epochs` - The number of epochs to train the model for.

  ## Returns
    :ok if successful.
  """
  @spec start_digit_detector(epochs :: integer()) :: :ok
  def start_digit_detector(epochs: epochs) when is_integer(epochs) and epochs > 0 do
    DigitDetector.TrainingAndEvaluation.train_and_evaluate_model(epochs: epochs)
    :ok
  end

  @doc """
  Starts the X-ray detector application by triggering training and evaluation of the model.

  ## Parameters
    * `epochs` - The number of epochs to train the model for.

  ## Returns
    :ok if successful.
  """
  @spec start_xray_detector(epochs :: integer()) :: :ok
  def start_xray_detector(epochs: epochs) when is_integer(epochs) and epochs > 0 do
    XRayClasifier.TrainingAndEvaluation.train_and_evaluate_model(epochs: epochs)
    :ok
  end
end
