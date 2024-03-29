defmodule Main do
  @moduledoc """
  Main module to start the application.
  """

  @doc """
  Starts the application by triggering training and evaluation of the model.

  ## Returns
  :ok if successful.
  """
  @spec start() :: :ok
  def start do
    TrainingAndEvaluation.train_and_evaluate_model()
    :ok
  end
end
