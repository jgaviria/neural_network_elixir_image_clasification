defmodule NeuralNetworkElixir1.Application do
  use Application

  def start(_type, _args) do
    children = [
      # Start the EXLA supervisor
      EXLA.Supervisor
    ]

    opts = [strategy: :one_for_one, name: NeuralNetworkElixir1.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
