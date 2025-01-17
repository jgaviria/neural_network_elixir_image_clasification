defmodule NeuralNetworkElixir1.MixProject do
  use Mix.Project

  def project do
    [
      app: :neural_network_elixir_1,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.6"},
      {:exla, "~> 0.6"},
      {:nx, "~> 0.7.1"},
      {:scidata, "~> 0.1.3"},
      {:image, "~> 0.37"}
    ]
  end
end
