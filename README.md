# NeuralNetworkElixir1

**TODO: Add description**

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `neural_network_elixir_1` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:neural_network_elixir_1, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/neural_network_elixir_1](https://hexdocs.pm/neural_network_elixir_1).

##Usage
To start the application and trigger the training and evaluation of the model, follow these steps:

**Clone the Repository:** Clone the repository to your local machine.

**Install Dependencies:** Ensure that you have Elixir installed on your machine. Then, navigate to the project directory and install the dependencies by running:

```elixir
mix deps.get
```
**Start the Application:** Start the application by running:


```elixir
iex -S mix
Main.start(epochs: 5)
```
or 

```elixir
mix run -e "Main.start(epochs: 5)"
```

This will initiate the training and evaluation process of the neural network model using the MNIST dataset.

##Application Overview
The application consists of several modules:

- **Main:** The main module responsible for starting the application and triggering the training and evaluation process.

- **TrainingAndEvaluation:** Module for training and evaluating machine learning models.

- **TrainTensor:** Module for preprocessing training data.

- **TestTensor:** Module for preprocessing test data.

- **NeuralNetworkModel:** Module for building the neural network model architecture.
