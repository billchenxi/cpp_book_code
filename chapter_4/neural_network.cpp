#include "neural_network.h"

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double lr)
    : learning_rate(lr),
      weights_input_hidden(Eigen::MatrixXd::Random(hidden_size, input_size)),
      weights_hidden_output(Eigen::MatrixXd::Random(output_size, hidden_size)),
      bias_hidden(Eigen::VectorXd::Random(hidden_size)),
      bias_output(Eigen::VectorXd::Random(output_size)) {}

Eigen::VectorXd NeuralNetwork::feedforward(const Eigen::VectorXd& input) const {
    Eigen::VectorXd hidden_input = weights_input_hidden * input + bias_hidden;
    Eigen::VectorXd hidden_output = sigmoid(hidden_input);

    Eigen::VectorXd final_input = weights_hidden_output * hidden_output + bias_output;
    Eigen::VectorXd final_output = sigmoid(final_input);
    return final_output;
}

void NeuralNetwork::train(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
    Eigen::VectorXd hidden_input = weights_input_hidden * input + bias_hidden;
    Eigen::VectorXd hidden_output = sigmoid(hidden_input);

    Eigen::VectorXd final_input = weights_hidden_output * hidden_output + bias_output;
    Eigen::VectorXd final_output = sigmoid(final_input);

    Eigen::VectorXd output_error = target - final_output;
    Eigen::VectorXd output_delta = sigmoid_derivative(final_output).cwiseProduct(output_error);

    Eigen::VectorXd hidden_error = weights_hidden_output.transpose() * output_delta;
    Eigen::VectorXd hidden_delta = sigmoid_derivative(hidden_output).cwiseProduct(hidden_error);

    weights_hidden_output += learning_rate * (output_delta * hidden_output.transpose());
    bias_output += learning_rate * output_delta;

    weights_input_hidden += learning_rate * (hidden_delta * input.transpose());
    bias_hidden += learning_rate * hidden_delta;
}
