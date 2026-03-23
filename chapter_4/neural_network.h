#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Eigen/Dense>

class NeuralNetwork {
private:
    Eigen::MatrixXd weights_input_hidden;   // [hidden, input]
    Eigen::MatrixXd weights_hidden_output;  // [output, hidden]
    Eigen::VectorXd bias_hidden;            // [hidden]
    Eigen::VectorXd bias_output;            // [output]
    double learning_rate;

    Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) const {
        return 1.0 / (1.0 + (-x.array()).exp());
    }

    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& y) const {
        return y.array() * (1.0 - y.array());
    }

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, double lr);

    Eigen::VectorXd feedforward(const Eigen::VectorXd& input) const;
    void train(const Eigen::VectorXd& input, const Eigen::VectorXd& target);
};

#endif
