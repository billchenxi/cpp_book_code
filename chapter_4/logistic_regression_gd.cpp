/*
Build from the repo root:
g++ -std=c++17 -O2 chapter_4/logistic_regression_gd.cpp -o chapter_4/logistic_reg

Run:
./chapter_4/logistic_reg
*/

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>

static inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

int main()
{
	std::mt19937 rng(123);
	std::normal_distribution<double> N0x(-2.0, 1.0), N0y(-2.0, 1.0);
	std::normal_distribution<double> N1x(2.0, 1.0), N1y(2.0, 1.0);

	// Two Gaussian blobs
	const int n_per = 200;
	std::vector<std::array<double, 2>> X;
	std::vector<int> y;
	X.reserve(2 * n_per);
	y.reserve(2 * n_per);
	for (int i = 0; i < n_per; ++i)
	{
		X.push_back({N0x(rng), N0y(rng)});
		y.push_back(0);
		X.push_back({N1x(rng), N1y(rng)});
		y.push_back(1);
	}
	const int n = static_cast<int>(X.size());

	// Parameters (single neuron with sigmoid)
	double w1 = 0.0, w2 = 0.0, b = 0.0;
	double lr = 0.1;
	const int epochs = 3000;

	for (int epoch = 1; epoch <= epochs; ++epoch)
	{
		double gw1 = 0.0, gw2 = 0.0, gb = 0.0;
		for (int i = 0; i < n; ++i)
		{
			double z = w1 * X[i][0] + w2 * X[i][1] + b;
			double p = sigmoid(z);
			double diff = (p - y[i]); // d(BCE)/dz for label in {0,1}
			gw1 += diff * X[i][0] / n;
			gw2 += diff * X[i][1] / n;
			gb += diff / n;
		}
		w1 -= lr * gw1;
		w2 -= lr * gw2;
		b -= lr * gb;

		if (epoch % 500 == 0)
		{
			double loss = 0.0;
			for (int i = 0; i < n; ++i)
			{
				double p = sigmoid(w1 * X[i][0] + w2 * X[i][1] + b);
				p = std::clamp(p, 1e-12, 1.0 - 1e-12);
				loss += -(y[i] * std::log(p) + (1 - y[i]) * std::log(1 - p));
			}
			loss /= n;
			std::cout << "Epoch " << epoch << "  BCE=" << loss
					  << "  w=[" << w1 << "," << w2 << "]  b=" << b << "\n";
		}
	}

	int correct = 0;
	for (int i = 0; i < n; ++i)
	{
		double p = sigmoid(w1 * X[i][0] + w2 * X[i][1] + b);
		int pred = (p >= 0.5);
		if (pred == y[i])
			correct++;
	}
	std::cout << "\nAccuracy: " << (100.0 * correct / n) << "%\n";
	std::cout << "Decision boundary (p=0.5): w1*x1 + w2*x2 + b = 0\n";
	return 0;
}
