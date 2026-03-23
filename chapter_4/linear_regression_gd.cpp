/*
Build from the repo root:
g++ -std=c++17 -O2 chapter_4/linear_regression_gd.cpp -o chapter_4/linear_reg

Run:
./chapter_4/linear_reg
*/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

int main()
{
	std::mt19937 rng(42);
	std::uniform_real_distribution<double> U(0.0, 1.0);
	std::normal_distribution<double> N(0.0, 0.2);

	// Data: y = 2.5x + 0.7 + noise
	const int n = 200;
	std::vector<double> x(n), y(n);
	for (int i = 0; i < n; ++i)
	{
		x[i] = U(rng);
		y[i] = 2.5 * x[i] + 0.7 + N(rng);
	}

	// Params + hyperparams
	double w = 0.0, b = 0.0;
	double lr = 0.1;
	const int epochs = 1000;

	for (int epoch = 1; epoch <= epochs; ++epoch)
	{
		// Per-sample SGD with MSE gradients
		for (int i = 0; i < n; ++i)
		{
			double y_hat = w * x[i] + b;
			double resid = y_hat - y[i];
			w -= lr * (2.0 * resid * x[i]); // d(MSE)/dw
			b -= lr * (2.0 * resid);		// d(MSE)/db
		}

		if (epoch % 200 == 0 || epoch == 1)
		{
			double mse = 0.0;
			for (int i = 0; i < n; ++i)
			{
				double e = (w * x[i] + b) - y[i];
				mse += e * e;
			}
			mse /= n;
			std::cout << "Epoch " << epoch
					  << " | MSE=" << mse
					  << " | w=" << w << " | b=" << b << "\n";
		}
	}

	std::cout << "\nTrained line: y ≈ " << w << " * x + " << b << "\n";
	double xt = 0.5;
	std::cout << "x=0.5 -> y_hat=" << (w * xt + b) << "\n";
	return 0;
}
