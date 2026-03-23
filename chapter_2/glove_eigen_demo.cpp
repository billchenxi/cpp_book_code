// glove_eigen_demo.cpp
/*
% g++ -std=c++17 glove_eigen_demo.cpp -o glove_demo -I"$(brew --prefix eigen)/include/eigen3"
% ./glove_demo

Output:
Tokens: natural language processing rocks
Embedding dim = 3
Matrix shape  = 4 x 3

natural -> [0.100, 0.200, 0.300]
language -> [0.400, 0.500, 0.600]
processing -> [0.700, 0.800, 0.900]
rocks -> [0.000, 0.000, 0.000]

Sentence (mean) -> [0.300, 0.375, 0.450]
*/

#include <Eigen/Dense>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

using EmbMap = std::unordered_map<std::string, Eigen::VectorXd>;

EmbMap loadGloveStream(std::istream &is, int dim)
{
	EmbMap E;
	std::string line, word;
	while (std::getline(is, line))
	{
		if (line.empty())
			continue;
		std::istringstream ss(line);
		if (!(ss >> word))
			continue;
		Eigen::VectorXd v(dim);
		for (int i = 0; i < dim; ++i)
		{
			if (!(ss >> v[i]))
			{
				v.setZero();
				break;
			}
		}
		E.emplace(std::move(word), std::move(v));
	}
	return E;
}

EmbMap loadGloveFile(const std::string &path, int dim)
{
	std::ifstream f(path);
	if (!f)
		return {};
	return loadGloveStream(f, dim);
}

EmbMap tinyFallback()
{
	static const char *kMini =
		"natural 0.10 0.20 0.30\n"
		"language 0.40 0.50 0.60\n"
		"processing 0.70 0.80 0.90\n"
		"nlp 0.05 0.10 0.15\n";
	std::istringstream ss(kMini);
	return loadGloveStream(ss, 3);
}

Eigen::MatrixXd embedTokens(const std::vector<std::string> &toks, const EmbMap &E, int dim)
{
	Eigen::MatrixXd M(toks.size(), dim);
	M.setZero();
	for (size_t i = 0; i < toks.size(); ++i)
	{
		auto it = E.find(toks[i]);
		if (it != E.end())
			M.row(i) = it->second.transpose();
	}
	return M;
}

int main(int argc, char **argv)
{
	const std::string glovePath = (argc > 1 ? argv[1] : "");
	int dim = (argc > 2 ? std::stoi(argv[2]) : 3);

	EmbMap E = glovePath.empty() ? tinyFallback() : loadGloveFile(glovePath, dim);
	if (E.empty())
	{
		std::cerr << "[info] Falling back to embedded mini-embeddings (dim=3).\n";
		E = tinyFallback();
		dim = 3;
	}

	std::vector<std::string> tokens = {"natural", "language", "processing", "rocks"};

	Eigen::MatrixXd V = embedTokens(tokens, E, dim);

	// --- FIX: ensure consistent type (column vector). ---
	Eigen::VectorXd sent;
	if (V.rows() > 0)
		sent = V.colwise().mean().transpose(); // row->col
	else
		sent = Eigen::VectorXd::Zero(dim);

	std::cout << "Tokens: ";
	for (auto &t : tokens)
		std::cout << t << ' ';
	std::cout << "\n";
	std::cout << "Embedding dim = " << dim << "\nMatrix shape  = " << V.rows() << " x " << V.cols() << "\n\n";

	std::cout << std::fixed << std::setprecision(3);
	for (int i = 0; i < V.rows(); ++i)
	{
		std::cout << tokens[i] << " -> [";
		for (int j = 0; j < V.cols(); ++j)
			std::cout << V(i, j) << (j + 1 < V.cols() ? ", " : "");
		std::cout << "]\n";
	}
	std::cout << "\nSentence (mean) -> [";
	for (int j = 0; j < sent.size(); ++j)
		std::cout << sent[j] << (j + 1 < sent.size() ? ", " : "");
	std::cout << "]\n";
	return 0;
}