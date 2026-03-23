// demo_multimodal_preproc.cpp
// Build (using pkg-config):
//   g++ -std=c++17 -O2 demo_multimodal_preproc.cpp -o demo `pkg-config --cflags --libs opencv4`
// Run:
//   ./demo input.jpg

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <cmath>

// ---------- Image: normalize [0,1] + Gaussian noise ----------
cv::Mat normalizeImage(const cv::Mat &img8)
{
	cv::Mat f;
	img8.convertTo(f, CV_32F, 1.0 / 255.0);
	return f; // [0,1]
}
cv::Mat addGaussianNoise01(const cv::Mat &img01, double stddev)
{
	cv::Mat noise(img01.size(), img01.type());
	cv::randn(noise, 0, stddev);
	cv::Mat out = img01 + noise;
	cv::Mat out0;
	cv::max(out, 0.0f, out0);
	cv::Mat out1;
	cv::min(out0, 1.0f, out1);
	return out1;
}

// ---------- Text: L2 normalize embeddings + token dropout ----------
void l2_normalize(std::vector<float> &x)
{
	double s = 0;
	for (float v : x)
		s += v * v;
	s = std::sqrt(s);
	if (s)
		for (float &v : x)
			v = float(v / s);
}
std::vector<std::string> token_dropout(const std::vector<std::string> &toks, float p)
{
	std::mt19937 rng(42);
	std::bernoulli_distribution keep(1.0f - p);
	std::vector<std::string> out;
	out.reserve(toks.size());
	for (auto &t : toks)
		if (keep(rng))
			out.push_back(t);
	return out;
}

// ---------- Numerical: z-score + jitter ----------
void zscore(std::vector<double> &x)
{
	double mu = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
	double v = 0;
	for (double a : x)
		v += (a - mu) * (a - mu);
	double sd = std::sqrt(v / x.size()) + 1e-12;
	for (double &a : x)
		a = (a - mu) / sd;
}
void jitter(std::vector<double> &x, double stddev)
{
	std::mt19937 rng(42);
	std::normal_distribution<double> N(0.0, stddev);
	for (double &a : x)
		a += N(rng);
}

template <class T>
void print_vec(const char *tag, const std::vector<T> &v, int k = 8)
{
	std::cout << tag << " [";
	for (int i = 0; i < (int)v.size() && i < k; ++i)
	{
		std::cout << v[i] << (i + 1 < (int)v.size() && i + 1 < k ? ", " : "");
	}
	if ((int)v.size() > k)
		std::cout << ", ...";
	std::cout << "]\n";
}

int main(int argc, char **argv)
{
	// ---- Image demo ----
	cv::Mat img = (argc > 1 ? cv::imread(argv[1]) : cv::Mat());
	if (img.empty())
	{
		// fallback: make a simple gradient image
		img = cv::Mat(256, 256, CV_8UC3);
		for (int r = 0; r < img.rows; ++r)
			for (int c = 0; c < img.cols; ++c)
				img.at<cv::Vec3b>(r, c) = cv::Vec3b((uchar)c, (uchar)r, (uchar)((r + c) / 2));
		std::cout << "[info] input.jpg not found; generated a dummy image.\n";
	}
	cv::Mat img01 = normalizeImage(img);
	cv::Mat noisy01 = addGaussianNoise01(img01, 0.08);
	cv::Mat norm8, noisy8;
	img01.convertTo(norm8, CV_8U, 255.0);
	noisy01.convertTo(noisy8, CV_8U, 255.0);
	cv::imwrite("normalized.png", norm8);
	cv::imwrite("noisy.png", noisy8);
	std::cout << "Wrote normalized.png and noisy.png\n";

	// ---- Text demo ----
	std::vector<float> emb = {0.2f, -0.4f, 0.1f, 0.3f};
	l2_normalize(emb);
	print_vec("L2-normalized embedding", emb);
	std::vector<std::string> toks = {"the", "movie", "was", "surprisingly", "good"};
	auto kept = token_dropout(toks, 0.3f);
	std::cout << "Token dropout (p=0.3): ";
	for (auto &t : kept)
		std::cout << t << " ";
	std::cout << "\n";

	// ---- Numerical demo ----
	std::vector<double> x = {10.0, 12.0, 9.0, 13.5, 11.0};
	zscore(x);
	print_vec("z-score", x);
	jitter(x, 0.1);
	print_vec("z-score + jitter", x);

	return 0;
}
