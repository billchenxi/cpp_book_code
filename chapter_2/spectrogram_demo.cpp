/*
Need to install fftw libsndfile opencv

Build: g++ -std=c++17 spectrogram_demo.cpp -o spec_demo $(pkg-config --cflags --libs fftw3 sndfile opencv4)
Run:   ./spec_demo input.wav

*/

#include <sndfile.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "usage: " << argv[0] << " input.wav\n";
		return 1;
	}

	// Load WAV (mono or first channel)
	SF_INFO inf{};
	SNDFILE *sf = sf_open(argv[1], SFM_READ, &inf);
	if (!sf)
	{
		std::cerr << "open fail\n";
		return 2;
	}
	std::vector<double> interleaved((size_t)inf.frames * inf.channels);
	sf_read_double(sf, interleaved.data(), interleaved.size());
	sf_close(sf);
	std::vector<double> x((size_t)inf.frames);
	for (sf_count_t i = 0; i < inf.frames; i++)
		x[i] = interleaved[(size_t)i * inf.channels];

	// STFT params
	const int N = 1024, H = 256; // window, hop
	const int F = std::max(1, (int)std::ceil((x.size() - N) / (double)H) + 1);
	std::vector<double> win(N);
	for (int n = 0; n < N; n++)
		win[n] = 0.5 - 0.5 * std::cos(2 * M_PI * n / N);

	// FFTW plan (real -> complex)
	auto *in = (double *)fftw_malloc(sizeof(double) * N);
	auto *X = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
	auto plan = fftw_plan_dft_r2c_1d(N, in, X, FFTW_MEASURE);

	// Spectrogram matrix: rows=freq bins, cols=frames
	cv::Mat S(N / 2 + 1, F, CV_32F);
	for (int t = 0; t < F; ++t)
	{
		int start = t * H;
		for (int n = 0; n < N; n++)
		{
			double s = (start + n < (int)x.size() ? x[start + n] : 0.0);
			in[n] = s * win[n];
		}
		fftw_execute(plan);
		for (int k = 0; k <= N / 2; k++)
		{
			double re = X[k][0], im = X[k][1];
			float mag = (float)std::sqrt(re * re + im * im) + 1e-12f;
			S.at<float>(k, t) = 20.0f * std::log10(mag);
		}
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(X);

	// Normalize -> image, flip so low freq at bottom, colorize
	double mn, mx;
	cv::minMaxLoc(S, &mn, &mx);
	cv::Mat img;
	S.convertTo(img, CV_8U, 255.0 / (mx - mn), -255.0 * mn / (mx - mn));
	cv::flip(img, img, 0);
	cv::applyColorMap(img, img, cv::COLORMAP_MAGMA);
	cv::imwrite("spectrogram.png", img);
	std::cout << "Wrote spectrogram.png (" << img.cols << "x" << img.rows << ")\n";
	return 0;
}
