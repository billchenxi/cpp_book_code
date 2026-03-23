// image_preproc_demo.cpp
/*
install opencv pkg-config
Build: % g++ -std=c++17 -O2 image_preproc_demo.cpp -o image_preproc_demo $(pkg-config --cflags --libs opencv4)
Run:   % ./image_preproc_demo input.jpg
*/

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

static Mat centerCrop(const Mat &img, int w, int h)
{
	int x = std::max(0, (img.cols - w) / 2), y = std::max(0, (img.rows - h) / 2);
	Rect r(x, y, std::min(w, img.cols - x), std::min(h, img.rows - y));
	return img(r).clone();
}
static Mat rotateDegrees(const Mat &img, double deg)
{
	Point2f c(img.cols / 2.f, img.rows / 2.f);
	Mat M = getRotationMatrix2D(c, deg, 1.0);
	Mat out;
	warpAffine(img, out, M, img.size(), INTER_LINEAR, BORDER_REFLECT101);
	return out;
}
static Mat slightPerspective(const Mat &img)
{
	Point2f s[4] = {{0, 0}, {(float)img.cols - 1, 0}, {(float)img.cols - 1, (float)img.rows - 1}, {0, (float)img.rows - 1}};
	float dx = 0.12f * img.cols, dy = 0.08f * img.rows;
	Point2f d[4] = {{dx, dy}, {(float)img.cols - 1 - dx, 0}, {(float)img.cols - 1, (float)img.rows - 1}, {0, (float)img.rows - 1}};
	Mat H = getPerspectiveTransform(s, d), out;
	warpPerspective(img, out, H, img.size(), INTER_LINEAR, BORDER_REFLECT101);
	return out;
}
static Mat equalizeColorY(const Mat &bgr)
{ // Histogram equalization (luminance only)
	Mat ycrcb;
	cvtColor(bgr, ycrcb, COLOR_BGR2YCrCb);
	std::vector<Mat> ch;
	split(ycrcb, ch);
	equalizeHist(ch[0], ch[0]);
	merge(ch, ycrcb);
	Mat out;
	cvtColor(ycrcb, out, COLOR_YCrCb2BGR);
	return out;
}
static Mat contrastGainBias(const Mat &img, double alpha = 1.2, double beta = 10.0)
{
	Mat out;
	img.convertTo(out, -1, alpha, beta);
	return out; // I' = alpha*I + beta
}
static Mat sobelMag8U(const Mat &gray)
{
	Mat gx, gy;
	Sobel(gray, gx, CV_32F, 1, 0, 3);
	Sobel(gray, gy, CV_32F, 0, 1, 3);
	Mat mag;
	magnitude(gx, gy, mag);
	normalize(mag, mag, 0, 255, NORM_MINMAX);
	Mat u8;
	mag.convertTo(u8, CV_8U);
	return u8;
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " input.jpg\n";
		return 1;
	}
	Mat img = imread(argv[1]);
	if (img.empty())
	{
		std::cerr << "Failed to read " << argv[1] << "\n";
		return 2;
	}

	Mat resized, cropped, flipped, rotated, persp, eq_gray, eq_color, contrast, sobel, canny;
	resize(img, resized, Size(256, 256));
	cropped = centerCrop(resized, 224, 224);
	flip(cropped, flipped, 1);
	rotated = rotateDegrees(cropped, 15.0);
	persp = slightPerspective(cropped);

	// Histogram equalization: grayscale & luminance-preserving color
	Mat gray;
	cvtColor(cropped, gray, COLOR_BGR2GRAY);
	equalizeHist(gray, eq_gray);
	eq_color = equalizeColorY(cropped);

	// Contrast tweak (gain/bias)
	contrast = contrastGainBias(cropped, 1.2, 10.0);

	// Edges: Sobel magnitude and Canny
	sobel = sobelMag8U(gray);
	Canny(gray, canny, 100, 200);

	// Write results
	imwrite("out_resized.jpg", resized);
	imwrite("out_cropped.jpg", cropped);
	imwrite("out_flipped.jpg", flipped);
	imwrite("out_rotated.jpg", rotated);
	imwrite("out_perspective.jpg", persp);
	imwrite("out_eq_gray.jpg", eq_gray);
	imwrite("out_eq_color.jpg", eq_color);
	imwrite("out_contrast.jpg", contrast);
	imwrite("out_sobel.jpg", sobel);
	imwrite("out_canny.jpg", canny);

	std::cout << "Wrote: out_*.jpg\n";
	return 0;
}
