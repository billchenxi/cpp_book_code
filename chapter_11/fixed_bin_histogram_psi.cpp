#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Build from repo root:
// g++ -std=c++17 -O2 -Wall -Wextra -pedantic chapter_11/fixed_bin_histogram_psi.cpp -o chapter_11/fixed_bin_histogram_psi
// ./chapter_11/fixed_bin_histogram_psi

	// Fixed-bin histogram for a bounded numeric feature.
	// Values below min_val go to the first bin.
	// Values above max_val go to the last bin.
	struct Histogram
{
	double min_val;
	double max_val;
	std::vector<std::uint64_t> bins;

	Histogram(double min_v, double max_v, std::size_t n_bins)
		: min_val(min_v), max_val(max_v), bins(n_bins, 0)
	{
		if (n_bins == 0)
		{
			throw std::invalid_argument("n_bins must be > 0");
		}
		if (max_val <= min_val)
		{
			throw std::invalid_argument("max_val must be greater than min_val");
		}
	}
};

// Return the bin index for a value.
std::size_t bin_index(const Histogram &h, double value)
{
	if (value <= h.min_val)
		return 0;
	if (value >= h.max_val)
		return h.bins.size() - 1;

	const double width =
		(h.max_val - h.min_val) / static_cast<double>(h.bins.size());

	std::size_t idx = static_cast<std::size_t>((value - h.min_val) / width);

	// Guard against floating-point edge cases.
	if (idx >= h.bins.size())
	{
		idx = h.bins.size() - 1;
	}
	return idx;
}

// Update the histogram with one observed value.
void update_histogram(Histogram &h, double value)
{
	const std::size_t idx = bin_index(h, value);
	h.bins[idx]++;
}

// Convert raw counts to proportions.
// epsilon avoids zero values in PSI calculation.
std::vector<double> to_proportions(const Histogram &h, double epsilon = 1e-12)
{
	const std::uint64_t total =
		std::accumulate(h.bins.begin(), h.bins.end(), std::uint64_t{0});

	std::vector<double> proportions(h.bins.size(), 0.0);

	if (total == 0)
	{
		const double uniform = 1.0 / static_cast<double>(h.bins.size());
		std::fill(proportions.begin(), proportions.end(), uniform);
		return proportions;
	}

	for (std::size_t i = 0; i < h.bins.size(); ++i)
	{
		proportions[i] =
			std::max(static_cast<double>(h.bins[i]) / static_cast<double>(total),
					 epsilon);
	}

	// Renormalize after epsilon floor.
	const double sum =
		std::accumulate(proportions.begin(), proportions.end(), 0.0);
	for (double &p : proportions)
	{
		p /= sum;
	}

	return proportions;
}

// Population Stability Index (PSI)
// ref = reference window proportions
// cur = current/live window proportions
double compute_psi(const std::vector<double> &ref,
				   const std::vector<double> &cur)
{
	if (ref.size() != cur.size())
	{
		throw std::invalid_argument("ref and cur must have the same size");
	}

	double psi = 0.0;
	for (std::size_t i = 0; i < ref.size(); ++i)
	{
		psi += (cur[i] - ref[i]) * std::log(cur[i] / ref[i]);
	}
	return psi;
}

// Convenience overload: compute PSI directly from histograms.
double compute_psi(const Histogram &reference,
				   const Histogram &current,
				   double epsilon = 1e-12)
{
	if (reference.bins.size() != current.bins.size())
	{
		throw std::invalid_argument("Histogram bin counts must match");
	}
	if (reference.min_val != current.min_val ||
		reference.max_val != current.max_val)
	{
		throw std::invalid_argument("Histogram ranges must match");
	}

	const std::vector<double> ref = to_proportions(reference, epsilon);
	const std::vector<double> cur = to_proportions(current, epsilon);
	return compute_psi(ref, cur);
}

void print_histogram(const Histogram &h, const std::string &name)
{
	const double width =
		(h.max_val - h.min_val) / static_cast<double>(h.bins.size());

	std::cout << "\n"
			  << name << "\n";
	for (std::size_t i = 0; i < h.bins.size(); ++i)
	{
		const double left = h.min_val + static_cast<double>(i) * width;
		const double right = left + width;
		std::cout << "  [" << std::fixed << std::setprecision(1) << left
				  << ", " << right << "): " << h.bins[i] << "\n";
	}
}

void print_proportions(const std::vector<double> &p, const std::string &name)
{
	std::cout << "\n"
			  << name << " proportions\n";
	for (std::size_t i = 0; i < p.size(); ++i)
	{
		std::cout << "  bin " << i << ": "
				  << std::fixed << std::setprecision(4) << p[i] << "\n";
	}
}

std::string psi_interpretation(double psi)
{
	if (psi < 0.1)
		return "little or no drift";
	if (psi < 0.25)
		return "moderate drift";
	return "significant drift";
}

int main()
{
	try
	{
		// Example: compare a healthy reference window with a live window.
		Histogram reference(0.0, 100.0, 10);
		Histogram current(0.0, 100.0, 10);

		// Example reference data: centered in lower-middle bins.
		const std::vector<double> reference_data = {
			10, 12, 15, 18, 20, 22, 25, 28, 30, 32,
			35, 38, 40, 42, 45, 48, 50, 52, 55, 58};

		// Example current data: shifted toward higher bins.
		const std::vector<double> current_data = {
			20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
			70, 72, 75, 78, 80, 82, 85, 88, 90, 95};

		for (double x : reference_data)
		{
			update_histogram(reference, x);
		}

		for (double x : current_data)
		{
			update_histogram(current, x);
		}

		print_histogram(reference, "Reference histogram");
		print_histogram(current, "Current histogram");

		const std::vector<double> ref_p = to_proportions(reference);
		const std::vector<double> cur_p = to_proportions(current);

		print_proportions(ref_p, "Reference");
		print_proportions(cur_p, "Current");

		const double psi = compute_psi(reference, current);

		std::cout << "\nPopulation Stability Index (PSI): "
				  << std::fixed << std::setprecision(4) << psi << "\n";
		std::cout << "Interpretation: " << psi_interpretation(psi) << "\n";

		return 0;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
