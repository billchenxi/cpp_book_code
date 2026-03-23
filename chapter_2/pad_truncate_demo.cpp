/*
% g++ -std=c++17 -O2 pad_truncate_demo.cpp -o pad_truncate_demo
% ./pad_truncate_demo
*/

#include <iostream>
#include <vector>
#include <algorithm>

// pad/truncate to fixed length
template <typename T>
std::vector<T> padOrTruncate(const std::vector<T> &ids, size_t max_len, T pad)
{
	if (ids.size() >= max_len)
		return {ids.begin(), ids.begin() + max_len};
	std::vector<T> out = ids;
	out.resize(max_len, pad);
	return out;
}

// attention mask: 1 for real tokens, 0 for padding
template <typename T>
std::vector<int64_t> makeMask(const std::vector<T> &ids, size_t max_len)
{
	std::vector<int64_t> m(std::min(ids.size(), max_len), 1);
	m.resize(max_len, 0);
	return m;
}

template <typename T>
void printVec(const std::vector<T> &v, const char *name)
{
	std::cout << name << " = [";
	for (size_t i = 0; i < v.size(); ++i)
	{
		if (i)
			std::cout << ' ';
		std::cout << v[i];
	}
	std::cout << "]\n";
}

int main()
{
	// Example token IDs (CLS ... SEP)
	std::vector<int64_t> ids_long = {101, 2023, 2003, 1037, 3978, 102};
	std::vector<int64_t> ids_short = {101, 102};

	const size_t MAXLEN = 5;
	const int64_t PAD = 0;

	auto long_fixed = padOrTruncate(ids_long, MAXLEN, PAD);
	auto long_mask = makeMask(ids_long, MAXLEN);

	auto short_fixed = padOrTruncate(ids_short, MAXLEN, PAD);
	auto short_mask = makeMask(ids_short, MAXLEN);

	printVec(long_fixed, "trunc(ids_long)");
	printVec(long_mask, "mask_long");
	printVec(short_fixed, "pad(ids_short)");
	printVec(short_mask, "mask_short");
	return 0;
}
