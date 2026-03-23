/*
Build: g++ -std=c++17 -O2 bigdata_demo.cpp -o bigdata_demo

Run:
1) Make a 64MB sample file (deterministic contents)
% ./bigdata_demo make-sample sample.bin 67108864

2) Memory-map it and compute a checksum
% ./bigdata_demo mmap sample.bin

3) Stream in 1 MiB batches (you can change chunk size)
% ./bigdata_demo batch sample.bin 1048576

4) Range-read windows (simulates DFS-style ranged I/O) with a 2 MiB window
% ./bigdata_demo range sample.bin 2097152
*/
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using u8 = unsigned char;

// ---------------------- Utility: FNV-1a 64-bit checksum ----------------------
static inline uint64_t fnv1a64_update(const u8 *p, size_t n, uint64_t h = 1469598103934665603ULL)
{
	constexpr uint64_t FNV_PRIME = 1099511628211ULL;
	for (size_t i = 0; i < n; ++i)
	{
		h ^= p[i];
		h *= FNV_PRIME;
	}
	return h;
}

// ---------------------------- Memory-mapped file -----------------------------
struct MappedFile
{
	void *data = nullptr;
	size_t size = 0;
	int fd = -1;

	explicit MappedFile(const std::string &path)
	{
		fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0)
			throw std::runtime_error("open failed: " + path);
		struct stat st{};
		if (fstat(fd, &st) != 0)
			throw std::runtime_error("fstat failed");
		size = static_cast<size_t>(st.st_size);
		if (size == 0)
			throw std::runtime_error("file is empty");
		data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
		if (data == MAP_FAILED)
			throw std::runtime_error("mmap failed");
	}
	~MappedFile()
	{
		if (data && data != MAP_FAILED)
			::munmap(data, size);
		if (fd >= 0)
			::close(fd);
	}
};

// ----------------------------- Batch (streaming) -----------------------------
static uint64_t process_in_batches(const std::string &path, size_t chunk = 1 << 20)
{
	std::ifstream f(path, std::ios::binary);
	if (!f)
		throw std::runtime_error("ifstream open failed: " + path);
	std::vector<u8> buf(chunk);
	uint64_t h = 1469598103934665603ULL;
	while (f)
	{
		f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
		std::streamsize n = f.gcount();
		if (n > 0)
			h = fnv1a64_update(buf.data(), static_cast<size_t>(n), h);
	}
	return h;
}

// --------------------------- Range (pread) “DFS” ----------------------------
struct FileRangeReader
{
	int fd = -1;
	size_t size = 0;
	explicit FileRangeReader(const std::string &path)
	{
		fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0)
			throw std::runtime_error("open failed: " + path);
		struct stat st{};
		if (fstat(fd, &st) != 0)
			throw std::runtime_error("fstat failed");
		size = static_cast<size_t>(st.st_size);
	}
	~FileRangeReader()
	{
		if (fd >= 0)
			::close(fd);
	}

	// Read exactly up to len bytes starting at off (clamped to file size).
	size_t read_range(uint64_t off, size_t len, std::vector<u8> &out)
	{
		if (off >= size)
		{
			out.clear();
			return 0;
		}
		size_t to_read = std::min<size_t>(len, size - static_cast<size_t>(off));
		out.resize(to_read);
		size_t done = 0;
		while (done < to_read)
		{
			ssize_t n = ::pread(fd, out.data() + done, to_read - done, static_cast<off_t>(off + done));
			if (n < 0)
				throw std::runtime_error("pread failed");
			if (n == 0)
				break;
			done += static_cast<size_t>(n);
		}
		out.resize(done);
		return done;
	}
};

static uint64_t process_by_ranges(const std::string &path, size_t window = 2 << 20)
{
	FileRangeReader r(path);
	uint64_t h = 1469598103934665603ULL;
	std::vector<u8> buf;
	for (uint64_t off = 0; off < r.size; off += window)
	{
		size_t n = r.read_range(off, window, buf);
		if (n == 0)
			break;
		h = fnv1a64_update(buf.data(), n, h);
	}
	return h;
}

// ---------------------------- Pretty hex preview -----------------------------
static void print_preview(const u8 *p, size_t n, size_t max_bytes = 64)
{
	size_t m = std::min(n, max_bytes);
	std::cout << "preview (" << m << " bytes): ";
	for (size_t i = 0; i < m; ++i)
	{
		static const char *hex = "0123456789abcdef";
		unsigned v = p[i];
		std::cout << hex[(v >> 4) & 0xF] << hex[v & 0xF] << (i + 1 < m ? ' ' : '\n');
	}
}

// ---------------------------- Sample file writer -----------------------------
static void make_sample(const std::string &path, size_t bytes)
{
	std::ofstream o(path, std::ios::binary | std::ios::trunc);
	if (!o)
		throw std::runtime_error("ofstream open failed: " + path);
	std::vector<u8> block(1 << 20);	   // 1 MiB block
	uint64_t x = 88172645463393265ULL; // simple xorshift seed
	auto next = [&]()
	{
		x ^= x << 7;
		x ^= x >> 9;
		x ^= x << 8;
		return static_cast<u8>(x & 0xFF);
	};
	size_t left = bytes;
	while (left > 0)
	{
		size_t n = std::min(left, block.size());
		for (size_t i = 0; i < n; ++i)
			block[i] = next();
		o.write(reinterpret_cast<const char *>(block.data()), static_cast<std::streamsize>(n));
		left -= n;
	}
	std::cout << "Wrote " << bytes << " bytes to " << path << "\n";
}

// ----------------------------------- main ------------------------------------
int main(int argc, char **argv)
{
	try
	{
		if (argc < 2)
		{
			std::cerr
				<< "Usage:\n"
				<< "  " << argv[0] << " make-sample <path> <bytes>\n"
				<< "  " << argv[0] << " mmap         <path>\n"
				<< "  " << argv[0] << " batch        <path> [chunk_bytes]\n"
				<< "  " << argv[0] << " range        <path> [window_bytes]\n";
			return 1;
		}

		std::string cmd = argv[1];

		if (cmd == "make-sample")
		{
			if (argc != 4)
				throw std::runtime_error("make-sample needs <path> <bytes>");
			std::string path = argv[2];
			size_t bytes = static_cast<size_t>(std::stoull(argv[3]));
			make_sample(path, bytes);
			return 0;
		}

		if (cmd == "mmap")
		{
			if (argc != 3)
				throw std::runtime_error("mmap needs <path>");
			MappedFile mf(argv[2]);
			uint64_t h = fnv1a64_update(reinterpret_cast<const u8 *>(mf.data), mf.size);
			std::cout << "[mmap] size=" << mf.size << " checksum=0x" << std::hex << h << std::dec << "\n";
			print_preview(reinterpret_cast<const u8 *>(mf.data), mf.size);
			return 0;
		}

		if (cmd == "batch")
		{
			if (argc < 3 || argc > 4)
				throw std::runtime_error("batch needs <path> [chunk_bytes]");
			size_t chunk = (argc == 4) ? static_cast<size_t>(std::stoull(argv[3])) : (1 << 20);
			uint64_t h = process_in_batches(argv[2], chunk);
			std::cout << "[batch] chunk=" << chunk << " checksum=0x" << std::hex << h << std::dec << "\n";
			return 0;
		}

		if (cmd == "range")
		{
			if (argc < 3 || argc > 4)
				throw std::runtime_error("range needs <path> [window_bytes]");
			size_t win = (argc == 4) ? static_cast<size_t>(std::stoull(argv[3])) : (2 << 20);
			uint64_t h = process_by_ranges(argv[2], win);
			std::cout << "[range] window=" << win << " checksum=0x" << std::hex << h << std::dec << "\n";
			return 0;
		}

		throw std::runtime_error("unknown command: " + cmd);
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error: " << e.what() << "\n";
		return 2;
	}
}
