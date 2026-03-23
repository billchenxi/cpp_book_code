/*
# 1) Point CMake to TorchConfig.cmake (your path may differ)
cmake -S . -B build -DTorch_DIR="../libtorch/share/cmake/Torch"

# (Alternative) If you prefer using CMAKE_PREFIX_PATH:
# cmake -S . -B build -DCMAKE_PREFIX_PATH="../libtorch"

# 2) Build
cmake --build build -j

# 3) Run (ensure LibTorch libs are on the search path)
DYLD_LIBRARY_PATH="../libtorch/lib" ./build/caffe2_normalize_demo
*/

#include <iostream>
#include <vector>

#if __has_include(<caffe2/core/init.h>) && \
	__has_include(<caffe2/core/tensor.h>) && \
	__has_include(<caffe2/core/workspace.h>)

#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/workspace.h>

using namespace caffe2;

template <typename T>
static void add_arg(OperatorDef *op, const std::string &name, T v)
{
	auto *a = op->add_arg();
	a->set_name(name);
	if constexpr (std::is_same_v<T, int>)
		a->set_i(v);
	else if constexpr (std::is_same_v<T, bool>)
		a->set_i(v ? 1 : 0);
}

int main(int argc, char **argv)
{
	GlobalInit(&argc, &argv);

	Workspace ws;

	{
		Blob *b = ws.CreateBlob("input");
		Tensor *t = BlobGetMutableTensor(b, CPU);
		t->Resize(3, 4, 4);
		float *p = t->mutable_data<float>();
		for (int i = 0; i < 3 * 4 * 4; ++i)
			p[i] = static_cast<float>(i);
	}

	{
		{
			Blob *b = ws.CreateBlob("mean");
			Tensor *t = BlobGetMutableTensor(b, CPU);
			t->Resize(3);
			float *p = t->mutable_data<float>();
			p[0] = 10.f;
			p[1] = 20.f;
			p[2] = 30.f;
		}
		{
			Blob *b = ws.CreateBlob("invstd");
			Tensor *t = BlobGetMutableTensor(b, CPU);
			t->Resize(3);
			float *p = t->mutable_data<float>();
			p[0] = 1.f / 5.f;
			p[1] = 1.f / 10.f;
			p[2] = 1.f / 20.f;
		}
	}

	NetDef net;
	net.set_name("normalize_net");
	{
		auto *sub = net.add_op();
		sub->set_type("Sub");
		sub->add_input("input");
		sub->add_input("mean");
		sub->add_output("centered");
		add_arg(sub, "broadcast", 1);
		add_arg(sub, "axis", 0);
	}
	{
		auto *mul = net.add_op();
		mul->set_type("Mul");
		mul->add_input("centered");
		mul->add_input("invstd");
		mul->add_output("normalized");
		add_arg(mul, "broadcast", 1);
		add_arg(mul, "axis", 0);
	}

	CAFFE_ENFORCE(ws.RunNetOnce(net));

	const Tensor &out = ws.GetBlob("normalized")->Get<Tensor>();
	const float *y = out.data<float>();
	std::cout << "normalized[0,0,0..3]: "
			  << y[0] << ", " << y[1] << ", " << y[2] << ", " << y[3] << "\n";
	std::cout << "normalized[1,0,0]: " << y[16] << "   "
			  << "normalized[2,0,0]: " << y[32] << "\n";

	return 0;
}

#else

int main()
{
	std::cout
		<< "This demo requires a Caffe2-enabled LibTorch package with public "
		   "caffe2/core headers. The vendored libtorch build in this repo "
		   "does not expose them, so the example is skipped.\n";
	return 0;
}

#endif
