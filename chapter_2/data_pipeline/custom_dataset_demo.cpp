/*
% LIBTORCH=/path/to/libtorch
% g++ -std=c++17 custom_dataset_demo.cpp \
	-I"$LIBTORCH/include" -I"$LIBTORCH/include/torch/csrc/api/include" \
	-L"$LIBTORCH/lib" -ltorch -ltorch_cpu -lc10 \
	-Wl,-rpath,"$LIBTORCH/lib" -o custom_dataset_demo
% ./custom_dataset_demo
*/
#include <torch/torch.h>
#include <iostream>

// Minimal dataset: features X (N,D), labels y (N,)
struct CustomDataset : torch::data::datasets::Dataset<CustomDataset>
{
	CustomDataset(torch::Tensor data, torch::Tensor labels)
		: data_(std::move(data)), labels_(std::move(labels)) {}
	torch::data::Example<> get(size_t index) override
	{
		return {data_[index], labels_[index]}; // (D,), scalar label
	}
	torch::optional<size_t> size() const override { return data_.size(0); }

private:
	torch::Tensor data_, labels_;
};

int main()
{
	torch::manual_seed(0);
	const int64_t N = 512, D = 16, C = 3;

	auto X = torch::randn({N, D});							// synthetic features
	auto y = torch::randint(/*high=*/C, {N}, torch::kLong); // class labels

	// Stack batches + random sampling (shuffle)
	auto ds = CustomDataset(X, y).map(torch::data::transforms::Stack<>());
	using RandomSampler = torch::data::samplers::RandomSampler;
	auto loader = torch::data::make_data_loader<RandomSampler>(
		std::move(ds), torch::data::DataLoaderOptions().batch_size(32));

	torch::nn::Linear model(D, C);
	torch::optim::SGD opt(model->parameters(), /*lr=*/0.1);
	torch::nn::CrossEntropyLoss loss;

	for (int epoch = 0; epoch < 2; ++epoch)
	{
		double running = 0.0;
		int it = 0;
		for (auto &batch : *loader)
		{
			model->train();
			auto logits = model(batch.data);
			auto L = loss(logits, batch.target);
			opt.zero_grad();
			L.backward();
			opt.step();
			running += L.item<double>();
			++it;
		}
		std::cout << "epoch " << epoch << "  loss=" << (running / it) << "\n";
	}

	// quick sanity check
	model->eval();
	auto pred = model(X.slice(0, 0, 8)).argmax(1);
	std::cout << "pred[0:8]: " << pred << "\n";
	return 0;
}
