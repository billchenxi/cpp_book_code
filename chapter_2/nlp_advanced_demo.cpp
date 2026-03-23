// nlp_advanced_demo.cpp
/*
1) Export models (once)
python3 -m venv .venv && source .venv/bin/activate
pip install -U "optimum[exporters,onnxruntime]" onnx

# NER (CoNLL03 fine-tuned BERT)
% optimum-cli export onnx \
  --task token-classification \
  --opset 14 \
  --model dbmdz/bert-base-cased-finetuned-conll03-english \
  ner_onnx/

# Sentiment (SST-2 fine-tuned DistilBERT)
% optimum-cli export onnx \
  --task text-classification \
  --opset 14 \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  sent_onnx/


You’ll get ner_onnx/model.onnx and sent_onnx/model.onnx.


2) Build
% g++ -std=c++17 -O2 nlp_advanced_demo.cpp -o nlp_advanced_demo \
		-I"$(brew --prefix onnxruntime)/include" \
		-L"$(brew --prefix onnxruntime)/lib" \
		-lonnxruntime

		DYLD_LIBRARY_PATH="$(brew --prefix onnxruntime)/lib" \
		./nlp_advanced_demo ner ./ner_onnx/model.onnx


3) Run
% DYLD_LIBRARY_PATH="$(brew --prefix onnxruntime)/lib" \
		./nlp_advanced_demo ner ./ner_onnx/model.onnx

Outputs:
Model inputs (3):
  input_ids : INT64, rank=2
  attention_mask : INT64, rank=2
  token_type_ids : INT64, rank=2
NER label ids (argmax per token):
  t=0 -> 0
  t=1 -> 0
  t=2 -> 0
  t=3 -> 2
  t=4 -> 0
  t=5 -> 0
  t=6 -> 0
  t=7 -> 0


% DYLD_LIBRARY_PATH="$(brew --prefix onnxruntime)/lib" \
	./nlp_advanced_demo sentiment ./sent_onnx/model.onnx

Outputs:
Model inputs (2):
  input_ids : INT64, rank=2
  attention_mask : INT64, rank=2
Sentiment class id: 1
*/

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using std::array;
using std::string;
using std::vector;

static const char *dtype_name(ONNXTensorElementDataType t)
{
	switch (t)
	{
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		return "INT8";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		return "INT16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		return "INT32";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		return "INT64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		return "BOOL";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return "FLOAT";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		return "DOUBLE";
	default:
		return "<unsupported>";
	}
}

template <typename T>
static Ort::Value make_tensor_cast(const Ort::MemoryInfo &mem,
								   const vector<int64_t> &src,
								   const array<int64_t, 2> &shape)
{
	static thread_local vector<T> buf;
	buf.resize(src.size());
	for (size_t i = 0; i < src.size(); ++i)
		buf[i] = static_cast<T>(src[i]);
	return Ort::Value::CreateTensor<T>(mem, buf.data(), buf.size(), shape.data(), 2);
}

static Ort::Value make_tensor_from_i64(const Ort::MemoryInfo &mem,
									   ONNXTensorElementDataType etype,
									   const vector<int64_t> &vals,
									   const array<int64_t, 2> &shape)
{
	switch (etype)
	{
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		return Ort::Value::CreateTensor<int64_t>(
			mem, const_cast<int64_t *>(vals.data()), vals.size(), shape.data(), 2);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		return make_tensor_cast<int32_t>(mem, vals, shape);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		return make_tensor_cast<int16_t>(mem, vals, shape);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		return make_tensor_cast<int8_t>(mem, vals, shape);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		return make_tensor_cast<uint32_t>(mem, vals, shape);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		return make_tensor_cast<uint16_t>(mem, vals, shape);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		return make_tensor_cast<uint8_t>(mem, vals, shape);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
	{
		static thread_local vector<uint8_t> b;
		b.resize(vals.size());
		for (size_t i = 0; i < vals.size(); ++i)
			b[i] = vals[i] != 0;
		return Ort::Value::CreateTensor<bool>(
			mem, reinterpret_cast<bool *>(b.data()), b.size(), shape.data(), 2);
	}
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return make_tensor_cast<float>(mem, vals, shape);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		return make_tensor_cast<double>(mem, vals, shape);
	default:
		throw std::runtime_error(std::string("Unsupported input dtype: ") + dtype_name(etype));
	}
}

static void debug_print_inputs(Ort::Session &sess)
{
	Ort::AllocatorWithDefaultOptions alloc;
	const size_t n = sess.GetInputCount();
	std::cerr << "Model inputs (" << n << "):\n";
	for (size_t i = 0; i < n; ++i)
	{
		auto nm = sess.GetInputNameAllocated(i, alloc);
		auto ti = sess.GetInputTypeInfo(i);
		if (ti.GetONNXType() != ONNX_TYPE_TENSOR)
		{
			std::cerr << "  " << nm.get() << " : <non-tensor>\n";
			continue;
		}
		auto ts = ti.GetTensorTypeAndShapeInfo();
		std::cerr << "  " << nm.get() << " : " << dtype_name(ts.GetElementType()) << ", rank=" << ts.GetShape().size() << "\n";
	}
}

// Build feeds. NOTE: token_type_ids_buf IS PROVIDED BY CALLER (stable lifetime).
static void build_inputs(Ort::Session &session,
						 const vector<int64_t> &ids,
						 const vector<int64_t> &mask,
						 const vector<int64_t> &token_type_ids_buf,
						 vector<string> &feed_names_keep,
						 vector<const char *> &feed_names,
						 vector<Ort::Value> &feeds)
{
	feed_names_keep.clear();
	feed_names.clear();
	feeds.clear();
	Ort::AllocatorWithDefaultOptions alloc;
	Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	array<int64_t, 2> shape{1, (int64_t)ids.size()};

	const size_t n_in = session.GetInputCount();
	feed_names_keep.reserve(n_in);
	feeds.reserve(n_in);

	for (size_t i = 0; i < n_in; ++i)
	{
		auto name_alloc = session.GetInputNameAllocated(i, alloc);
		string in_name = name_alloc.get();

		auto tinfo = session.GetInputTypeInfo(i);
		if (tinfo.GetONNXType() != ONNX_TYPE_TENSOR)
		{
			std::cerr << "Skip non-tensor " << in_name << "\n";
			continue;
		}
		auto tsi = tinfo.GetTensorTypeAndShapeInfo();
		auto etype = tsi.GetElementType();

		// Choose the backing buffer that LIVES in the caller:
		const vector<int64_t> *src = nullptr;
		if (in_name == "input_ids")
			src = &ids;
		else if (in_name == "attention_mask")
			src = &mask;
		else if (in_name == "token_type_ids" || in_name == "segment_ids")
			src = &token_type_ids_buf;
		else
		{ // unexpected tensor input of shape [1,T] -> zeros (also caller-managed)
			src = &token_type_ids_buf;
		}

		if (!src)
			throw std::runtime_error("No source buffer for input: " + in_name);
		Ort::Value t = make_tensor_from_i64(mem, etype, *src, shape);
		feed_names_keep.push_back(in_name);
		feeds.emplace_back(std::move(t));
	}

	// Freeze pointers AFTER names are done (avoid reallocation invalidation)
	feed_names.reserve(feed_names_keep.size());
	for (auto &s : feed_names_keep)
		feed_names.push_back(s.c_str());
}

static void choose_outputs(Ort::Session &sess,
						   vector<string> &out_keep, vector<const char *> &out_names)
{
	out_keep.clear();
	out_names.clear();
	Ort::AllocatorWithDefaultOptions alloc;
	const size_t n = sess.GetOutputCount();
	for (size_t i = 0; i < n; ++i)
	{
		auto nm = sess.GetOutputNameAllocated(i, alloc);
		if (string(nm.get()) == "logits")
		{
			out_keep.push_back("logits");
			out_names.push_back(out_keep.back().c_str());
			return;
		}
	}
	if (n == 0)
		throw std::runtime_error("Model has no outputs");
	auto nm0 = sess.GetOutputNameAllocated(0, alloc);
	out_keep.push_back(nm0.get());
	out_names.push_back(out_keep.back().c_str());
}

static vector<Ort::Value> run(Ort::Session &s,
							  const vector<const char *> &in_names,
							  vector<Ort::Value> &in_vals,
							  const vector<const char *> &out_names)
{
	return s.Run(Ort::RunOptions{}, in_names.data(), in_vals.data(),
				 in_vals.size(), out_names.data(), out_names.size());
}

static void demo_ner(const string &model)
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ner");
	Ort::SessionOptions so;
	so.SetIntraOpNumThreads(1);
	Ort::Session session(env, model.c_str(), so);

	debug_print_inputs(session);

	// Dummy example: [CLS] John lives in New York . [SEP]
	vector<int64_t> ids = {101, 2198, 3268, 1999, 2047, 2259, 1012, 102};
	vector<int64_t> mask(ids.size(), 1);

	// IMPORTANT: token_type_ids buffer must outlive Run()
	vector<int64_t> token_type(ids.size(), 0); // all zeros for single-sentence

	vector<string> feed_keep;
	vector<const char *> feed_names;
	vector<Ort::Value> feeds;
	build_inputs(session, ids, mask, token_type, feed_keep, feed_names, feeds);

	vector<string> out_keep;
	vector<const char *> out_names;
	choose_outputs(session, out_keep, out_names);

	auto outs = run(session, feed_names, feeds, out_names);
	if (outs.empty() || !outs[0].IsTensor())
		throw std::runtime_error("No tensor output");

	auto info = outs[0].GetTensorTypeAndShapeInfo();
	auto shp = info.GetShape(); // expect [1, T, C]
	if (shp.size() != 3)
		throw std::runtime_error("Unexpected NER output rank");
	int64_t T = shp[1], C = shp[2];
	const float *y = outs[0].GetTensorData<float>();

	std::cout << "NER label ids (argmax per token):\n";
	for (int64_t t = 0; t < T; ++t)
	{
		const float *row = y + t * C;
		auto it = std::max_element(row, row + C);
		std::cout << "  t=" << t << " -> " << (int64_t)std::distance(row, it) << "\n";
	}
}

static void demo_sent(const string &model)
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sent");
	Ort::SessionOptions so;
	so.SetIntraOpNumThreads(1);
	Ort::Session session(env, model.c_str(), so);

	debug_print_inputs(session);

	vector<int64_t> ids = {101, 1996, 3185, 2001, 2307, 999, 102};
	vector<int64_t> mask(ids.size(), 1);
	vector<int64_t> token_type(ids.size(), 0);

	vector<string> feed_keep;
	vector<const char *> feed_names;
	vector<Ort::Value> feeds;
	build_inputs(session, ids, mask, token_type, feed_keep, feed_names, feeds);

	vector<string> out_keep;
	vector<const char *> out_names;
	choose_outputs(session, out_keep, out_names);

	auto outs = run(session, feed_names, feeds, out_names);
	if (outs.empty() || !outs[0].IsTensor())
		throw std::runtime_error("No tensor output");

	auto info = outs[0].GetTensorTypeAndShapeInfo();
	auto shp = info.GetShape(); // expect [1, C]
	if (shp.size() != 2 || shp[0] != 1)
		throw std::runtime_error("Unexpected sentiment shape");
	int64_t C = shp[1];
	const float *y = outs[0].GetTensorData<float>();
	auto it = std::max_element(y, y + C);
	std::cout << "Sentiment class id: " << (int64_t)std::distance(y, it) << "\n";
}

int main(int argc, char **argv)
{
	try
	{
		if (argc < 3)
		{
			std::cerr << "Usage:\n  " << argv[0] << " ner /abs/path/to/ner_onnx/model.onnx\n"
					  << "  " << argv[0] << " sentiment /abs/path/to/sent_onnx/model.onnx\n";
			return 1;
		}
		string mode = argv[1], model = argv[2];
		if (mode == "ner")
			demo_ner(model);
		else if (mode == "sentiment")
			demo_sent(model);
		else
		{
			std::cerr << "Unknown mode\n";
			return 1;
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error: " << e.what() << "\n";
		return 2;
	}
	return 0;
}
