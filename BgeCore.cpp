#include "BgeCore.h"

#include <QDebug>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>

namespace {
void normalize(std::vector<float> &v) {
  float norm = 0.0f;
  for (float val : v) {
    norm += val * val;
  }
  norm = std::sqrt(norm);
  if (norm < 1e-12f) {
    norm = 1e-12f;
  }
  for (float &val : v) {
    val /= norm;
  }
}

bool copy_to_buffer(const std::vector<float> &vec, float *out_buffer,
                    int out_capacity, int *out_dim) {
  if (out_dim) {
    *out_dim = static_cast<int>(vec.size());
  }
  if (!out_buffer || out_capacity < static_cast<int>(vec.size())) {
    return false;
  }
  std::copy(vec.begin(), vec.end(), out_buffer);
  return true;
}
} // namespace

BgeOnnxEngine::BgeOnnxEngine(const QString &model_path,
                             const QString &vocab_path)
    : tokenizer_(vocab_path) {
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "BgeEngine");
  session_options_ = std::make_unique<Ort::SessionOptions>();
  session_options_->SetIntraOpNumThreads(0);
  session_options_->SetInterOpNumThreads(0);
  session_options_->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  session_options_->SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
  std::wstring model_path_wstr = model_path.toStdWString();
  session_ = std::make_unique<Ort::Session>(*env_, model_path_wstr.c_str(),
                                            *session_options_);
#else
  std::string model_path_stdstr = model_path.toStdString();
  session_ = std::make_unique<Ort::Session>(*env_, model_path_stdstr.c_str(),
                                            *session_options_);
#endif

  auto output_type_info = session_->GetOutputTypeInfo(0);
  auto shape = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() >= 3 && shape[2] > 0) {
    embedding_dim_ = static_cast<int>(shape[2]);
  }
}

BgeOnnxEngine::~BgeOnnxEngine() = default;

BertTokenizer &BgeOnnxEngine::tokenizer() { return tokenizer_; }

const BertTokenizer &BgeOnnxEngine::tokenizer() const { return tokenizer_; }

int BgeOnnxEngine::embeddingDim() const { return embedding_dim_; }

std::vector<float> BgeOnnxEngine::embedText(const QString &text,
                                            int max_length) const {
  BertTokenizer::Encoding encoding =
      tokenizer_.encodeForModel(text, max_length);
  return embedEncoding(encoding);
}

std::vector<float>
BgeOnnxEngine::embedEncoding(const BertTokenizer::Encoding &encoding) const {
  std::vector<int64_t> input_ids = encoding.input_ids;
  std::vector<int64_t> attention_mask = encoding.attention_mask;
  std::vector<int64_t> token_type_ids = encoding.token_type_ids;
  std::vector<int64_t> input_shape = {1,
                                      static_cast<int64_t>(input_ids.size())};

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor_1 = Ort::Value::CreateTensor<int64_t>(
      memory_info, input_ids.data(), input_ids.size(), input_shape.data(),
      input_shape.size());
  Ort::Value input_tensor_2 = Ort::Value::CreateTensor<int64_t>(
      memory_info, attention_mask.data(), attention_mask.size(),
      input_shape.data(), input_shape.size());
  Ort::Value input_tensor_3 = Ort::Value::CreateTensor<int64_t>(
      memory_info, token_type_ids.data(), token_type_ids.size(),
      input_shape.data(), input_shape.size());

  const char *input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
  const char *output_names[] = {"last_hidden_state"};
  Ort::Value input_tensors[3] = {std::move(input_tensor_1),
                                 std::move(input_tensor_2),
                                 std::move(input_tensor_3)};

  auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names,
                                      input_tensors, 3, output_names, 1);

  float *floatarr = output_tensors[0].GetTensorMutableData<float>();
  auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
  auto shape = type_info.GetShape();
  int hidden_size = static_cast<int>(shape[2]);
  std::vector<float> cls_embedding(floatarr, floatarr + hidden_size);
  normalize(cls_embedding);
  return cls_embedding;
}

struct BgeEngineHandle {
  std::unique_ptr<BgeOnnxEngine> engine;
};

extern "C" {

BgeEngineHandle *bge_engine_create(const char *model_path_utf8,
                                   const char *vocab_path_utf8) {
  if (!model_path_utf8 || !vocab_path_utf8) {
    return nullptr;
  }
  try {
    auto handle = std::make_unique<BgeEngineHandle>();
    handle->engine = std::make_unique<BgeOnnxEngine>(
        QString::fromUtf8(model_path_utf8), QString::fromUtf8(vocab_path_utf8));
    return handle.release();
  } catch (...) {
    return nullptr;
  }
}

void bge_engine_destroy(BgeEngineHandle *handle) { delete handle; }

float *bge_engine_embed_text(BgeEngineHandle *handle, const char *text_utf8,
                             int max_length, int *out_dim) {
  if (out_dim) {
    *out_dim = 0;
  }
  if (!handle || !handle->engine || !text_utf8 || max_length <= 0) {
    return nullptr;
  }
  try {
    std::vector<float> vec =
        handle->engine->embedText(QString::fromUtf8(text_utf8), max_length);
    if (out_dim) {
      *out_dim = static_cast<int>(vec.size());
    }
    float *buffer = new float[vec.size()];
    std::copy(vec.begin(), vec.end(), buffer);
    return buffer;
  } catch (...) {
    return nullptr;
  }
}

float *bge_engine_embed_pair(BgeEngineHandle *handle, const char *text_a_utf8,
                             const char *text_b_utf8, int max_length,
                             int *out_dim) {
  if (out_dim) {
    *out_dim = 0;
  }
  if (!handle || !handle->engine || !text_a_utf8 || !text_b_utf8 ||
      max_length <= 0) {
    return nullptr;
  }
  try {
    BertTokenizer::Encoding encoding =
        handle->engine->tokenizer().encodePairForModel(
            QString::fromUtf8(text_a_utf8), QString::fromUtf8(text_b_utf8),
            max_length);
    std::vector<float> vec = handle->engine->embedEncoding(encoding);
    if (out_dim) {
      *out_dim = static_cast<int>(vec.size());
    }
    float *buffer = new float[vec.size()];
    std::copy(vec.begin(), vec.end(), buffer);
    return buffer;
  } catch (...) {
    return nullptr;
  }
}

int bge_engine_get_embedding_dim(BgeEngineHandle *handle) {
  if (!handle || !handle->engine) {
    return 0;
  }
  return handle->engine->embeddingDim();
}

int bge_engine_embed_text_to_buffer(BgeEngineHandle *handle,
                                    const char *text_utf8, int max_length,
                                    float *out_buffer, int out_capacity,
                                    int *out_dim) {
  if (out_dim) {
    *out_dim = 0;
  }
  if (!handle || !handle->engine || !text_utf8 || max_length <= 0) {
    return 0;
  }
  try {
    std::vector<float> vec =
        handle->engine->embedText(QString::fromUtf8(text_utf8), max_length);
    return copy_to_buffer(vec, out_buffer, out_capacity, out_dim) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

int bge_engine_embed_pair_to_buffer(BgeEngineHandle *handle,
                                    const char *text_a_utf8,
                                    const char *text_b_utf8, int max_length,
                                    float *out_buffer, int out_capacity,
                                    int *out_dim) {
  if (out_dim) {
    *out_dim = 0;
  }
  if (!handle || !handle->engine || !text_a_utf8 || !text_b_utf8 ||
      max_length <= 0) {
    return 0;
  }
  try {
    BertTokenizer::Encoding encoding =
        handle->engine->tokenizer().encodePairForModel(
            QString::fromUtf8(text_a_utf8), QString::fromUtf8(text_b_utf8),
            max_length);
    std::vector<float> vec = handle->engine->embedEncoding(encoding);
    return copy_to_buffer(vec, out_buffer, out_capacity, out_dim) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

void bge_engine_free_float_array(float *ptr) { delete[] ptr; }

} // extern "C"
