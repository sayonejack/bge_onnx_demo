#pragma once

#include "BertTokenizer.h"
#include "BgeCoreC.h"

#include <QtGlobal>
#include <memory>
#include <vector>

class OrtEnv;
namespace Ort {
class Env;
class Session;
class SessionOptions;
} // namespace Ort

class BGECORE_EXPORT BgeOnnxEngine {
public:
  BgeOnnxEngine(const QString &model_path, const QString &vocab_path);
  ~BgeOnnxEngine();

  BgeOnnxEngine(const BgeOnnxEngine &) = delete;
  BgeOnnxEngine &operator=(const BgeOnnxEngine &) = delete;

  BertTokenizer &tokenizer();
  const BertTokenizer &tokenizer() const;
  int embeddingDim() const;

  std::vector<float>
  embedEncoding(const BertTokenizer::Encoding &encoding) const;
  std::vector<float> embedText(const QString &text, int max_length = 512) const;

private:
  BertTokenizer tokenizer_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::Session> session_;
  int embedding_dim_ = 0;
};
