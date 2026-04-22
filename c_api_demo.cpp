#include "BgeCoreC.h"

#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static fs::path resolve_resource_path(const fs::path &relative_path) {
  const fs::path roots[] = {fs::current_path()};
  for (const auto &root : roots) {
    fs::path dir = root;
    for (int i = 0; i < 6; ++i) {
      fs::path candidate = dir / relative_path;
      if (fs::exists(candidate)) {
        return candidate;
      }
      if (!dir.has_parent_path()) {
        break;
      }
      dir = dir.parent_path();
    }
  }
  return relative_path;
}

int main(int argc, char **argv) {
  const std::string model_arg =
      (argc > 1) ? argv[1] : "onnx/bge-large-zh-v1.5-fp32.onnx";
  const std::string vocab_arg =
      (argc > 2) ? argv[2]
                 : "bge_onnx_emo/Xenova-bge-large-zh-v1.5/vocab.txt";
  const std::string text_arg =
      (argc > 3) ? argv[3] : "这是一个商品标题示例";

  const fs::path model_path =
      resolve_resource_path(fs::path(model_arg)).lexically_normal();
  const fs::path vocab_path =
      resolve_resource_path(fs::path(vocab_arg)).lexically_normal();

  BgeEngineHandle *handle = bge_engine_create(model_path.string().c_str(),
                                              vocab_path.string().c_str());
  if (!handle) {
    std::cerr << "failed to create engine\n";
    return 1;
  }

  const int embedding_dim = bge_engine_get_embedding_dim(handle);
  std::cout << "embedding_dim=" << embedding_dim << "\n";

  std::vector<float> embedding(embedding_dim > 0 ? embedding_dim : 1024);
  int dim = 0;
  const int ok = bge_engine_embed_text_to_buffer(
      handle, text_arg.c_str(), 512, embedding.data(),
      static_cast<int>(embedding.size()), &dim);
  if (!ok || dim <= 0) {
    std::cerr << "failed to embed text\n";
    bge_engine_destroy(handle);
    return 2;
  }

  std::cout << "dim=" << dim << "\n";
  std::cout << "first_values=";
  for (int i = 0; i < std::min(dim, 8); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << std::fixed << std::setprecision(6) << embedding[i];
  }
  std::cout << "\n";

  bge_engine_destroy(handle);
  return 0;
}
