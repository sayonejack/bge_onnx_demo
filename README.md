# BGE ONNX C++ 推理库

这是一个基于 Qt5 和 ONNX Runtime 的 BGE 推理项目，已经拆分为可复用的动态库 `BgeCore.dll`，并提供：

- C++ 封装类 `BgeOnnxEngine`
- 纯 C ABI，方便其他语言或程序直接调用
- 示例程序 `BgeOnnxTest.exe`
- C ABI 示例程序 `BgeCApiDemo.exe`

当前使用的模型与词表位于：

- `onnx/bge-large-zh-v1.5-fp32.onnx`
- `bge_onnx_demo/Xenova-bge-large-zh-v1.5/vocab.txt`

当前内置的 ONNX Runtime 版本为：

- `onnxruntime-win-x64-1.17.1`

## 功能

- 使用 WordPiece/BERT tokenizer 对输入文本编码
- 调用 ONNX Runtime 执行 BGE 推理
- 提取 `last_hidden_state` 的 `[CLS]` 向量
- 输出 1024 维归一化 embedding
- 支持单句和双句输入

## 生成物

编译后会生成这些目标：

- `BgeCore.dll`：动态库
- `BgeCore.lib`：导入库
- `BgeOnnxTest.exe`：Qt 示例程序
- `BgeCApiDemo.exe`：纯 C API 示例程序

## 构建

### 1. 配置

在 `cpp_test` 目录下执行：

```powershell
mkdir build
cd build
cmake ..
```

如果你要显式指定 vcpkg 工具链，也可以这样配置：

```powershell
cmake .. -DCMAKE_TOOLCHAIN_FILE="D:/Qt/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
```

构建脚本默认会使用同目录下的 `onnxruntime-win-x64-1.17.1`，这是微软发布的 ONNX Runtime Windows 预编译包，不是 vcpkg 安装产物。

### 2. 编译

```powershell
cmake --build . --config Release
```

## 运行

### BgeOnnxTest

示例程序会直接走 `BgeCore.dll` 中的 C++ 封装。

```powershell
.\Release\BgeOnnxTest.exe
```

它支持这些调试参数：

- `--tokenize_text`：打印单句 token 结果
- `--tokenize_text_b`：配合 `--tokenize_text` 做双句测试
- `--max_length`：最大长度
- `--pad_to_max_length`：是否补齐到最大长度

### BgeCApiDemo

这是纯 C 接口示例，演示如何直接通过 DLL 计算 embedding：

```powershell
.\Release\BgeCApiDemo.exe
```

## C API

头文件是 `BgeCoreC.h`。外部程序可以直接包含它，然后链接 `BgeCore.lib` / 加载 `BgeCore.dll`。

可用接口：

- `bge_engine_create`
- `bge_engine_destroy`
- `bge_engine_get_embedding_dim`
- `bge_engine_embed_text_to_buffer`
- `bge_engine_embed_pair_to_buffer`
- `bge_engine_embed_text`
- `bge_engine_embed_pair`
- `bge_engine_free_float_array`

### 推荐用法

优先使用 buffer 版接口：

- `bge_engine_embed_text_to_buffer`
- `bge_engine_embed_pair_to_buffer`

这样由调用方自行分配输出缓冲区，不需要额外释放返回值。

如果你希望由 DLL 分配内存，也可以使用 `bge_engine_embed_text` / `bge_engine_embed_pair`，然后用 `bge_engine_free_float_array` 释放。

## C++ 封装

如果你在 C++ 项目里集成，可以直接包含 `BgeCore.h`，使用 `BgeOnnxEngine`：

- `tokenizer()`：访问 tokenizer
- `embeddingDim()`：查询 embedding 维度
- `embedText(...)`：单句推理
- `embedEncoding(...)`：使用已编码输入推理

## 目录说明

- `BertTokenizer.h/.cpp`：BERT/WordPiece tokenizer
- `BgeCore.h/.cpp`：BGE 推理引擎和 C ABI
- `BgeCoreC.h`：纯 C 头文件
- `main.cpp`：Qt 示例程序
- `c_api_demo.cpp`：纯 C API 示例程序
- `compare_tokenizer.py`：与 Hugging Face tokenizer 对照脚本

## 注意事项

1. 当前工程使用 `onnxruntime-win-x64-1.17.1`。如果你替换为其它版本，需同步检查头文件、DLL 名称和运行时依赖。
1. `BgeCore.dll` 依赖 `onnxruntime.dll` 和 `onnxruntime_providers_shared.dll`，运行时要保证它们在同一目录或系统可搜索路径中。
2. 词表 `vocab.txt` 必须保持原始行顺序，包括空行，否则 token id 会偏移。
3. 当前推理输出默认取 `last_hidden_state` 的第 0 个 token，也就是 `[CLS]` 向量，并做 L2 归一化。
4. 模型路径和词表路径支持相对路径，会自动从当前目录和程序目录向上搜索。
