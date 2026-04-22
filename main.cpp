#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDataStream>
#include <QDebug>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QString>
#include <QStringList>
#include <QTextStream>
#include <QUrl>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "BgeCore.h"

static QString resolveResourcePath(const QString &relative_path) {
  const QStringList roots = {QDir::currentPath(),
                             QCoreApplication::applicationDirPath()};
  for (const QString &root : roots) {
    QDir dir(root);
    for (int i = 0; i < 6; ++i) {
      const QString candidate = dir.absoluteFilePath(relative_path);
      if (QFileInfo::exists(candidate)) {
        return candidate;
      }
      if (!dir.cdUp()) {
        break;
      }
    }
  }
  return relative_path;
}

// 向量 L2 归一化
void normalize(std::vector<float> &v) {
  float norm = 0.0f;
  for (float val : v) {
    norm += val * val;
  }
  norm = std::sqrt(norm);
  if (norm < 1e-12)
    norm = 1e-12;
  for (float &val : v) {
    val /= norm;
  }
}

// 计算两个向量的余弦相似度 (假设已经过 L2 归一化)
float cosine_similarity(const std::vector<float> &v1,
                        const std::vector<float> &v2) {
  if (v1.size() != v2.size() || v1.empty())
    return 0.0f;
  float dot_product = 0.0f;
  for (size_t i = 0; i < v1.size(); ++i) {
    dot_product += v1[i] * v2[i];
  }
  return dot_product;
}

// ---------------------------------------------------------
// 调用 AI LLM 进行批量的分类决断
// ---------------------------------------------------------
QList<QJsonObject> call_ai_batch(const QString &api_base,
                                 const QString &model_name,
                                 const QStringList &titles,
                                 const QList<QStringList> &candidates_list,
                                 int top_k, qint64 &elapsed_ms) {
  QElapsedTimer timer;
  timer.start();

  QNetworkAccessManager manager;
  QString full_url = api_base.trimmed();
  if (!full_url.contains("/chat/completions")) {
    if (!full_url.endsWith("/"))
      full_url += "/";
    full_url += "chat/completions";
  }

  QUrl url(full_url);
  QNetworkRequest request(url);
  request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  QStringList lines;
  lines << "你将收到多个商品标题，每个标题都有一组带有序号的候选类别。"
        << QString("请为每个标题从其候选类别中选择最合适的 1 "
                   "个类别，并**只返回该类别的序号**（数字 1 到 %1）。")
               .arg(top_k)
        << "必须严格输出 JSON（不要 markdown，不要解释，不要多余字符）。"
        << "输出 JSON schema："
        << "{ \"results\": [ {\"idx\": 1, \"pick_id\": 3}, ... ] }"
        << "规则："
        << "- idx 必须和输入标题序号一致"
        << "- pick_id 必须是你选中的候选类别的序号（数字）"
        << ""
        << "输入：";

  for (int i = 0; i < titles.size(); ++i) {
    lines << QString("%1. 商品标题：%2").arg(i + 1).arg(titles[i]);
    lines << "候选类别：";
    const QStringList &candidates = candidates_list[i];
    for (int j = 0; j < candidates.size(); ++j) {
      lines << QString("  %1. %2").arg(j + 1).arg(candidates[j]);
    }
    lines << "";
  }

  QString prompt = lines.join("\n");

  QJsonObject systemMsg;
  systemMsg["role"] = "system";
  systemMsg["content"] =
      "你是一个专业的电商分类专家。只输出严格 JSON（必须使用双引号，不要 "
      "markdown，不要解释，不要尾随逗号）。";

  QJsonObject userMsg;
  userMsg["role"] = "user";
  userMsg["content"] = prompt;

  QJsonArray messages;
  messages.append(systemMsg);
  messages.append(userMsg);

  QJsonObject payload;
  payload["model"] = model_name;
  payload["messages"] = messages;
  payload["temperature"] = 0.0;
  payload["max_tokens"] = 2000;
  payload["stream"] = false;
  payload["reasoning_effort"] = "none";

  QJsonDocument doc(payload);
  QByteArray data = doc.toJson(QJsonDocument::Compact);

  QNetworkReply *reply = manager.post(request, data);
  QEventLoop loop;
  QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
  loop.exec();

  elapsed_ms = timer.elapsed();

  QList<QJsonObject> parsed_results;
  QString result_text;

  if (reply->error() == QNetworkReply::NoError) {
    QByteArray responseData = reply->readAll();
    QJsonDocument responseDoc = QJsonDocument::fromJson(responseData);
    if (!responseDoc.isNull() && responseDoc.isObject()) {
      QJsonObject rootObj = responseDoc.object();
      if (rootObj.contains("choices") && rootObj["choices"].isArray()) {
        QJsonArray choices = rootObj["choices"].toArray();
        if (!choices.isEmpty()) {
          QJsonObject firstChoice = choices[0].toObject();
          if (firstChoice.contains("message") &&
              firstChoice["message"].isObject()) {
            QJsonObject messageObj = firstChoice["message"].toObject();
            if (messageObj.contains("content")) {
              result_text = messageObj["content"].toString().trimmed();
            }
          }
        }
      }
    }
  } else {
    qCritical() << "LLM 批量请求失败:" << reply->errorString();
  }
  reply->deleteLater();

  // 清理 <think> 标签内容
  int thinkStart = result_text.indexOf("<think>", 0, Qt::CaseInsensitive);
  int thinkEnd = result_text.indexOf("</think>", 0, Qt::CaseInsensitive);
  if (thinkStart != -1 && thinkEnd != -1 && thinkEnd > thinkStart) {
    result_text.remove(thinkStart, thinkEnd + 8 - thinkStart);
    result_text = result_text.trimmed();
  } else if (thinkStart != -1 && thinkEnd == -1) {
    result_text.remove(thinkStart, result_text.length() - thinkStart);
    result_text = result_text.trimmed();
  }

  // 尝试解析 JSON
  QJsonDocument json_reply = QJsonDocument::fromJson(result_text.toUtf8());
  if (json_reply.isNull()) {
    // 如果解析失败，尝试用正则提取 {}
    QRegularExpression re("\\{[\\s\\S]*\\}");
    QRegularExpressionMatch match = re.match(result_text);
    if (match.hasMatch()) {
      json_reply = QJsonDocument::fromJson(match.captured(0).toUtf8());
    }
  }

  if (!json_reply.isNull() && json_reply.isObject()) {
    QJsonObject root = json_reply.object();
    if (root.contains("results") && root["results"].isArray()) {
      QJsonArray resArray = root["results"].toArray();
      for (int i = 0; i < resArray.size(); ++i) {
        if (resArray[i].isObject()) {
          QJsonObject item = resArray[i].toObject();
          int idx = item.contains("idx") ? item["idx"].toInt(-1) : -1;
          int pick_id =
              item.contains("pick_id") ? item["pick_id"].toInt(-1) : -1;

          if (idx >= 1 && idx <= titles.size()) {
            QString title = titles[idx - 1];
            const QStringList &cands = candidates_list[idx - 1];
            QString category = "未知类别";
            if (pick_id >= 1 && pick_id <= cands.size()) {
              category = cands[pick_id - 1];
            }

            QJsonObject out_item;
            out_item["idx"] = idx;
            out_item["title"] = title;
            out_item["pick_id"] = pick_id;
            out_item["category"] = category;
            parsed_results.append(out_item);
          }
        }
      }
    }
  }

  if (parsed_results.isEmpty() && !result_text.isEmpty()) {
    qWarning() << "批量 JSON 解析失败或缺失 results. 原始返回:" << result_text;
  }

  return parsed_results;
}

// ---------------------------------------------------------
// 调用 AI LLM 进行最终的分类决断 (单条)
// ---------------------------------------------------------
QString call_ai(const QString &api_base, const QString &model_name,
                const QString &title, const QStringList &candidates,
                qint64 &elapsed_ms) {
  QElapsedTimer timer;
  timer.start();

  QNetworkAccessManager manager;
  QString full_url = api_base.trimmed();
  if (!full_url.contains("/chat/completions")) {
    if (!full_url.endsWith("/"))
      full_url += "/";
    full_url += "chat/completions";
  }

  QUrl url(full_url);
  QNetworkRequest request(url);
  request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  QString options_text;
  for (int i = 0; i < candidates.size(); ++i) {
    options_text += QString::number(i + 1) + ". " + candidates[i] + "\n";
  }

  QString prompt =
      QString("商品标题：'%1'\n\n候选类别：\n%"
              "2\n请从上面的候选类别中，选出最合适该商品的一个类别。只输出该类"
              "别的完整文本，不要解释，不要包含序号数字。")
          .arg(title, options_text);

  QJsonObject systemMsg;
  systemMsg["role"] = "system";
  systemMsg["content"] = "你是一个专业的电商分类专家。直接输出最终的类别名称，"
                         "不要输出思考过程。必须输出完整的类别路径。";

  QJsonObject userMsg;
  userMsg["role"] = "user";
  userMsg["content"] = prompt;

  QJsonArray messages;
  messages.append(systemMsg);
  messages.append(userMsg);

  QJsonObject payload;
  payload["model"] = model_name;
  payload["messages"] = messages;
  payload["temperature"] = 0.0;
  // 增加 max_tokens 避免被截断
  payload["max_tokens"] = 200;
  payload["stream"] = false;
  // 禁用 reasoning (类似 python 中的 "reasoning_effort": "none") 避免内容为空
  payload["reasoning_effort"] = "none";

  QJsonDocument doc(payload);
  QByteArray data = doc.toJson(QJsonDocument::Compact);

  QNetworkReply *reply = manager.post(request, data);
  QEventLoop loop;
  QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
  loop.exec();

  elapsed_ms = timer.elapsed();

  QString result_text;
  if (reply->error() == QNetworkReply::NoError) {
    QByteArray responseData = reply->readAll();
    // qDebug() << "AI API Raw Response:" << responseData;
    QJsonDocument responseDoc = QJsonDocument::fromJson(responseData);
    if (!responseDoc.isNull() && responseDoc.isObject()) {
      QJsonObject rootObj = responseDoc.object();
      if (rootObj.contains("choices") && rootObj["choices"].isArray()) {
        QJsonArray choices = rootObj["choices"].toArray();
        if (!choices.isEmpty()) {
          QJsonObject firstChoice = choices[0].toObject();
          if (firstChoice.contains("message") &&
              firstChoice["message"].isObject()) {
            QJsonObject messageObj = firstChoice["message"].toObject();
            // 兼容某些返回的推理内容为空但是普通内容在content里的情况
            if (messageObj.contains("content")) {
              result_text = messageObj["content"].toString().trimmed();
            }
          }
        }
      }
    }
    if (result_text.isEmpty()) {
      result_text = "未找到预期的回复格式, 原始返回: " + responseData;
    }
  } else {
    qCritical() << "LLM 单条请求失败:" << reply->errorString();
    result_text = QString("LLM 单条请求失败: %1").arg(reply->errorString());
  }

  reply->deleteLater();

  // 清理 <think> 标签内容
  int thinkStart = result_text.indexOf("<think>", 0, Qt::CaseInsensitive);
  int thinkEnd = result_text.indexOf("</think>", 0, Qt::CaseInsensitive);
  if (thinkStart != -1 && thinkEnd != -1 && thinkEnd > thinkStart) {
    result_text.remove(thinkStart, thinkEnd + 8 - thinkStart);
    result_text = result_text.trimmed();
  } else if (thinkStart != -1 && thinkEnd == -1) {
    // 防止未闭合的标签
    result_text.remove(thinkStart, result_text.length() - thinkStart);
    result_text = result_text.trimmed();
  }

  return result_text;
}

// ---------------------------------------------------------

int main(int argc, char *argv[]) {
  QCoreApplication a(argc, argv);

  QCommandLineParser parser;
  parser.setApplicationDescription(
      "BGE ONNX C++ Qt5 Test (With Custom Tokenizer)");
  parser.addHelpOption();

  QCommandLineOption topKOption(QStringList() << "k"
                                              << "top_k",
                                "BGE的召回数量 (默认: 30)", "number", "30");
  parser.addOption(topKOption);

  QCommandLineOption batchSizeOption(QStringList() << "b"
                                                   << "batch_size",
                                     "AI批量处理的大小 (默认: 10)", "number",
                                     "10");
  parser.addOption(batchSizeOption);

  QCommandLineOption modelPathOption(QStringList() << "m"
                                                   << "onnx_model",
                                     "指定 ONNX 模型路径", "path");
  parser.addOption(modelPathOption);

  QCommandLineOption aiModelNameOption(QStringList() << "n"
                                                     << "model_name",
                                       "指定 AI 模型名称 (默认: local-model)",
                                       "name", "local-model");
  parser.addOption(aiModelNameOption);

  QCommandLineOption apiBaseOption(
      QStringList() << "a"
                    << "api_base",
      "指定 AI API Base URL (默认: http://localhost:1234/v1)", "url",
      "http://localhost:1234/v1");
  parser.addOption(apiBaseOption);

  QCommandLineOption tokenizeTextOption(QStringList() << "t"
                                                      << "tokenize_text",
                                        "仅打印指定文本的 tokenizer "
                                        "输出并退出",
                                        "text");
  parser.addOption(tokenizeTextOption);

  QCommandLineOption tokenizeTextBOption(QStringList() << "u"
                                                       << "tokenize_text_b",
                                         "与 --tokenize_text 组成双句输入",
                                         "text_b");
  parser.addOption(tokenizeTextBOption);

  QCommandLineOption maxLengthOption(QStringList() << "l"
                                                   << "max_length",
                                     "tokenizer 最大长度 (默认: 512)", "number",
                                     "512");
  parser.addOption(maxLengthOption);

  QCommandLineOption padOption(QStringList() << "p"
                                             << "pad_to_max_length",
                               "tokenizer 输出补齐到 max_length");
  parser.addOption(padOption);

  parser.process(a);

  int top_k = parser.value(topKOption).toInt();
  if (top_k <= 0)
    top_k = 30;

  int batch_size = parser.value(batchSizeOption).toInt();
  if (batch_size <= 0)
    batch_size = 10;

  int max_length = parser.value(maxLengthOption).toInt();
  if (max_length <= 0)
    max_length = 512;
  const bool pad_to_max_length = parser.isSet(padOption);

  QString api_base = parser.value(apiBaseOption);
  const QString model_rel_path = "onnx/bge-large-zh-v1.5-fp32.onnx";
  const QString vocab_rel_path = "bge_onnx_emo/Xenova-bge-large-zh-v1.5/vocab.txt";
  const QString categories_rel_path = "categorys.txt";
  const QString vectors_rel_path = "category_vectors.bin";
  QString model_path_str;
#ifdef _WIN32
  model_path_str = model_rel_path;
#else
  model_path_str = model_rel_path;
#endif
  if (parser.isSet(modelPathOption)) {
    model_path_str = parser.value(modelPathOption);
  }

  QString ai_model_name = parser.value(aiModelNameOption);

  qDebug() << "=========================================";
  qDebug() << "BGE ONNX C++ Qt5 Test (With Custom Tokenizer)";
  qDebug() << "=========================================";
  qDebug() << "配置: Top K =" << top_k << ", 模型 =" << model_path_str;
  qDebug() << "API Base =" << api_base;

  try {
    const QString vocab_path = resolveResourcePath(vocab_rel_path);
    qDebug() << "正在加载 Tokenizer 词表:" << vocab_path;
    BertTokenizer tokenizer(vocab_path);

    if (parser.isSet(tokenizeTextOption)) {
      QString text = parser.value(tokenizeTextOption);
      const bool has_pair = parser.isSet(tokenizeTextBOption);
      BertTokenizer::EncodeOptions options;
      options.max_length = max_length;
      options.pad_to_max_length = pad_to_max_length;
      BertTokenizer::Encoding encoding =
          has_pair ? tokenizer.encodePairForModel(
                         text, parser.value(tokenizeTextBOption), options)
                   : tokenizer.encodeForModel(text, options);

      auto joinIds = [](const std::vector<int64_t> &values) {
        QString out;
        for (size_t i = 0; i < values.size(); ++i) {
          if (i)
            out += ' ';
          out += QString::number(values[i]);
        }
        return out;
      };

      qDebug().noquote() << "input_ids:" << joinIds(encoding.input_ids);
      qDebug().noquote() << "attention_mask:"
                         << joinIds(encoding.attention_mask);
      qDebug().noquote() << "token_type_ids:"
                         << joinIds(encoding.token_type_ids);
      return 0;
    }

    const QString categories_path = resolveResourcePath(categories_rel_path);
    qDebug() << "正在加载类别文本数据:" << categories_path;
    QStringList categories;
    QFile cat_file(categories_path);
    if (cat_file.open(QIODevice::ReadOnly | QIODevice::Text)) {
      QTextStream in(&cat_file);
      in.setCodec("UTF-8");
      while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        QStringList parts = line.split('\t');
        if (parts.size() >= 2) {
          categories.append(parts[1]);
        } else {
          categories.append(line);
        }
      }
      cat_file.close();
    } else {
      qCritical() << "加载 categorys.txt 失败!";
    }
    qDebug() << "共加载了" << categories.size() << "个类别";

    // 注意：C++ 原生无法直接读取 Python 的 pickle (.pkl) 文件。
    // 在实际项目中，建议在 Python 中将 category_vectors
    // 导出为简单的二进制浮点数组 (.bin) 此处为了在 C++
    // 中跑通完整流程，我们从预先准备的二进制文件读取向量 您可以运行: conda run
    // -n bge_test python -c "import pickle; import numpy as np;
    // f=open('../../../BAAI_bge-large-zh-v1.5.pkl', 'rb'); v=pickle.load(f);
    // v.astype(np.float32).tofile('../../../category_vectors.bin')"
    const QString vectors_path = resolveResourcePath(vectors_rel_path);
    qDebug() << "正在加载类别向量数据:" << vectors_path;
    std::vector<std::vector<float>> category_vectors;
    QFile vec_file(vectors_path);
    if (vec_file.open(QIODevice::ReadOnly)) {
      QByteArray data = vec_file.readAll();
      int num_vectors = categories.size();
      int hidden_size = 1024;

      if (data.size() == num_vectors * hidden_size * sizeof(float)) {
        const float *raw_data =
            reinterpret_cast<const float *>(data.constData());
        for (int i = 0; i < num_vectors; ++i) {
          std::vector<float> vec(raw_data + i * hidden_size,
                                 raw_data + (i + 1) * hidden_size);
          category_vectors.push_back(vec);
        }
        qDebug() << "类别向量加载成功! 形状: (" << category_vectors.size()
                 << "," << hidden_size << ")";
      } else {
        qCritical() << "类别向量数据大小不匹配! 文件大小:" << data.size()
                    << "预期:" << num_vectors * hidden_size * sizeof(float);
      }
      vec_file.close();
    } else {
      qCritical() << "加载 category_vectors.bin 失败! 请确保已使用 Python 将 "
                     "pkl 转换为 bin 格式。";
    }

    const QString resolved_model_path = resolveResourcePath(model_path_str);
    qDebug() << "正在加载模型:" << resolved_model_path;
    BgeOnnxEngine engine(resolved_model_path, vocab_path);
    qDebug() << "模型加载成功!\n";

    // 真实文本测试集
    QStringList test_titles = {
        QString::fromUtf8("苹果手机 iPhone 15 Pro Max 256GB 钛金属"),
        QString::fromUtf8("阿迪达斯官网 adidas 男子 跑步鞋 运动鞋"),
        QString::fromUtf8("欧莱雅男士洗面奶 控油抗痘 火山岩 洁面膏"),
        QString::fromUtf8("小米手环 9 NFC版 智能运动手环 心率睡眠监测"),
        QString::fromUtf8("华为路由器 AX3 Pro 双核WiFi6 千兆家用无线"),
        QString::fromUtf8("联想拯救者 R9000P 16英寸 R7-7845HX 32G 1T RTX4060"),
        QString::fromUtf8("戴森 Dyson 吹风机 HD15 负离子护发 速干造型"),
        QString::fromUtf8("海尔 10公斤 滚筒洗衣机 变频全自动 以旧换新"),
        QString::fromUtf8("九阳破壁机 家用加热豆浆机 预约免滤料理机"),
        QString::fromUtf8("三只松鼠 坚果礼盒 每日坚果 混合坚果零食大礼包"),
        QString::fromUtf8("伊利 金典 纯牛奶 250ml*24盒 整箱"),
        QString::fromUtf8("十月稻田 东北大米 5kg 稻花香2号 真空包装"),
        QString::fromUtf8("花王 妙而舒 纸尿裤 M号 婴儿尿不湿 超薄透气"),
        QString::fromUtf8("帮宝适 拉拉裤 XL码 婴儿学步裤"),
        QString::fromUtf8("皇家 CANIN 成猫粮 2kg 室内猫粮"),
        QString::fromUtf8("美的 空调 1.5匹 变频冷暖 挂机 一级能效"),
        QString::fromUtf8("得力 0.5mm 中性笔 黑色签字笔 12支装"),
        QString::fromUtf8("晨光 A4 复印纸 70g 500张*5包 办公用纸"),
        QString::fromUtf8("乐高 LEGO 机械组 科技系列 拼装积木"),
        QString::fromUtf8("苹果 AirPods Pro 2代 USB-C 主动降噪"),
        QString::fromUtf8("周大福 足金戒指 999黄金 女款"),
        QString::fromUtf8("雷朋 Ray-Ban 太阳镜 偏光墨镜 经典飞行员款"),
        QString::fromUtf8("车载充电器 120W 快充 点烟器一拖二 带Type-C"),
        QString::fromUtf8("汽车机油 全合成 5W-30 4L SN级"),
        QString::fromUtf8("宜家 IKE A 书桌 简约电脑桌 白色 120cm"),
        QString::fromUtf8("耐克 NIKE 男子 速干短袖 运动T恤"),
        QString::fromUtf8("Kindle 电子书阅读器 6英寸 电子墨水屏")};

    qDebug() << "开始进行推理性能测试...";
    QElapsedTimer timer;

    QStringList current_batch_titles;
    QList<QStringList> current_batch_candidates;

    for (const QString &test_title : test_titles) {
      qDebug() << "\n--- 测试标题:" << test_title << "---";

      // 添加 Instruct Prompt (指令提示) 以提升 BGE 模型的检索与分类准确率
      QString prompt =
          QString::fromUtf8("为这个商品标题寻找对应的电商商品类别：");
      QString query_with_prompt = prompt + test_title;

      // 1. 分词计时
      timer.start();
      BertTokenizer::Encoding encoding =
          engine.tokenizer().encodeForModel(query_with_prompt, 512);
      qint64 tokenize_time = timer.nsecsElapsed();
      qDebug() << "分词耗时:" << tokenize_time / 1000000.0 << "ms";

      // 打印 Token IDs (可注释以避免输出过多)
      // QString ids_str;
      // for(auto id : input_ids) ids_str += QString::number(id) + " ";
      // qDebug() << "Token IDs:" << ids_str;

      // 2. ONNX 推理计时
      timer.start();
      std::vector<float> cls_embedding = engine.embedEncoding(encoding);
      qint64 infer_time = timer.nsecsElapsed();

      qDebug() << "推理及提取耗时:" << infer_time / 1000000.0 << "ms";

      // 4. 计算余弦相似度并排序前10个
      if (!category_vectors.empty()) {
        timer.start();
        std::vector<std::pair<float, size_t>> sim_scores;
        sim_scores.reserve(category_vectors.size());

        for (size_t i = 0; i < category_vectors.size(); ++i) {
          float sim = cosine_similarity(cls_embedding, category_vectors[i]);
          sim_scores.push_back({sim, i});
        }

        // 对分数进行降序排序
        std::partial_sort(sim_scores.begin(), sim_scores.begin() + top_k,
                          sim_scores.end(),
                          [](const std::pair<float, size_t> &a,
                             const std::pair<float, size_t> &b) {
                            return a.first > b.first;
                          });

        qint64 sim_time = timer.nsecsElapsed();
        qDebug() << "相似度计算及排序耗时:" << sim_time / 1000000.0 << "ms";
        qDebug() << "总耗时:"
                 << (tokenize_time + infer_time + sim_time) / 1000000.0 << "ms";

        qDebug() << "Top-" << top_k << " 匹配结果:";
        QStringList top_candidates;
        for (int i = 0; i < top_k; ++i) {
          float score = sim_scores[i].first;
          size_t idx = sim_scores[i].second;
          QString cat = (idx < static_cast<size_t>(categories.size()))
                            ? categories[idx]
                            : "未知类别";
          qDebug().noquote() << QString("  %1. [%2] %3")
                                    .arg(i + 1, 2)
                                    .arg(score, 0, 'f', 4)
                                    .arg(cat);
          top_candidates.append(cat);
        }

        current_batch_titles.append(test_title);
        current_batch_candidates.append(top_candidates);

        // 如果达到 batch_size 或者是最后一条记录，则调用批量分类推理
        if (current_batch_titles.size() >= batch_size ||
            test_title == test_titles.last()) {
          qDebug() << "\n正在调用 AI LLM 进行批量分类 (当前批次大小:"
                   << current_batch_titles.size() << ")...";
          qint64 ai_time = 0;
          // 调用 AI LLM 进行批量分类决断
          QList<QJsonObject> results =
              call_ai_batch(api_base, ai_model_name, current_batch_titles,
                            current_batch_candidates, top_k, ai_time);

          qDebug() << "AI 决断耗时:" << ai_time / 1000.0 << "s";
          qDebug() << "批量分类结果:";
          for (const auto &res : results) {
            qDebug().noquote()
                << QString("  [标题] %1").arg(res["title"].toString());
            qDebug().noquote()
                << QString("  [分类] %1\n").arg(res["category"].toString());
          }

          // 清空当前批次
          current_batch_titles.clear();
          current_batch_candidates.clear();
        }
      } else {
        qDebug() << "类别向量为空，跳过相似度计算。总耗时:"
                 << (tokenize_time + infer_time) / 1000000.0 << "ms";
      }
    }
  } catch (const std::exception &e) {
    qCritical() << "系统异常:" << e.what();
  }

  return 0;
}
