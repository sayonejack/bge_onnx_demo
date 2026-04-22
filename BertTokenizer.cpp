#include "BertTokenizer.h"

#include <QDebug>
#include <QFile>
#include <QTextStream>

BertTokenizer::BertTokenizer(const QString &vocab_path) {
  QFile file(vocab_path);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    qCritical() << "Failed to open vocab file:" << vocab_path;
    return;
  }
  QTextStream in(&file);
  in.setCodec("UTF-8");
  int64_t index = 0;
  while (!in.atEnd()) {
    QString line = in.readLine();
    if (!line.isEmpty() && line.endsWith('\r')) {
      line.chop(1);
    }
    vocab.insert(line, index++);
  }

  unk_token_id = tokenId("[UNK]", 100);
  cls_token_id = tokenId("[CLS]", 101);
  sep_token_id = tokenId("[SEP]", 102);
  pad_token_id = tokenId("[PAD]", 0);
  never_split = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"};
}

std::vector<int64_t> BertTokenizer::encode(const QString &text,
                                           int max_length) const {
  return encodeSingle(text, max_length);
}

BertTokenizer::Encoding BertTokenizer::encodeForModel(const QString &text,
                                                      int max_length) const {
  return encodeForModel(text, EncodeOptions{max_length, false});
}

BertTokenizer::Encoding
BertTokenizer::encodeForModel(const QString &text,
                              const EncodeOptions &options) const {
  Encoding encoding;
  encoding.input_ids = encodeSingle(text, options.max_length);
  encoding.attention_mask.assign(encoding.input_ids.size(), 1);
  encoding.token_type_ids.assign(encoding.input_ids.size(), 0);
  if (options.pad_to_max_length && options.max_length > 0 &&
      encoding.input_ids.size() < static_cast<size_t>(options.max_length)) {
    padEncoding(encoding, options.max_length);
  }
  return encoding;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
BertTokenizer::encodePair(const QString &text_a, const QString &text_b,
                          int max_length) const {
  std::vector<int64_t> ids_a = encodeText(text_a);
  std::vector<int64_t> ids_b = encodeText(text_b);
  truncatePair(ids_a, ids_b, max_length);

  std::vector<int64_t> input_ids;
  std::vector<int64_t> token_type_ids;
  input_ids.reserve(ids_a.size() + ids_b.size() + 3);
  token_type_ids.reserve(ids_a.size() + ids_b.size() + 3);

  input_ids.push_back(cls_token_id);
  token_type_ids.push_back(0);

  appendTokens(input_ids, &token_type_ids, ids_a, 0);
  input_ids.push_back(sep_token_id);
  token_type_ids.push_back(0);

  appendTokens(input_ids, &token_type_ids, ids_b, 1);
  input_ids.push_back(sep_token_id);
  token_type_ids.push_back(1);

  return {input_ids, token_type_ids};
}

BertTokenizer::Encoding
BertTokenizer::encodePairForModel(const QString &text_a, const QString &text_b,
                                  int max_length) const {
  return encodePairForModel(text_a, text_b, EncodeOptions{max_length, false});
}

BertTokenizer::Encoding
BertTokenizer::encodePairForModel(const QString &text_a, const QString &text_b,
                                  const EncodeOptions &options) const {
  Encoding encoding;
  auto pair_tokens = encodePair(text_a, text_b, options.max_length);
  encoding.input_ids = std::move(pair_tokens.first);
  encoding.token_type_ids = std::move(pair_tokens.second);
  encoding.attention_mask.assign(encoding.input_ids.size(), 1);
  if (options.pad_to_max_length && options.max_length > 0 &&
      encoding.input_ids.size() < static_cast<size_t>(options.max_length)) {
    padEncoding(encoding, options.max_length);
  }
  return encoding;
}

std::vector<int64_t> BertTokenizer::encodeSingle(const QString &text,
                                                 int max_length) const {
  const std::vector<int64_t> body_ids = encodeText(text);
  std::vector<int64_t> truncated_body = body_ids;
  truncateSingleBody(truncated_body, max_length);

  std::vector<int64_t> ids;
  ids.reserve(truncated_body.size() + 2);
  ids.push_back(cls_token_id);
  ids.insert(ids.end(), truncated_body.begin(), truncated_body.end());
  ids.push_back(sep_token_id);
  return ids;
}

std::vector<int64_t> BertTokenizer::encodeText(const QString &text) const {
  std::vector<int64_t> ids;
  const QStringList basic_tokens = basicTokenize(text);
  for (const QString &token : basic_tokens) {
    const std::vector<int64_t> sub_tokens = wordpieceTokenize(token);
    ids.insert(ids.end(), sub_tokens.begin(), sub_tokens.end());
  }
  return ids;
}

void BertTokenizer::appendTokens(std::vector<int64_t> &input_ids,
                                 std::vector<int64_t> *token_type_ids,
                                 const std::vector<int64_t> &tokens,
                                 int type_id) const {
  input_ids.insert(input_ids.end(), tokens.begin(), tokens.end());
  if (token_type_ids) {
    token_type_ids->insert(token_type_ids->end(), tokens.size(), type_id);
  }
}

void BertTokenizer::truncatePair(std::vector<int64_t> &ids_a,
                                 std::vector<int64_t> &ids_b,
                                 int max_length) const {
  if (max_length <= 0) {
    return;
  }
  const size_t special_tokens = 3; // [CLS], [SEP], [SEP]
  if (static_cast<size_t>(max_length) <= special_tokens) {
    ids_a.clear();
    ids_b.clear();
    return;
  }

  const size_t max_body = static_cast<size_t>(max_length) - special_tokens;
  while (ids_a.size() + ids_b.size() > max_body) {
    if (ids_a.size() > ids_b.size()) {
      ids_a.pop_back();
    } else {
      ids_b.pop_back();
    }
  }
}

void BertTokenizer::truncateSingleBody(std::vector<int64_t> &body_ids,
                                       int max_length) const {
  if (max_length <= 0) {
    return;
  }
  const size_t special_tokens = 2; // [CLS], [SEP]
  if (static_cast<size_t>(max_length) <= special_tokens) {
    body_ids.clear();
    return;
  }
  const size_t max_body = static_cast<size_t>(max_length) - special_tokens;
  if (body_ids.size() > max_body) {
    body_ids.resize(max_body);
  }
}

void BertTokenizer::padEncoding(Encoding &encoding, int max_length) const {
  const size_t target = static_cast<size_t>(max_length);
  if (encoding.input_ids.size() >= target) {
    return;
  }
  const size_t pad_count = target - encoding.input_ids.size();
  encoding.input_ids.insert(encoding.input_ids.end(), pad_count, pad_token_id);
  encoding.attention_mask.insert(encoding.attention_mask.end(), pad_count, 0);
  encoding.token_type_ids.insert(encoding.token_type_ids.end(), pad_count, 0);
}

int64_t BertTokenizer::tokenId(const QString &token, int64_t fallback) const {
  auto it = vocab.constFind(token);
  return it == vocab.constEnd() ? fallback : it.value();
}

QStringList BertTokenizer::basicTokenize(const QString &text) const {
  QString cleaned = cleanText(text);
  if (tokenize_chinese_chars) {
    cleaned = tokenizeChineseChars(cleaned);
  }

  QStringList whitespace_tokens = whitespaceTokenize(cleaned);
  QStringList split_tokens;
  for (const QString &token : whitespace_tokens) {
    QString cur = token;
    if (never_split.contains(cur)) {
      split_tokens.append(cur);
      continue;
    }
    if (do_lower_case) {
      cur = cur.toLower();
    }
    if (strip_accents_enabled) {
      cur = stripAccents(cur);
    }
    split_tokens.append(splitOnPunctuation(cur));
  }
  return split_tokens;
}

QString BertTokenizer::cleanText(const QString &text) const {
  QString output;
  output.reserve(text.size());
  QVector<uint> chars = text.toUcs4();
  for (uint ch : chars) {
    if (ch == 0) {
      continue;
    }
    if (isControl(ch)) {
      continue;
    }
    if (isWhitespace(ch)) {
      output += QLatin1Char(' ');
    } else {
      output += QString::fromUcs4(&ch, 1);
    }
  }
  return output;
}

QString BertTokenizer::tokenizeChineseChars(const QString &text) const {
  QString output;
  QVector<uint> chars = text.toUcs4();
  for (uint ch : chars) {
    if (isCjk(ch)) {
      output += QLatin1Char(' ');
      output += QString::fromUcs4(&ch, 1);
      output += QLatin1Char(' ');
    } else {
      output += QString::fromUcs4(&ch, 1);
    }
  }
  return output;
}

QString BertTokenizer::stripAccents(const QString &text) const {
  QString decomposed = text.normalized(QString::NormalizationForm_D);
  QString output;
  QVector<uint> chars = decomposed.toUcs4();
  for (uint ch : chars) {
    if (ch <= 0xFFFF) {
      QChar qc(static_cast<ushort>(ch));
      const auto category = qc.category();
      if (category == QChar::Mark_NonSpacing ||
          category == QChar::Mark_SpacingCombining ||
          category == QChar::Mark_Enclosing) {
        continue;
      }
      output += qc;
    } else {
      output += QString::fromUcs4(&ch, 1);
    }
  }
  return output;
}

QStringList BertTokenizer::whitespaceTokenize(const QString &text) const {
  return text.split(QLatin1Char(' '), Qt::SkipEmptyParts);
}

QStringList BertTokenizer::splitOnPunctuation(const QString &token) const {
  QStringList output;
  QString current;
  QVector<uint> chars = token.toUcs4();
  for (uint ch : chars) {
    if (isPunctuation(ch)) {
      if (!current.isEmpty()) {
        output.append(current);
        current.clear();
      }
      output.append(QString::fromUcs4(&ch, 1));
    } else {
      current += QString::fromUcs4(&ch, 1);
    }
  }
  if (!current.isEmpty()) {
    output.append(current);
  }
  return output;
}

std::vector<int64_t>
BertTokenizer::wordpieceTokenize(const QString &token) const {
  if (token.isEmpty()) {
    return {unk_token_id};
  }
  if (token.length() > max_input_chars_per_word) {
    return {unk_token_id};
  }

  std::vector<int64_t> sub_tokens;
  int start = 0;
  while (start < token.length()) {
    int end = token.length();
    QString cur_substr;
    int64_t cur_id = -1;

    while (start < end) {
      QString sub = token.mid(start, end - start);
      if (start > 0) {
        sub = "##" + sub;
      }

      auto it = vocab.constFind(sub);
      if (it != vocab.constEnd()) {
        cur_substr = sub;
        cur_id = it.value();
        break;
      }
      --end;
    }

    if (cur_substr.isEmpty()) {
      return {unk_token_id};
    }

    sub_tokens.push_back(cur_id);
    start = end;
  }

  return sub_tokens;
}

bool BertTokenizer::isCjk(uint u) const {
  return (u >= 0x4E00 && u <= 0x9FFF) || (u >= 0x3400 && u <= 0x4DBF) ||
         (u >= 0x20000 && u <= 0x2A6DF) || (u >= 0x2A700 && u <= 0x2B73F) ||
         (u >= 0x2B740 && u <= 0x2B81F) || (u >= 0x2B820 && u <= 0x2CEAF) ||
         (u >= 0xF900 && u <= 0xFAFF) || (u >= 0x2F800 && u <= 0x2FA1F);
}

bool BertTokenizer::isPunctuation(uint u) const {
  if ((u >= 33 && u <= 47) || (u >= 58 && u <= 64) || (u >= 91 && u <= 96) ||
      (u >= 123 && u <= 126)) {
    return true;
  }
  if (u <= 0xFFFF) {
    QChar qc(static_cast<ushort>(u));
    switch (qc.category()) {
    case QChar::Punctuation_Connector:
    case QChar::Punctuation_Dash:
    case QChar::Punctuation_Open:
    case QChar::Punctuation_Close:
    case QChar::Punctuation_InitialQuote:
    case QChar::Punctuation_FinalQuote:
    case QChar::Punctuation_Other:
      return true;
    default:
      break;
    }
  }
  if ((u >= 0x2000 && u <= 0x206F) || (u >= 0x2E00 && u <= 0x2E7F) ||
      (u >= 0x3000 && u <= 0x303F) || (u >= 0xFF00 && u <= 0xFF65))
    return true;
  return false;
}

bool BertTokenizer::isWhitespace(uint u) const {
  return u == ' ' || u == '\t' || u == '\n' || u == '\r' || u == 0x3000;
}

bool BertTokenizer::isControl(uint u) const {
  if (u == '\t' || u == '\n' || u == '\r')
    return false;
  return (u < 32) || (u == 127);
}
