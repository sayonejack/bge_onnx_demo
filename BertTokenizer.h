#pragma once

#include <QHash>
#include <QSet>
#include <QString>
#include <QStringList>
#include <QtGlobal>
#include <vector>

#if defined(BGECORE_LIBRARY)
#define BGECORE_EXPORT Q_DECL_EXPORT
#else
#define BGECORE_EXPORT Q_DECL_IMPORT
#endif

class BGECORE_EXPORT BertTokenizer {
public:
  struct Encoding {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
  };

  struct EncodeOptions {
    int max_length = 512;
    bool pad_to_max_length = false;
  };

  explicit BertTokenizer(const QString &vocab_path);

  std::vector<int64_t> encode(const QString &text, int max_length = 512) const;
  Encoding encodeForModel(const QString &text, int max_length = 512) const;
  Encoding encodeForModel(const QString &text,
                          const EncodeOptions &options) const;

  std::pair<std::vector<int64_t>, std::vector<int64_t>>
  encodePair(const QString &text_a, const QString &text_b,
             int max_length = 512) const;

  Encoding encodePairForModel(const QString &text_a, const QString &text_b,
                              int max_length = 512) const;
  Encoding encodePairForModel(const QString &text_a, const QString &text_b,
                              const EncodeOptions &options) const;

private:
  QHash<QString, int64_t> vocab;
  int64_t unk_token_id = 100;
  int64_t cls_token_id = 101;
  int64_t sep_token_id = 102;
  int64_t pad_token_id = 0;
  bool do_lower_case = true;
  bool tokenize_chinese_chars = true;
  bool strip_accents_enabled = true;
  int max_input_chars_per_word = 100;
  QSet<QString> never_split;

  std::vector<int64_t> encodeSingle(const QString &text, int max_length) const;
  std::vector<int64_t> encodeText(const QString &text) const;
  void appendTokens(std::vector<int64_t> &input_ids,
                    std::vector<int64_t> *token_type_ids,
                    const std::vector<int64_t> &tokens, int type_id) const;
  void truncatePair(std::vector<int64_t> &ids_a, std::vector<int64_t> &ids_b,
                    int max_length) const;
  void truncateSingleBody(std::vector<int64_t> &body_ids, int max_length) const;
  void padEncoding(Encoding &encoding, int max_length) const;
  int64_t tokenId(const QString &token, int64_t fallback) const;

  QStringList basicTokenize(const QString &text) const;
  QString cleanText(const QString &text) const;
  QString tokenizeChineseChars(const QString &text) const;
  QString stripAccents(const QString &text) const;
  QStringList whitespaceTokenize(const QString &text) const;
  QStringList splitOnPunctuation(const QString &token) const;
  std::vector<int64_t> wordpieceTokenize(const QString &token) const;
  bool isCjk(uint u) const;
  bool isPunctuation(uint u) const;
  bool isWhitespace(uint u) const;
  bool isControl(uint u) const;
};
