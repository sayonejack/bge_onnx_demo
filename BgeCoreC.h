#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(BGECORE_LIBRARY)
#define BGECORE_C_EXPORT __declspec(dllexport)
#else
#define BGECORE_C_EXPORT __declspec(dllimport)
#endif
#else
#if defined(BGECORE_LIBRARY)
#define BGECORE_C_EXPORT __attribute__((visibility("default")))
#else
#define BGECORE_C_EXPORT
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for the engine instance.
typedef struct BgeEngineHandle BgeEngineHandle;

// Create an engine from UTF-8 encoded model and vocab paths.
BGECORE_C_EXPORT BgeEngineHandle *bge_engine_create(const char *model_path_utf8,
                                                    const char *vocab_path_utf8);
// Destroy an engine handle created by bge_engine_create().
BGECORE_C_EXPORT void bge_engine_destroy(BgeEngineHandle *handle);

// Embed a single UTF-8 sentence. The returned array must be freed with
// bge_engine_free_float_array().
BGECORE_C_EXPORT float *bge_engine_embed_text(BgeEngineHandle *handle,
                                              const char *text_utf8,
                                              int max_length,
                                              int *out_dim);
// Embed a UTF-8 sentence pair. The returned array must be freed with
// bge_engine_free_float_array().
BGECORE_C_EXPORT float *bge_engine_embed_pair(BgeEngineHandle *handle,
                                              const char *text_a_utf8,
                                              const char *text_b_utf8,
                                              int max_length,
                                              int *out_dim);

// Return the model embedding dimension, or 0 on error.
BGECORE_C_EXPORT int bge_engine_get_embedding_dim(BgeEngineHandle *handle);
// Embed a single UTF-8 sentence into caller-provided buffer.
// Returns 1 on success, 0 on failure or insufficient capacity.
BGECORE_C_EXPORT int bge_engine_embed_text_to_buffer(BgeEngineHandle *handle,
                                                     const char *text_utf8,
                                                     int max_length,
                                                     float *out_buffer,
                                                     int out_capacity,
                                                     int *out_dim);
// Embed a UTF-8 sentence pair into caller-provided buffer.
// Returns 1 on success, 0 on failure or insufficient capacity.
BGECORE_C_EXPORT int bge_engine_embed_pair_to_buffer(BgeEngineHandle *handle,
                                                     const char *text_a_utf8,
                                                     const char *text_b_utf8,
                                                     int max_length,
                                                     float *out_buffer,
                                                     int out_capacity,
                                                     int *out_dim);
// Free arrays returned by bge_engine_embed_text() / bge_engine_embed_pair().
BGECORE_C_EXPORT void bge_engine_free_float_array(float *ptr);

#ifdef __cplusplus
} // extern "C"
#endif
