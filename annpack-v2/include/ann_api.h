#ifndef ANN_API_H
#define ANN_API_H

#include <stdint.h>
#include "ann_format.h"

// API surface for the ANN runtime.
#define ANN_API_VERSION 1
int ann_result_size_bytes(void);
void *ann_load_index(const char *url);
void ann_free_index(void *ctx);
int ann_search(void *ctx, const float *query, ann_result_t *out_results, int k);
void ann_set_probe(int probe);
void ann_set_max_k(int cap);
void ann_set_max_scan(int cap);
const char *ann_last_error(void);
unsigned long long ann_last_scan_count(void);
int ann_get_n_lists(void *ctx);

#endif
