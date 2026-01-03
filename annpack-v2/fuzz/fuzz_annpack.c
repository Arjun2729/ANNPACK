#include <stddef.h>
#include <stdint.h>

int ann_fuzz_parse(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    ann_fuzz_parse(data, size);
    return 0;
}
