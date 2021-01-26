#ifndef MEMORY_H
#define MEMORY_H

#include <stdlib.h>

#include "enum.h"

#ifdef __cplusplus
extern "C" { //}
#endif


/** @brief Used to allocate on the correct device */
#define OUR_MALLOC(count, data_type, dev_type) \
   (data_type*)_OUR_MALLOC((count) * sizeof(data_type), dev_type);

/** @brief Used to allocate on the correct device */
#define OUR_CALLOC(count, data_type, dev_type) \
   (data_type*)_OUR_CALLOC((count), sizeof(data_type), dev_type);

#define OUR_FREE(mem, dev_type) _OUR_FREE(mem, dev_type);

#define OUR_MEMCPY(dst, src, nbytes, dst_dev, src_dev) _OUR_MEMCPY((dst), (src), nbytes, dst_dev, src_dev);

#define OUR_MEMSET(ptr, value, num_bytes, dev) _OUR_MEMSET( ptr, value, num_bytes, dev);

/** @brief Used to allocate on the correct device */
void* _OUR_MALLOC(size_t bytes, device_type t);

/** @brief Used to memset on the correct device */
void _OUR_MEMSET(void* ptr, int value, size_t num_bytes, device_type dev);

/** @brief Used to free on the correct device */
void _OUR_FREE(void* mem, device_type t);

/** @brief Used to allocate on the correct device */
void* _OUR_CALLOC(int count, size_t ele_size, device_type t);

/** @brief Used to copy on the correct devices */
void _OUR_MEMCPY(void* dst, const void* src, size_t bytes, device_type dst_dev, device_type src_dev);

void check_device(device_type dev);

#ifdef __cplusplus
}
#endif

#endif
