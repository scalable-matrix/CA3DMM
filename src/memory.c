#include <stdio.h>

#include "memory.h"
#include <string.h>
#include <assert.h>

#if USE_GPU
#include "gpu.h"
#endif

#ifdef __cplusplus
extern "C" { //}
#endif

void check_device(device_type dev) {
    if(dev < DEVICE_TYPE_HOST || dev > DEVICE_TYPE_HOST_PINNED) {
        fprintf(stderr, "Unknown dev\n");
        assert(-1 && "Unknown device"); 
    } else {
#if DEBUG
        printf("Valid dev: %i\n", dev);
#endif
    }
}

void* _OUR_MALLOC(size_t bytes, device_type t ) {
  check_device(t);
  double* mem = NULL;
#if DEBUG
  printf("Mallocing %i bytes\n", bytes);
#endif
  if (t == DEVICE_TYPE_HOST) {
    mem = malloc(bytes);

  } 
else if (t == DEVICE_TYPE_DEVICE) {
#if USE_GPU
    gpuErrchk(cudaMalloc((void**) &mem, bytes));
#else
    mem = malloc(bytes);
#endif
  }
else if (t == DEVICE_TYPE_HOST_PINNED) {
#if USE_GPU
    gpuErrchk(cudaMallocHost((void**) &mem, bytes));
#else
    mem = malloc(bytes);
#endif
} else
{
    printf("Unknown device %i\n", t);
    assert(0);
    exit(-1);
}


  if ((mem == NULL) && (bytes != 0)) {
    printf("ERR MALLOCING %li bytes on %i\n", bytes, t);
    assert(0);
    exit(-1);
  }
  return mem;
}

void* _OUR_CALLOC(int count, size_t ele_size, device_type t ) {
  check_device(t);
  double* mem = NULL;
  if (t == DEVICE_TYPE_HOST) {
    mem = calloc(count, ele_size);
  } 
else if (t == DEVICE_TYPE_DEVICE) {
#if USE_GPU
    gpuErrchk(cudaMalloc((void**) &mem, count * ele_size));
    gpuErrchk(cudaMemset(mem, 0, ele_size * count));
#else
    mem = calloc(count, ele_size);
#endif
  }
else if (t == DEVICE_TYPE_HOST_PINNED) {
#if USE_GPU
    gpuErrchk(cudaMallocHost((void**) &mem, count * ele_size));
    memset(mem, 0, count * ele_size);
#else
    mem = calloc(count, ele_size);
#endif
}else
{
    printf("Unknown device %i\n", t);
    assert(0);
    exit(-1);
}


  if (mem == NULL) {
    printf("ERR CALLOCING\n");
    exit(-1);
  }
  return mem;
}

void _OUR_FREE(void * mem , device_type t ) {
  int nullflag = 0;
  if(mem == NULL) {
    nullflag = 1;
  }

  if (t == DEVICE_TYPE_HOST) {
    free(mem);
  } 
else if (t == DEVICE_TYPE_DEVICE) {
#if USE_GPU
    gpuErrchk(cudaFree(mem));
#else
    free(mem);
#endif
  }
else if (t == DEVICE_TYPE_HOST_PINNED) {
#if USE_GPU
    gpuErrchk(cudaFreeHost(mem));
#else
    free(mem);
#endif
  }else
{
    printf("Unknown device %i\n", t);
    assert(0);
    exit(-1);
}


  if ((!nullflag) && (mem == NULL)) {
    printf("ERR FREEING\n");
    exit(-1);
  }
}

void _OUR_MEMSET(void* ptr, int value, size_t num_bytes, device_type device) {
  check_device(device);
    if(device == DEVICE_TYPE_DEVICE) {
#if USE_GPU
        gpuErrchk(cudaMemset(ptr, value, num_bytes));
#else
        memset(ptr, value, num_bytes);
#endif
    } else {
        memset(ptr, value, num_bytes);
    }

}

void _OUR_MEMCPY(void* dst, const void* src, size_t bytes, device_type dst_dev, device_type src_dev) {
  check_device(dst_dev);
  check_device(src_dev);
#if USE_GPU
  if (src_dev == DEVICE_TYPE_HOST_PINNED) {
    src_dev = DEVICE_TYPE_HOST;
  }
  if (dst_dev == DEVICE_TYPE_HOST_PINNED) {
    dst_dev = DEVICE_TYPE_HOST;
  }
  if (dst_dev == DEVICE_TYPE_HOST && src_dev == DEVICE_TYPE_HOST) {
    memcpy(dst, src, bytes);
  } else if (dst_dev == DEVICE_TYPE_DEVICE && src_dev == DEVICE_TYPE_HOST) {
      gpuErrchk(cudaMemcpy(dst, (void*) (src), bytes, cudaMemcpyHostToDevice));
  } else if (dst_dev == DEVICE_TYPE_DEVICE && src_dev == DEVICE_TYPE_DEVICE) {
      gpuErrchk(cudaMemcpy(dst, (void*) (src), bytes, cudaMemcpyDeviceToDevice));
  } else if (dst_dev == DEVICE_TYPE_HOST && src_dev == DEVICE_TYPE_DEVICE) {
      gpuErrchk(cudaMemcpy(dst, (void*) (src), bytes, cudaMemcpyDeviceToHost));
  } else
{
    printf("Unknown device pair %i, %i\n", dst_dev, src_dev);
    assert(0);
    exit(-1);
}

#else
    memcpy(dst, src, bytes);
#endif
}


#ifdef __cplusplus
}
#endif
