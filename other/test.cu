#include <cub/cub.cuh>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>

void sortWithSegmentedRadixSort(int* h_keys_in, int* h_values_in, int num_items, int num_segments, int* h_offsets) {
    // 设备内存指针
    int *d_keys_in, *d_keys_out;
    int *d_values_in, *d_values_out;
    int *d_offsets;

    // 分配设备内存
    cudaMalloc(&d_keys_in, num_items * sizeof(int));
    cudaMalloc(&d_keys_out, num_items * sizeof(int));
    cudaMalloc(&d_values_in, num_items * sizeof(int));
    cudaMalloc(&d_values_out, num_items * sizeof(int));
    cudaMalloc(&d_offsets, (num_segments + 1) * sizeof(int));

    // 复制数据到设备
    cudaMemcpy(d_keys_in, h_keys_in, num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, h_values_in, num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // 临时存储
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 确定临时存储大小
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    auto start = std::chrono::high_resolution_clock::now();

    // 执行分段排序
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1
    );

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "DeviceSegmentedRadixSort::SortPairs time: " << elapsed.count() << " seconds\n";

    // 释放内存
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_offsets);
    cudaFree(d_temp_storage);
}

void sortEachSegmentWithRadixSort(int* h_keys_in, int* h_values_in, int num_items, int num_segments, int* h_offsets) {
    // 设备内存指针
    int *d_keys_in, *d_keys_out;
    int *d_values_in, *d_values_out;

    // 分配设备内存
    cudaMalloc(&d_keys_in, num_items * sizeof(int));
    cudaMalloc(&d_keys_out, num_items * sizeof(int));
    cudaMalloc(&d_values_in, num_items * sizeof(int));
    cudaMalloc(&d_values_out, num_items * sizeof(int));

    // 计算最大段长度
    int max_segment_size = 0;
    for (int i = 0; i < num_segments; ++i) {
        int segment_size = h_offsets[i + 1] - h_offsets[i];
        if (segment_size > max_segment_size) {
            max_segment_size = segment_size;
        }
    }

    // 临时存储
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 确定最大段所需的临时存储大小
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        max_segment_size
    );

    // 分配一次临时存储
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    auto start = std::chrono::high_resolution_clock::now();

    // 对每个段单独排序
    for (int i = 0; i < num_segments; ++i) {
        int segment_start = h_offsets[i];
        int segment_end = h_offsets[i + 1];
        int segment_size = segment_end - segment_start;

        // 复制数据到设备
        cudaMemcpy(d_keys_in, h_keys_in + segment_start, segment_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values_in, h_values_in + segment_start, segment_size * sizeof(int), cudaMemcpyHostToDevice);

        // 执行排序
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            d_values_in, d_values_out,
            segment_size
        );

        // 复制排序结果回主机
        cudaMemcpy(h_keys_in + segment_start, d_keys_out, segment_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_values_in + segment_start, d_values_out, segment_size * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "DeviceRadixSort (per segment) time: " << elapsed.count() << " seconds\n";

    // 释放内存
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_temp_storage);
}

int main() {
    // 数据定义
    const int num_items = 50000000;     // 键值对总数
    const int num_segments = 100;       // 分段数量

    // 主机上的数据
    std::vector<int> h_keys_in(num_items);
    std::vector<int> h_values_in(num_items);
    std::vector<int> h_offsets(num_segments + 1);

    // 初始化数据
    int offset = 0;
    for (int i = 0; i < num_segments; ++i) {
        int segment_length = rand() % (num_items / num_segments) + 1; // 每段长度不等
        h_offsets[i] = offset;
        for (int j = 0; j < segment_length && offset < num_items; ++j, ++offset) {
            h_keys_in[offset] = rand() % 100; // 随机键
            h_values_in[offset] = offset;     // 值为索引
        }
    }
    h_offsets[num_segments] = num_items;

    // 使用分段排序
    sortWithSegmentedRadixSort(h_keys_in.data(), h_values_in.data(), num_items, num_segments, h_offsets.data());

    // 使用普通排序对每段单独排序
    sortEachSegmentWithRadixSort(h_keys_in.data(), h_values_in.data(), num_items, num_segments, h_offsets.data());

    return 0;
}



// nvcc -o sort test.cu -arch=sm_89
