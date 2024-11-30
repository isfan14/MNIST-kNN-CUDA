#include <cmath>

#include "../utils/mnist.hpp"
#include "../utils/helper.cuh"

__global__ void compute_diff(int *train_images, int *test_images, size_t train_pitch, size_t test_pitch, float *diffs, size_t train_offset, size_t test_offset)
{
  int train_idx = blockIdx.x;
  int test_idx = blockIdx.y;
  int pixel_idx = threadIdx.x;

  int *train_image = (int *)((char *)train_images + (train_offset + train_idx) * train_pitch);
  int *test_image = (int *)((char *)test_images + (test_offset + test_idx) * test_pitch);

  int diff_idx = test_idx * TRAIN_SIZE * IMAGE_SIZE + (train_offset + train_idx) * IMAGE_SIZE + pixel_idx;

  int diff = train_image[pixel_idx] - test_image[pixel_idx];

  diffs[diff_idx] = diff * diff;
}

__global__ void compute_distance(float *diffs, float *distances)
{
  int train_idx = blockIdx.x;
  int test_idx = blockIdx.y;

  float sum = 0.0;

  for (int pixel_idx = 0; pixel_idx < IMAGE_SIZE; pixel_idx++)
  {
    sum += diffs[test_idx * TRAIN_SIZE * IMAGE_SIZE + train_idx * IMAGE_SIZE + pixel_idx], 2;
  }

  distances[test_idx * TRAIN_SIZE + train_idx] = sqrt(sum);
}

__global__ void predict(float *distances, size_t *best_idxs)
{
  size_t test_idx = blockIdx.y;
  size_t tmp_best_idx = 0;

  float best_distance = distances[test_idx * TRAIN_SIZE + tmp_best_idx];

  for (size_t i = 1; i < TRAIN_SIZE; i++)
  {
    if (distances[test_idx * TRAIN_SIZE + i] < best_distance)
    {
      tmp_best_idx = i;
      best_distance = distances[test_idx * TRAIN_SIZE + tmp_best_idx];
    }
  }

  best_idxs[test_idx] = tmp_best_idx;
}

const size_t TRAIN_IMAGES_SIZE = TRAIN_SIZE * IMAGE_SIZE * sizeof(int);
const size_t TRAIN_LABELS_SIZE = TRAIN_SIZE * sizeof(int);

const size_t TEST_IMAGES_SIZE = TEST_SIZE * IMAGE_SIZE * sizeof(int);
const size_t TEST_LABELS_SIZE = TEST_SIZE * sizeof(int);

const size_t DIFFS_SIZE = TRAIN_SIZE * IMAGE_SIZE * sizeof(float);
const size_t DISTANCES_SIZE = TRAIN_SIZE * sizeof(float);

int main(int argc, char *argv[])
{
  int *h_train_images_pixels, *h_test_images_pixels, *d_train_images_pixels, *d_test_images_pixels;
  int *h_train_labels, *h_test_labels;
  size_t train_pitch, test_pitch;

  float *d_diffs, *d_distances;

  size_t *d_best_idxs, *h_best_idxs;

  int label;
  int prediction;
  size_t true_predictions = 0;

  // allocate host memory to hold mnist data
  cudaMallocHost((void **)&h_train_images_pixels, TRAIN_IMAGES_SIZE);
  cudaMallocHost((void **)&h_train_labels, TRAIN_LABELS_SIZE);
  cudaMallocHost((void **)&h_test_images_pixels, TEST_IMAGES_SIZE);
  cudaMallocHost((void **)&h_test_labels, TEST_LABELS_SIZE);

  // load mnist train data to host
  std::cout << "loading train data..." << std::endl;
  read_mnist("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", h_train_images_pixels, h_train_labels);
  std::cout << std::endl;

  // load mnist test data to host
  std::cout << "loading test data..." << std::endl;
  read_mnist("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", h_test_images_pixels, h_test_labels);
  std::cout << std::endl;

  // copy loaded host mnist data to device memory
  std::cout << "loading train data to device memory..." << std::endl;
  // cudaMalloc((void **)&d_train_images_pixels, TRAIN_IMAGES_SIZE);
  // cudaMemcpy(d_train_images_pixels, h_train_images_pixels, TRAIN_IMAGES_SIZE, cudaMemcpyHostToDevice);
  cudaMallocPitch(&d_train_images_pixels, &train_pitch, IMAGE_SIZE * sizeof(int), TRAIN_SIZE);
  cudaMemcpy2D(d_train_images_pixels, train_pitch, h_train_images_pixels, IMAGE_SIZE * sizeof(int), IMAGE_SIZE * sizeof(int), TRAIN_SIZE, cudaMemcpyHostToDevice);
  cudaFreeHost(h_train_images_pixels);
  std::cout << std::endl;

  std::cout << "loading test data to device memory..." << std::endl;
  // cudaMalloc((void **)&d_test_images_pixels, TEST_IMAGES_SIZE);
  // cudaMemcpy(d_test_images_pixels, h_test_images_pixels, TEST_IMAGES_SIZE, cudaMemcpyHostToDevice);
  cudaMallocPitch(&d_test_images_pixels, &test_pitch, IMAGE_SIZE * sizeof(int), TEST_SIZE);
  cudaMemcpy2D(d_test_images_pixels, test_pitch, h_test_images_pixels, IMAGE_SIZE * sizeof(int), IMAGE_SIZE * sizeof(int), TEST_SIZE, cudaMemcpyHostToDevice);
  cudaFreeHost(h_test_images_pixels);
  std::cout << std::endl;

  // some events to count the execution time
  cudaEvent_t start_event, diff_start, distance_start, predict_start, end_event;

  cudaEventCreate(&start_event);
  cudaEventCreate(&diff_start);
  cudaEventCreate(&distance_start);
  cudaEventCreate(&predict_start);
  cudaEventCreate(&end_event);

  float elapsed_time_ms;

  cudaMalloc((void **)&d_diffs, TEST_BATCH_SIZE * DIFFS_SIZE);
  cudaMalloc((void **)&d_distances, TEST_BATCH_SIZE * DISTANCES_SIZE);
  cudaMalloc((void **)&d_best_idxs, TEST_BATCH_SIZE * sizeof(size_t));
  cudaMallocHost((void **)&h_best_idxs, TEST_BATCH_SIZE * sizeof(size_t));

  // int batch = 0;
  cudaEventRecord(start_event, 0);
  for (size_t test_offset = 0; test_offset < TEST_SIZE; test_offset += TEST_BATCH_SIZE)
  {
    // std::cout << "batch: " << batch++ << " offset: " << test_offset << std::endl;
    // 1. compute euclidean distance (distance = sqrt(diff))
    // 1.1. compute diff (inside of sqrt)
    // std::cout << "computing diffs : ";
    // cudaEventRecord(diff_start, 0);
    for (size_t train_offset = 0; train_offset < TRAIN_SIZE; train_offset += TRAIN_BATCH_SIZE)
    {
      compute_diff<<<dim3(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE), IMAGE_SIZE>>>(d_train_images_pixels, d_test_images_pixels, train_pitch, test_pitch, d_diffs, train_offset, test_offset);
      CUDACHECK(cudaPeekAtLastError());
    }

    // cudaEventRecord(distance_start, 0);
    // cudaEventSynchronize(distance_start);
    // cudaEventElapsedTime(&elapsed_time_ms, diff_start, distance_start);
    // std::cout << elapsed_time_ms << "ms" << std::endl;

    // 1.2. compute distance
    // std::cout << "computing dist. : ";
    compute_distance<<<dim3(TRAIN_SIZE, TEST_BATCH_SIZE), 1>>>(d_diffs, d_distances);
    CUDACHECK(cudaPeekAtLastError());

    // cudaEventRecord(predict_start, 0);
    // cudaEventSynchronize(predict_start);
    // cudaEventElapsedTime(&elapsed_time_ms, distance_start, predict_start);
    // std::cout << elapsed_time_ms << "ms" << std::endl;

    // 2. find the closest train image to the current test image
    cudaMemset(d_best_idxs, 0, TEST_BATCH_SIZE * sizeof(size_t));
    // std::cout << "predicting      : ";
    predict<<<dim3(1, TEST_BATCH_SIZE), 1>>>(d_distances, d_best_idxs);
    CUDACHECK(cudaPeekAtLastError());

    // cudaEventRecord(end_event, 0);
    // cudaEventSynchronize(end_event);
    // cudaEventElapsedTime(&elapsed_time_ms, predict_start, end_event);
    // std::cout << elapsed_time_ms << "ms" << std::endl;
    // std::cout << std::endl;

    cudaMemcpy(h_best_idxs, d_best_idxs, TEST_BATCH_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost);

    for (size_t test_idx = 0; test_idx < TEST_BATCH_SIZE; test_idx++)
    {
      label = h_test_labels[test_offset + test_idx];
      prediction = h_train_labels[h_best_idxs[test_idx]];
      // std::cout << test_idx << ": label: " << label << " best_idx: " << h_best_idxs[test_idx] << std::endl;
      if (label == prediction)
      {
        true_predictions++;
      }
    }
  }
  cudaEventRecord(end_event, 0);
  cudaEventSynchronize(end_event);
  cudaEventElapsedTime(&elapsed_time_ms, start_event, end_event);

  float percentage = true_predictions * 100.0 / TEST_SIZE;

  std::cout << "true predictions: " << true_predictions << " percentage: " << percentage << " elapsed: " << elapsed_time_ms << " ms" << std::endl;

  cudaFree(d_train_images_pixels);
  cudaFree(d_test_images_pixels);
  cudaFree(d_diffs);
  cudaFree(d_distances);
  cudaFree(d_best_idxs);

  cudaFreeHost(h_train_labels);
  cudaFreeHost(h_test_labels);

  cudaFreeHost(h_best_idxs);
}
