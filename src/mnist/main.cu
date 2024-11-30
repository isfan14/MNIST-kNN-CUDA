#include <cmath>

#include "../utils/mnist.hpp"
#include "../utils/helper.cuh"

__global__ void compute_diff(int *train_images, int *test_images, size_t train_pitch, size_t test_pitch, float *diffs, size_t test_idx)
{
  int train_idx = blockIdx.x;
  int pixel_idx = threadIdx.x;

  int *train_image = (int *)((char *)train_images + train_idx * train_pitch);
  int *test_image = (int *)((char *)test_images + +test_idx * test_pitch);

  int diff_idx = train_idx * IMAGE_SIZE + pixel_idx;

  int diff = train_image[pixel_idx] - test_image[pixel_idx];

  diffs[diff_idx] = diff * diff;
}

__global__ void compute_distance(float *diffs, float *distances, int test_idx)
{
  int train_idx = blockIdx.x;

  float sum = 0.0;

  for (int pixel_idx = 0; pixel_idx < IMAGE_SIZE; pixel_idx++)
  {
    sum += diffs[train_idx * IMAGE_SIZE + pixel_idx];
  }

  distances[train_idx] = sqrt(sum);
}

const size_t TRAIN_IMAGES_SIZE = TRAIN_SIZE * IMAGE_SIZE * sizeof(int);
const size_t TRAIN_LABELS_SIZE = TRAIN_SIZE * sizeof(int);

const size_t TEST_IMAGES_SIZE = TEST_SIZE * IMAGE_SIZE * sizeof(int);
const size_t TEST_LABELS_SIZE = TEST_SIZE * sizeof(int);

const size_t DIFFS_SIZE = TRAIN_SIZE * IMAGE_SIZE * sizeof(float);
const size_t DISTANCES_SIZE = TRAIN_SIZE * sizeof(float);

int main(int argc, char *argv[])
{
  int *h_train_images, *h_test_images, *d_train_images, *d_test_images;
  int *h_train_labels, *h_test_labels;

  size_t train_pitch, test_pitch;

  float *d_diffs, *d_distances, *h_distances;

  size_t best_idx;
  float best_distance;

  int label;
  int prediction;
  size_t true_prediction = 0;

  float elapsed_time_ms;

  // allocate host memory to hold mnist data
  cudaMallocHost((void **)&h_train_images, TRAIN_IMAGES_SIZE);
  cudaMallocHost((void **)&h_train_labels, TRAIN_LABELS_SIZE);
  cudaMallocHost((void **)&h_test_images, TEST_IMAGES_SIZE);
  cudaMallocHost((void **)&h_test_labels, TEST_LABELS_SIZE);

  // load mnist train data to host
  std::cout << "loading train data..." << std::endl;
  read_mnist("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", h_train_images, h_train_labels);
  std::cout << std::endl;

  // load mnist test data to host
  std::cout << "loading test data..." << std::endl;
  read_mnist("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", h_test_images, h_test_labels);

  // load mnist test data to device
  cudaMallocPitch(&d_train_images, &train_pitch, IMAGE_SIZE * sizeof(int), TRAIN_SIZE);
  cudaMemcpy2D(d_train_images, train_pitch, h_train_images, IMAGE_SIZE * sizeof(int), IMAGE_SIZE * sizeof(int), TRAIN_SIZE, cudaMemcpyHostToDevice);
  cudaFreeHost(h_train_images);

  // load mnist test data to device
  cudaMallocPitch(&d_test_images, &test_pitch, IMAGE_SIZE * sizeof(int), TEST_SIZE);
  cudaMemcpy2D(d_test_images, test_pitch, h_test_images, IMAGE_SIZE * sizeof(int), IMAGE_SIZE * sizeof(int), TEST_SIZE, cudaMemcpyHostToDevice);
  cudaFreeHost(h_test_images);

  cudaMalloc((void **)&d_diffs, DIFFS_SIZE);
  cudaMalloc((void **)&d_distances, DISTANCES_SIZE);

  cudaMallocHost((void **)&h_distances, DISTANCES_SIZE);

  // some events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start to count execution time of GPU version
  cudaEventRecord(start, 0);

  // loop through all test data to calculate the distance
  for (size_t test_idx = 0; test_idx < TEST_SIZE; test_idx++)
  {
    // 1. compute euclidean distance (distance = sqrt(diff))
    // 1.1. compute diff (inside of sqrt)
    compute_diff<<<TRAIN_SIZE, IMAGE_SIZE>>>(d_train_images, d_test_images, train_pitch, test_pitch, d_diffs, test_idx);
    // CUDACHECK(cudaPeekAtLastError());

    // 1.2. compute distance
    compute_distance<<<TRAIN_SIZE, 1>>>(d_diffs, d_distances, test_idx);
    // CUDACHECK(cudaPeekAtLastError());

    cudaMemcpy(h_distances, d_distances, DISTANCES_SIZE, cudaMemcpyDeviceToHost);

    // 2. find the closest train image to the current test image
    best_idx = 0;
    best_distance = h_distances[best_idx];

    for (unsigned int j = 1; j < TRAIN_SIZE; j++)
    {
      float distance = h_distances[j];

      if (distance < best_distance)
      {
        best_idx = j;
        best_distance = distance;
      }
    }

    label = h_test_labels[test_idx];
    prediction = h_train_labels[best_idx];

    if (label == prediction)
    {
      true_prediction++;
    }

    // std::cout << "i: " << test_idx << " label: " << label << " prediction: " << prediction << " distance: " << best_distance << std::endl;
  }

  cudaFree(d_diffs);
  cudaFree(d_distances);
  cudaFreeHost(h_distances);

  // time counting terminate
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // compute time elapse on GPU computing
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  // compute accuracy
  float percentage = true_prediction * 100.0 / TEST_SIZE;

  std::cout << "true predictions: " << true_prediction << " percentage: " << percentage << " elapsed: " << elapsed_time_ms << std::endl;

  // free mnist data from device memory
  cudaFree(d_train_images);
  cudaFree(d_test_images);

  // free distances data from host memory
  cudaFreeHost(h_train_labels);
  cudaFreeHost(h_test_labels);
}
