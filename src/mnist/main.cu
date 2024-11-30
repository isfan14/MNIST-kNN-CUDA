#include <cmath>

#include "../utils/mnist.hpp"
#include "../utils/helper.cuh"

__global__ void compute_diff(int *train_images, int *test_images, float *diffs, int test_idx)
{
  int train_idx = blockIdx.x;
  int pixel_idx = threadIdx.x;

  int train_pixel_idx = train_idx * IMAGE_SIZE + pixel_idx;
  int test_pixel_idx = test_idx * IMAGE_SIZE + pixel_idx;
  int diff_idx = train_idx * IMAGE_SIZE + pixel_idx;

  int diff = train_images[train_pixel_idx] - test_images[test_pixel_idx];

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
  int *h_train_images_pixels, *h_test_images_pixels, *d_train_images_pixels, *d_test_images_pixels;
  int *h_train_labels, *h_test_labels;

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

  // allocate device memory to hold mnist data
  cudaMalloc((void **)&d_train_images_pixels, TRAIN_IMAGES_SIZE);
  cudaMalloc((void **)&d_test_images_pixels, TEST_IMAGES_SIZE);

  // copy loaded host mnist data to device memory
  cudaMemcpy(d_train_images_pixels, h_train_images_pixels, TRAIN_IMAGES_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_images_pixels, h_test_images_pixels, TEST_IMAGES_SIZE, cudaMemcpyHostToDevice);

  // free mnist data from host memory
  cudaFreeHost(h_train_images_pixels);
  cudaFreeHost(h_test_images_pixels);

  float *d_diffs, *d_distances, *h_distances;

  cudaMalloc((void **)&d_diffs, DIFFS_SIZE);
  cudaMalloc((void **)&d_distances, DISTANCES_SIZE);

  cudaMallocHost((void **)&h_distances, DISTANCES_SIZE);

  unsigned int true_prediction = 0;
  unsigned int best_idx;
  float best_distance;

  int label;
  int prediction;

  // define number of blocks
  // each block handle 1 train image
  int numBlocks = TRAIN_SIZE;

  // define number of threads per block
  // using 2d threads correspond to each pixel in a train image
  // each thread handle 1 pixel of distance calculation
  int threadsPerBlock = IMAGE_SIZE;

  float elapsed_time_ms;

  // some events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start to count execution time of GPU version
  cudaEventRecord(start, 0);

  // loop through all test data to calculate the distance
  for (int test_idx = 0; test_idx < TEST_SIZE; test_idx++)
  {
    // 1. compute euclidean distance (distance = sqrt(diff))
    // 1.1. compute diff (inside of sqrt)
    compute_diff<<<numBlocks, threadsPerBlock>>>(d_train_images_pixels, d_test_images_pixels, d_diffs, test_idx);
    CUDACHECK(cudaPeekAtLastError());

    // cudaDeviceSynchronize();
    // CUDACHECK(cudaPeekAtLastError());

    // 1.2. compute distance
    // compute_distance_atomic<<<numBlocks, IMAGE_SIZE>>>(d_diffs, d_distances, test_idx);
    compute_distance<<<numBlocks, 1>>>(d_diffs, d_distances, test_idx);
    CUDACHECK(cudaPeekAtLastError());

    // cudaDeviceSynchronize();
    // CUDACHECK(cudaPeekAtLastError());

    cudaMemcpy(h_distances, d_distances, DISTANCES_SIZE, cudaMemcpyDeviceToHost);
    CUDACHECK(cudaPeekAtLastError());

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
  cudaFree(d_train_images_pixels);
  cudaFree(d_test_images_pixels);

  // free distances data from host memory
  cudaFreeHost(h_train_labels);
  cudaFreeHost(h_test_labels);
}
