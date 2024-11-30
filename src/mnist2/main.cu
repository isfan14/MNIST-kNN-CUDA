#include <cmath>

#include "../utils/mnist.hpp"
#include "../utils/helper.cuh"

__global__ void compute_diffs(int *test_images, int *train_images, float *diffs)
{
  int test_image_idx = blockIdx.x;
  int train_image_idx = blockIdx.y;
  int diff_image_idx = train_image_idx * TEST_SIZE + test_image_idx;

  int test_pixel_idx = test_image_idx * IMAGE_SIZE + threadIdx.y * IMAGE_W + threadIdx.x;
  int train_pixel_idx = train_image_idx * IMAGE_SIZE + threadIdx.y * IMAGE_W + threadIdx.x;
  int diff_pixel_idx = diff_image_idx * IMAGE_SIZE + threadIdx.y * IMAGE_W + threadIdx.x;

  int diff = test_images[test_pixel_idx] - train_images[train_pixel_idx];

  diffs[diff_pixel_idx] = diff * diff;
}

__global__ void compute_distances(float *diffs, float *distances)
{
  int test_image_idx = blockIdx.x;
  int train_image_idx = blockIdx.y;
  int diff_image_idx = train_image_idx * TEST_SIZE + test_image_idx;

  float sum = 0.0f;

  for (int i = 0; i < IMAGE_SIZE; i++)
  {
    sum += diffs[diff_image_idx + i];
  }

  distances[diff_image_idx] = sqrt(sum);
}

int main(int argc, char *argv[])
{
  size_t TEST_IMAGES_SIZE = TEST_SIZE * IMAGE_SIZE * sizeof(int);
  size_t TEST_LABELS_SIZE = TEST_SIZE * sizeof(int);

  size_t TRAIN_IMAGES_SIZE = TRAIN_SIZE * IMAGE_SIZE * sizeof(int);
  size_t TRAIN_LABELS_SIZE = TRAIN_SIZE * sizeof(int);

  size_t DIFFS_SIZE = TEST_SIZE * TRAIN_SIZE * IMAGE_SIZE * sizeof(float);
  size_t DISTANCES_SIZE = TEST_SIZE * TRAIN_SIZE * sizeof(float);

  int *h_train_images, *h_test_images, *d_train_images, *d_test_images;
  int *h_train_labels, *h_test_labels;

  // allocate host memory to hold mnist data
  cudaMallocHost((void **)&h_test_images, TEST_IMAGES_SIZE);
  cudaMallocHost((void **)&h_test_labels, TEST_LABELS_SIZE);
  cudaMallocHost((void **)&h_train_images, TRAIN_IMAGES_SIZE);
  cudaMallocHost((void **)&h_train_labels, TRAIN_LABELS_SIZE);

  // load mnist test data to host
  std::cout << "loading test data..." << std::endl;
  read_mnist("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", h_test_images, h_test_labels);

  // load mnist train data to host
  std::cout << "loading train data..." << std::endl;
  read_mnist("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", h_train_images, h_train_labels);
  std::cout << std::endl;

  // allocate device memory to hold mnist data
  cudaMalloc((void **)&d_test_images, TEST_IMAGES_SIZE);
  cudaMalloc((void **)&d_train_images, TRAIN_IMAGES_SIZE);

  // copy loaded host mnist data to device memory
  cudaMemcpy(d_test_images, h_test_images, TEST_IMAGES_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_train_images, h_train_images, TRAIN_IMAGES_SIZE, cudaMemcpyHostToDevice);

  // free mnist data from host memory
  cudaFreeHost(h_train_images);
  cudaFreeHost(h_test_images);

  unsigned int true_prediction = 0;
  // define number of blocks
  // using 2d blocks correspond to test size * train size
  // each block handle 1 image
  dim3 numBlocks = dim3(TEST_SIZE, TRAIN_SIZE);

  // define number of threads per block
  // using 2d threads correspond to each pixel in a train image
  // each thread handle 1 pixel of distance calculation
  dim3 threadsPerBlock = dim3(IMAGE_L, IMAGE_W);

  float *d_diffs, *d_distances, *h_distances;

  cudaMalloc((void **)&d_diffs, DIFFS_SIZE);
  cudaMalloc((void **)&d_distances, DISTANCES_SIZE);

  cudaMallocHost((void **)&h_distances, DISTANCES_SIZE);

  // 1. compute euclidean distance (distance = sqrt(diff))
  // 1.1. compute diff (inside of sqrt)
  compute_diffs<<<numBlocks, threadsPerBlock>>>(d_test_images, d_train_images, d_diffs);
  CUDACHECK(cudaPeekAtLastError());

  cudaDeviceSynchronize();
  CUDACHECK(cudaPeekAtLastError());

  // 1.2. compute distance
  compute_distances<<<numBlocks, 1>>>(d_diffs, d_distances);
  CUDACHECK(cudaPeekAtLastError());

  cudaDeviceSynchronize();
  CUDACHECK(cudaPeekAtLastError());

  cudaMemcpy(h_distances, d_distances, DISTANCES_SIZE, cudaMemcpyDeviceToHost);

  cudaFree(d_diffs);
  cudaFree(d_distances);

  for (int j = 0; j < TEST_SIZE; j++)
  {
    // 2. find the closest train image to the current test image
    int best_idx = 0;
    float best_distance = h_distances[best_idx];

    for (int i = 1; i < TRAIN_SIZE; i++)
    {
      int idx = j * TEST_SIZE + i;
      float distance = h_distances[idx];

      if (distance < best_distance)
      {
        best_idx = i;
        best_distance = distance;
      }
    }

    int label = h_test_labels[j];
    int prediction = h_train_labels[best_idx];

    if (label == prediction)
    {
      true_prediction++;
    }
  }

  // std::cout << "i: " << test_index << " label: " << label << " prediction: " << prediction << " distance: " << best_distance << std::endl;

  cudaFreeHost(h_distances);

  float percentage = true_prediction * 100.0 / TEST_SIZE;

  std::cout << "true predictions: " << true_prediction << " percentage: " << percentage << std::endl;

  // free mnist data from device memory
  cudaFree(d_train_images);
  cudaFree(d_test_images);

  // free distances data from host memory
  cudaFreeHost(h_train_labels);
  cudaFreeHost(h_test_labels);
}
