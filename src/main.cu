#include <cmath>

#include "mnist.hpp"
#include "helper.cu"

__global__ void compute_diff(int *train_images_pixels, int *test_images_pixels, float *diffs, int test_index)
{
  int train_index = blockIdx.x;
  int pixel_index_x = threadIdx.x;
  int pixel_index_y = threadIdx.y;

  int train_pixel_index = train_index * IMAGE_SIZE + pixel_index_x * IMAGE_W + pixel_index_y;
  int test_pixel_index = test_index * IMAGE_SIZE + pixel_index_x * IMAGE_W + pixel_index_y;
  int diff_index = train_index * IMAGE_SIZE + pixel_index_x * IMAGE_W + pixel_index_y;

  int trainPixel = train_images_pixels[train_pixel_index];
  int testPixel = test_images_pixels[test_pixel_index];

  float diff = pow((float)(trainPixel - testPixel) / 256, 2);
  diffs[diff_index] = diff;
}

__global__ void compute_distance(float *diffs, float *distances, int test_index)
{
  int train_index = blockIdx.x;
  float sum = 0.0f;

  for (int pixel_index = 0; pixel_index < IMAGE_SIZE; pixel_index++)
  {
    sum += diffs[train_index * IMAGE_SIZE + pixel_index];
  }

  distances[train_index] = sqrt(sum);
}

int main(int argc, char *argv[])
{
  const unsigned long TRAIN_IMAGES_SIZE = TRAIN_SIZE * IMAGE_SIZE * sizeof(int);
  const unsigned long TRAIN_LABELS_SIZE = TRAIN_SIZE * sizeof(int);

  const unsigned long TEST_IMAGES_SIZE = TEST_SIZE * IMAGE_SIZE * sizeof(int);
  const unsigned long TEST_LABELS_SIZE = TEST_SIZE * sizeof(int);

  const unsigned long DIFFS_SIZE = TRAIN_SIZE * IMAGE_SIZE * sizeof(float);
  const unsigned long DISTANCES_SIZE = TRAIN_SIZE * sizeof(float);

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

  unsigned int true_prediction = 0;
  // define number of blocks
  // each block handle 1 train image
  int numBlocks = TRAIN_SIZE;

  // define number of threads per block
  // using 2d threads correspond to each pixel in a train image
  // each thread handle 1 pixel of distance calculation
  dim3 threadsPerBlock = dim3(IMAGE_L, IMAGE_W);

  // loop through all test data to calculate the distance
  for (int test_index = 0; test_index < TEST_SIZE; test_index++)
  {
    float *d_diffs;
    float *d_distances, *h_distances;

    cudaMalloc((void **)&d_diffs, DIFFS_SIZE);

    compute_diff<<<numBlocks, threadsPerBlock>>>(d_train_images_pixels, d_test_images_pixels, d_diffs, test_index);
    CUDACHECK(cudaPeekAtLastError());

    cudaDeviceSynchronize();
    CUDACHECK(cudaPeekAtLastError());

    cudaMalloc((void **)&d_distances, DISTANCES_SIZE);

    compute_distance<<<numBlocks, 1>>>(d_diffs, d_distances, test_index);
    CUDACHECK(cudaPeekAtLastError());

    cudaDeviceSynchronize();
    CUDACHECK(cudaPeekAtLastError());

    cudaMallocHost((void **)&h_distances, DISTANCES_SIZE);
    cudaMemcpy(h_distances, d_distances, DISTANCES_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_diffs);
    cudaFree(d_distances);

    unsigned int best_index = 0;
    float best_distance = h_distances[best_index];

    for (unsigned int j = 1; j < TRAIN_SIZE; j++)
    {
      float distance = h_distances[j];

      if (distance < best_distance)
      {
        best_index = j;
        best_distance = distance;
      }
    }

    int label = h_test_labels[test_index];
    int prediction = h_train_labels[best_index];

    if (label == prediction)
    {
      true_prediction++;
    }

    // std::cout << "i: " << test_index << " label: " << label << " prediction: " << prediction << " distance: " << best_distance << std::endl;

    cudaFreeHost(h_distances);
  }

  std::cout << "true predictions: " << true_prediction << " percentage: " << true_prediction / TEST_SIZE * 100 << std::endl;

  // free mnist data from device memory
  cudaFree(d_train_images_pixels);
  cudaFree(d_test_images_pixels);

  // free distances data from host memory
  cudaFreeHost(h_train_labels);
  cudaFreeHost(h_test_labels);
}
