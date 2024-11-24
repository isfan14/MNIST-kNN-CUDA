#include "mnist.hpp"

__global__ void GetPixelDistance(int *train_images_pixels, int *test_images_pixels, float *distances, int &test_index)
{
  __shared__ float sum;
  sum = 0;
  __syncthreads();

  int train_index = blockIdx.x;
  int pixel_index_x = threadIdx.x;
  int pixel_index_y = threadIdx.y;

  int train_pixel_index = train_index * IMAGE_W * IMAGE_L + pixel_index_x * IMAGE_W + pixel_index_y;
  int test_pixel_index = test_index * IMAGE_W * IMAGE_L + pixel_index_x * IMAGE_W + pixel_index_y;
  int test_distance_index = test_index * N_TRAIN + train_index;

  int trainPixel = train_images_pixels[train_pixel_index];
  int testPixel = test_images_pixels[test_pixel_index];

  int diff = trainPixel - testPixel;

  sum += diff * diff;
  __syncthreads();

  distances[test_distance_index] = sqrt(sum);
}

int main(int argc, char *argv[])
{
  int *h_train_images_pixels, *h_test_images_pixels, *d_train_images_pixels, *d_test_images_pixels;
  char *h_train_labels, *h_test_labels, *d_train_labels, *d_test_labels;
  float *h_test_distances, *d_test_distances;

  // allocate host memory to hold mnist data
  cudaMallocHost(&h_train_images_pixels, N_TRAIN * IMAGE_L * IMAGE_W * sizeof(int));
  cudaMallocHost(&h_test_images_pixels, N_TEST * IMAGE_L * IMAGE_W * sizeof(int));
  cudaMallocHost(&h_train_labels, N_TRAIN * sizeof(char));
  cudaMallocHost(&h_test_labels, N_TEST * sizeof(char));

  // load mnist train data to host
  std::cout << "loading train data..." << std::endl;
  read_mnist("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", h_train_images_pixels, h_train_labels);
  std::cout << std::endl;

  // load mnist test data to host
  std::cout << "loading test data..." << std::endl;
  read_mnist("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", h_train_images_pixels, h_test_labels);

  // allocate device memory to hold mnist data
  cudaMalloc(&d_train_images_pixels, N_TRAIN * IMAGE_L * IMAGE_W * sizeof(int));
  cudaMalloc(&d_test_images_pixels, N_TEST * IMAGE_L * IMAGE_W * sizeof(int));
  cudaMalloc(&d_train_labels, N_TRAIN * sizeof(char));
  cudaMalloc(&d_test_labels, N_TEST * sizeof(char));

  // copy loaded host mnist data to device memory
  cudaMemcpy(d_train_images_pixels, d_train_images_pixels, sizeof(*d_train_images_pixels), cudaMemcpyHostToDevice);
  cudaMemcpy(d_train_labels, d_train_labels, sizeof(*d_train_labels), cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_images_pixels, d_test_images_pixels, sizeof(*d_test_images_pixels), cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_labels, d_test_labels, sizeof(*d_test_labels), cudaMemcpyHostToDevice);

  // free mnist data from host memory
  cudaFreeHost(&h_train_images_pixels);
  cudaFreeHost(&h_test_images_pixels);
  cudaFreeHost(&h_train_labels);
  cudaFreeHost(&h_test_labels);

  // allocate device memory to hold distances data
  cudaMalloc(&d_test_distances, N_TEST * N_TRAIN * sizeof(float));

  // define number of blocks
  // each block handle 1 train image
  int numBlocks = N_TRAIN;

  // define number of threads per block
  // using 2d threads correspond to each pixel in a train image
  // each thread handle 1 pixel of distance calculation
  dim3 threadsPerBlock = dim3(IMAGE_L, IMAGE_W);

  // loop through all test data to calculate the distance
  for (int i = 0; i < N_TEST; i++)
  {
    GetPixelDistance<<<numBlocks, threadsPerBlock>>>(d_train_images_pixels, d_test_images_pixels, d_test_distances, i);
  }

  // free mnist data from device memory
  cudaFree(&d_train_images_pixels);
  cudaFree(&d_test_images_pixels);
  cudaFree(&d_train_labels);
  cudaFree(&d_test_labels);

  // allocate host memory to hold distances data
  cudaMallocHost(&h_test_distances, N_TEST * N_TRAIN * sizeof(float));

  // copy calculated distances data from device to host memory
  cudaMemcpy(h_test_distances, d_test_distances, sizeof(d_test_distances), cudaMemcpyDeviceToHost);

  // free distances data from device memory
  cudaFree(&d_test_distances);

  // TODO: get inferred label based on the calculated distances

  // free distances data from host memory
  cudaFreeHost(&h_test_distances);
}
