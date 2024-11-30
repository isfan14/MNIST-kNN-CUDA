#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define X_DATA_OFFSET 16
#define Y_DATA_OFFSET 8
#define IMAGE_L 28
#define IMAGE_W 28
#define IMAGE_SIZE 28 * 28
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

uint32_t swap_endian(uint32_t val)
{
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

void read_mnist(const char *image_filename, const char *label_filename, int *images_pixels, int *labels)
{
  // Open files
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read((char *)(&magic), 4);
  magic = swap_endian(magic);
  if (magic != 2051)
  {
    std::cout << "Incorrect image file magic: " << magic << std::endl;
    return;
  }

  label_file.read((char *)(&magic), 4);
  magic = swap_endian(magic);
  if (magic != 2049)
  {
    std::cout << "Incorrect image file magic: " << magic << std::endl;
    return;
  }

  image_file.read((char *)(&num_items), 4);
  num_items = swap_endian(num_items);
  label_file.read((char *)(&num_labels), 4);
  num_labels = swap_endian(num_labels);

  if (num_items != num_labels)
  {
    std::cout << "image file nums should equal to label num" << std::endl;
    return;
  }

  image_file.read((char *)(&rows), 4);
  rows = swap_endian(rows);
  image_file.read((char *)(&cols), 4);
  cols = swap_endian(cols);

  std::cout << "image and label num is: " << num_items << std::endl;
  std::cout << "image rows: " << rows << ", cols: " << cols << std::endl;

  for (int i = 0; i < (num_items * rows * cols); i++)
  {
    image_file.read((char *)(&images_pixels[i]), sizeof(char));
  }
  std::cout << "images loaded" << std::endl;

  for (int i = 0; i < num_items; i++)
  {
    label_file.read(((char *)&labels[i]), sizeof(char));
  }
  std::cout << "labels loaded" << std::endl;
}