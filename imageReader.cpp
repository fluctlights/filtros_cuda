#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
//#include <string>

using namespace cv;

// Comprobar errores
void testCudaError(cudaError_t errorCode)
{
  if (errorCode != cudaSuccess)
  {
    printf("Error: %s", cudaGetErrorString(errorCode));
  }
}

// Imagenes de entrada y salida
cv::Mat imageInput;
cv::Mat imageOutput;

cv::Mat inImage;
cv::Mat outImage;

uchar4 *d_input_image__;
uchar4 *d_output_image__;

unsigned char *d_in_image__;
unsigned char *d_out_image__;

// Getters para filas y columnas
int numRows() { return imageInput.rows; }
int numCols() { return imageInput.cols; }

void abrirImagen(const std::string &filename){
  // Leer imagen BGR
  cv::Mat image;
  image = cv::imread(filename.c_str(), IMREAD_COLOR);
  inImage = cv::imread(filename.c_str(), CV_8UC1);
  if (image.empty())
  {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  // Transformar BGR a RGBA
  cv::cvtColor(image, imageInput, COLOR_BGR2RGBA);
}

// Cargar imagen de salida como nueva entrada para aplicar nuevo filtro
void loadImage(uchar4 **h_input_image, uchar4 **h_output_image, uchar4 **d_input_image, uchar4 **d_output_image)
{
  // Alojar memoria para imagen de salida
  imageOutput.create(imageInput.rows, imageInput.cols, imageInput.type());

  // Obtener los punteros a las imagenes
  *h_input_image = (uchar4 *)imageInput.ptr<unsigned char>(0);
  *h_output_image = (uchar4 *)imageOutput.ptr<unsigned char>(0);

  // Alojar memoria en el device para imagenes de entrada y salida
  const int numPixels = numRows() * numCols();

  testCudaError(cudaMalloc(d_input_image, sizeof(uchar4) * numPixels));
  testCudaError(cudaMalloc(d_output_image, sizeof(uchar4) * numPixels));
  testCudaError(cudaMemset(*d_output_image, 0, numPixels * sizeof(uchar4)));

  // Copiar imagen de entrada al device (GPU)
  testCudaError(cudaMemcpy(*d_input_image, *h_input_image, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
  d_input_image__ = *d_input_image;
  d_output_image__ = *d_output_image;
}


// Cargar imagen de salida como nueva entrada para aplicar nuevo filtro
void loadSobel(unsigned char **h_in, unsigned char **h_out, unsigned char **d_in, unsigned char **d_out)
{
  // Alojar memoria para imagen de salida
  outImage.create(inImage.rows, inImage.cols, CV_8UC1);

  // Obtener los punteros a las imagenes
  *h_in = inImage.ptr<uchar>(0);
  *h_out = outImage.ptr<unsigned char>(0);

  // Alojar memoria en el device para imagenes de entrada y salida
  const int numPixels = numRows() * numCols();

  testCudaError(cudaMalloc(d_in, sizeof(unsigned char) * numPixels));
  testCudaError(cudaMalloc(d_out, sizeof(unsigned char) * numPixels));
  testCudaError(cudaMemset(*d_out, 0, numPixels * sizeof(unsigned char)));

  // Copiar imagen de entrada al device (GPU)
  testCudaError(cudaMemcpy(*d_in, *h_in, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
  d_in_image__ = *d_in;
  d_out_image__ = *d_out;
}

