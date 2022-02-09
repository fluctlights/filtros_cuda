#include "utils.h"
#include <stdio.h>
#include <math.h>

// GTX 1060
#define TxB 1024


#define FILTRO_BLANCO_NEGRO 1
#define FILTRO_SEPIA 2
#define FILTRO_VOLTEADO_VERTICAL 3
#define FILTRO_ESPECTRO_ROJO 4
#define FILTRO_ESPECTRO_VERDE 5
#define FILTRO_ESPECTRO_AZUL 6
#define FILTRO_SOBEL 7

#define R 1
#define G 2
#define B 3

__global__ void f_black_white_kernel(const uchar4 *const input_image, uchar4 *const output_image, int rows, int cols)
{
  long long int ix = blockIdx.x * blockDim.x + threadIdx.x;
  uchar4 px = input_image[ix];

  output_image[ix].x = (int)(px.x * 0.2986) + (int)(px.y * 0.587) + (int)(px.z * 0.114); // .x -> R
  output_image[ix].y = output_image[ix].x;                                     // .y -> G
  output_image[ix].z = output_image[ix].x;                                     // .z -> B
}

__global__ void f_sepia_kernel(const uchar4 *const input_image, uchar4 *const output_image, int rows, int cols)
{
  long long int ix = blockIdx.x * blockDim.x + threadIdx.x;
  uchar4 px = input_image[ix];
  output_image[ix].x = (int)(px.x * 0.393) + (int)(px.y * 0.769) + (int)(px.z * 0.189); // .x -> R
  if(output_image[ix].x > 255) output_image[ix].x = 255;
  output_image[ix].y = (int)(px.x * 0.349) + (int)(px.y * 0.686) + (int)(px.z * 0.168); // .y -> G
  if(output_image[ix].y > 255) output_image[ix].y = 255;
  output_image[ix].z = (int)(px.x * 0.272) + (int)(px.y * 0.534) + (int)(px.z * 0.131); // .z -> B
  if(output_image[ix].z > 255) output_image[ix].z = 255;
}

__global__ void f_volteado_vertical_kernel(const uchar4 *const input_image, uchar4 *const output_image, int rows, int cols)
{
  long long int ix = blockIdx.x * blockDim.x + threadIdx.x;
  long long int iy = threadIdx.y + blockIdx.y * blockDim.y;

  output_image[ix*cols + (cols-iy-1)] = input_image[ix*cols + iy];
}

__global__ void f_espectro_kernel(const uchar4 *const input_image, uchar4 *const output_image, int rows, int cols, int canal)
{
  long long int ix = blockIdx.x * blockDim.x + threadIdx.x;
  uchar4 px = input_image[ix];

  output_image[ix].x = 0; // .x -> R
  output_image[ix].y = 0; // .y -> G
  output_image[ix].z = 0; // .z -> B

  switch (canal)
  {
  case R: // Rojo
    output_image[ix].x = px.x;
    break;
  case G: // Verde
    output_image[ix].y = px.y;
    break;
  case B: // Azul
    output_image[ix].z = px.z;
    break;
  }
}

__global__ void f_sobel_kernel(unsigned char *dataIn, unsigned char *dataOut, int rows, int cols)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * cols + xIndex;
    int Gx = 0;
    int Gy = 0;

    if (xIndex > 0 && xIndex < cols - 1 && yIndex > 0 && yIndex < rows - 1)
    {
        Gx = dataIn[(yIndex - 1) * cols + xIndex + 1] + 2 * dataIn[yIndex * cols + xIndex + 1] + dataIn[(yIndex + 1) * cols + xIndex + 1]
            - (dataIn[(yIndex - 1) * cols + xIndex - 1] + 2 * dataIn[yIndex * cols + xIndex - 1] + dataIn[(yIndex + 1) * cols + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * cols + xIndex - 1] + 2 * dataIn[(yIndex - 1) * cols + xIndex] + dataIn[(yIndex - 1) * cols + xIndex + 1]
            - (dataIn[(yIndex + 1) * cols + xIndex - 1] + 2 * dataIn[(yIndex + 1) * cols + xIndex] + dataIn[(yIndex + 1) * cols + xIndex + 1]);
        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}

void image_filter(uchar4 *const d_input_image, uchar4 *const d_output_image, int rows, int cols, int filtro, unsigned char* const d_in, unsigned char* const d_out)
{
  long long int total_px = rows * cols;          // Numero de pixeles
  long int blocks_x_grid = ceil(total_px / TxB); // Bloques por grid
  const dim3 blockSize(TxB, 1, 1);
  const dim3 gridSize(blocks_x_grid,1,1);
  const dim3 blockSizeEspecial(32, 32, 1);
  const dim3 gridSizeEspecial((cols + blockSizeEspecial.x - 1) / blockSizeEspecial.x, (rows + blockSizeEspecial.y - 1) / blockSizeEspecial.y);
  const dim3 gridSizeEspecial2((rows + blockSizeEspecial.x - 1) / blockSizeEspecial.x, (cols + blockSizeEspecial.y - 1) / blockSizeEspecial.y);
  cudaError_t errorCode;


  // Llamada al kernel
  switch (filtro)
  {
    case FILTRO_BLANCO_NEGRO:
      f_black_white_kernel<<<gridSize, blockSize>>>(d_input_image, d_output_image, rows, cols);
      break;
    case FILTRO_SEPIA:
      f_sepia_kernel<<<gridSize, blockSize>>>(d_input_image, d_output_image, rows, cols);
      break;
    case FILTRO_VOLTEADO_VERTICAL:
      f_volteado_vertical_kernel<<<gridSizeEspecial2, blockSizeEspecial>>>(d_input_image, d_output_image, rows, cols);
      break;
    case FILTRO_ESPECTRO_ROJO:
      f_espectro_kernel<<<gridSize, blockSize>>>(d_input_image, d_output_image, rows, cols, R);
      break;
    case FILTRO_ESPECTRO_VERDE:
      f_espectro_kernel<<<gridSize, blockSize>>>(d_input_image, d_output_image, rows, cols, G);
      break;
    case FILTRO_ESPECTRO_AZUL:
      f_espectro_kernel<<<gridSize, blockSize>>>(d_input_image, d_output_image, rows, cols, B);
      break;
    case FILTRO_SOBEL:
      f_sobel_kernel <<<gridSizeEspecial, blockSizeEspecial>>>(d_in, d_out, rows, cols);
      break;
  }

  // Comprobar error
  errorCode = cudaGetLastError();
  if (errorCode != cudaSuccess)
    printf("Error: %s", cudaGetErrorString(errorCode));

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

