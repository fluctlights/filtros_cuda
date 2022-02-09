#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>
#include <cctype>
#include <ctype.h>
#include <stdlib.h>

//Incluimos el archivo encargado de obtener la imagen
#include "imageReader.cpp"

#define TITULO_VENTANA "Imagen Resultante"

#define FILTRO_BLANCO_NEGRO 1
#define FILTRO_SEPIA 2
#define FILTRO_VOLTEADO_VERTICAL 3
#define FILTRO_ESPECTRO_ROJO 4
#define FILTRO_ESPECTRO_VERDE 5
#define FILTRO_ESPECTRO_AZUL 6
#define FILTRO_SOBEL 7
#define CERRAR_PROGRAMA "Cualquier tecla"


// Funcion de kernel para el procesamiento de imagen
void image_filter(uchar4 *const d_input_image, uchar4 *const d_output_image, int rows, int cols, int filtro, unsigned char* const d_in, unsigned char* const d_out);

// Comprobar errores
void testCudaError(cudaError_t errorCode);

//Main
int main(int argc, char **argv)
{
  uchar4 *h_input_image, *h_output_image;
  uchar4 *d_input_image, *d_output_image;

  unsigned char *h_in, *d_in;
  unsigned char *h_out, *d_out;

  // Comprobar que el entorno se incia correctamente
  testCudaError(cudaFree(0));

  //Variables nombres de archivo - inicializacion
  std::string input_file;
  std::string output_file;

  //Detectar posibles errores al iniciar el contexto
  cudaGetErrorString(cudaFree(0));

  //Continuar con el programa solo si tenemos los argumentos necesarios
  switch (argc)
  {
  case 3:
    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);
    break;

  default:
    std::cerr << "Usage: ./img_filter <input file name> <output file name>" << std::endl;
    exit(1);
  }

  abrirImagen(input_file);

  bool cont = true;
  int opcion = -1;
  int filtro;


    std::cout << "******************************************************\n* Filtrado y modificacion de imagenes - CUDA y OpenCV *\n******************************************************\n"
              << std::endl;
    std::cout << "Que quieres hacer?\n\n\t" << std::endl;
    std::cout << "\t[" << FILTRO_BLANCO_NEGRO << "] Imagen con filtro blanco-negro" << std::endl;
    std::cout << "\t[" << FILTRO_SEPIA << "] Imagen con filtro sepia" << std::endl;
    std::cout << "\t[" << FILTRO_VOLTEADO_VERTICAL << "] Imagen sin filtro volteado vertical" << std::endl;
    std::cout << "\t[" << FILTRO_ESPECTRO_ROJO << "] Imagen con filtro espectro rojo" << std::endl;
    std::cout << "\t[" << FILTRO_ESPECTRO_VERDE << "] Imagen con filtro espectro verde" << std::endl;
    std::cout << "\t[" << FILTRO_ESPECTRO_AZUL << "] Imagen con filtro espectro azul" << std::endl;
    std::cout << "\t[" << FILTRO_SOBEL << "] Imagen con filtro de bordes" << std::endl;
    std::cout << "\t[" << CERRAR_PROGRAMA << "] Cerrar el programa" << std::endl;

    std::cin >> opcion;

    switch (opcion)
      {
        case FILTRO_BLANCO_NEGRO:
          filtro = opcion;
          break;
        case FILTRO_SEPIA:
          filtro = opcion;
          break;
        case FILTRO_VOLTEADO_VERTICAL:
          filtro = opcion;
          break;
        case FILTRO_ESPECTRO_ROJO:
          filtro = opcion;
          break;
        case FILTRO_ESPECTRO_VERDE:
          filtro = opcion;
          break;
        case FILTRO_ESPECTRO_AZUL:
          filtro = opcion;
          break;
        case FILTRO_SOBEL:
          filtro = opcion;
          break;
        default:
          std::cout << "Cerrando programa..." << std::endl;
          exit(0);
      }

  cv::Mat img_output;
  // Numero de pixeles a procesar
  int totalPixels = numRows() * numCols();

  // Usando imageReader.cpp - cargar imagen y crear los punteros de imagen origen y destino
  if(opcion == FILTRO_SOBEL){
    loadSobel(&h_in, &h_out, &d_in, &d_out);

    //Llamando al método definido en rgba_to_grey.cpp - llama al kernel de CUDA
    image_filter(d_input_image, d_output_image, numRows(), numCols(), filtro, d_in, d_out);

    // Copiar de Host a Device la imagen generada
    testCudaError(cudaMemcpy(h_out, d_out, totalPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Mover imagen de salida
    img_output = outImage;

  }else{
    loadImage(&h_input_image, &h_output_image, &d_input_image, &d_output_image);

    //Llamando al método definido en rgba_to_grey.cpp - llama al kernel de CUDA
    image_filter(d_input_image, d_output_image, numRows(), numCols(), filtro, d_in, d_out);

    // Copiar de Host a Device la imagen generada
    testCudaError(cudaMemcpy(h_output_image, d_output_image, sizeof(uchar4) * totalPixels, cudaMemcpyDeviceToHost));

    // Crear imagen de salida
    cv::cvtColor(imageOutput, img_output, COLOR_RGBA2BGR);
  }

  // Nueva ventana con titulo concreto
  cv::namedWindow(TITULO_VENTANA);
  // Mostrando nuestra imagen de salida
  cv::imshow(TITULO_VENTANA, img_output);
  std::cout << "Cierre la ventana generada para continuar\n";
  // Esperar a que el usuario cierre la ventana
  cv::waitKey(0);
  // Guardar la imagen de salida
  cv::imwrite(output_file.c_str(), img_output);

  std::cout << "\nSe ha guardado la imagen generada como " << output_file.c_str() << "\n"
            << std::endl;

  // Liberar memoria usada
  testCudaError(cudaFree(d_input_image__));
  testCudaError(cudaFree(d_output_image__));
  testCudaError(cudaFree(d_in_image__));
  testCudaError(cudaFree(d_out_image__));


  return 0;
}
