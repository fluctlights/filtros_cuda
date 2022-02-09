
# Procesado de imágenes usando CUDA y OpenCV


## 1. Introducción

El proyecto de prácticas consistirá en la implementación de un filtro de procesamiento de imágenes digitales sobre plataformas GPU 
(en concreto plataformas con soporte CUDA), siendo el objetivo principal poner en práctica los conocimientos adquiridos en teoría relativos al
diseño de programas paralelos, así como los adquiridos en prácticas relativos a la programación en CUDA.

## 2. Instalación y ejecución

CMake es un una plataforma que nos permite generar un makefile completo de forma automática a partir de una serie de parámetros introducidos en 
un archivo de texto llamado CmakeLists.txt. Al ejecutar el comando:

```
cmake.
```
Generaremos un makefile que contendrá todo lo necesario para poder compilar nuestro programa. Tras esto, al ejecutar el comando:

```
make
```
Lo que haremos será una compilación por lotes con todos los archivos necesarios para poder arrancar. Gracias a Cmake, esto es automático y
no tendremos que preocuparnos de más. Finalmente, para ejecutar el programa simplemente habrá que utilizar el siguiente comando:

```
./proyecto_cuda <nombre imagen entrada> <nombre imagen salida>
```

Usaremos las imágenes que incluiremos en el entregable: imagen.jpg e imagen_grande.jpg. El nombre de salida puede ser cualquier nombre, 
**con formato .jpg o similar.**


## 3. Implementación

Los filtros seleccionados han sido:

1. Blanco y negro
2. Sepia
3. Volteado
4. Espectro rojo
5. Espectro verde
6. Espectro azul
7. Detección de bordes

Para los filtros del 1 al 6 la imagen se almacena en formato RGBA.

Siendo TxB el máximo de hilos por bloque para nuestra gráfica, la dimensión del bloque es (TxB, 1, 1). Para la dimensión del grid se
calcula cuantos bloques de TxB hilos son necesarios para abarcar todos los pixeles utilizando:

```
blocks_x_grid = ceil(total_px / TxB)
```
Con este dato la dimensión del grid es (blocks_x_grid, 1, 1).

Para el filtro 7 la imagen se almacena en un solo canal. Para este filtro la dimensión del bloque es (32, 32, 1) y la dimensión del grid 
se calcula con la siguiente fórmula:

A -> (cols + blockSizeEspecial.x - 1) / blockSizeEspecial.x
B -> (rows + blockSizeEspecial.y - 1) / blockSizeEspecial.y

```
gridSizeEspecial(A, B)
```

## 4.Bibliografía

**Manejo de imágenes con openCV** : el modo de leer imagen está
extraído del repositorio de GitHub:

```
https://github.com/jsdario/cudacv-bw
```
**Uso de Cmake:**

```
https://cmake.org/cmake/help/latest/guide/tutorial/index.html
```
**Operador Sobel:** para esto ha sido necesario entender la explicación
teórica del funcionamiento

```
https://programmerclick.com/article/
```
