# Computación Parlela - Proyecto 3 Cuda

Universidad del Valle de Guatemala

Facultad de Ingeniería

Departamento de Ciencias de la Computación

- Sara Maria Paguaga Gonzales - 20634
- Guillermo Santos Barrios - 191517
- Cristian Fernando Laynez Bachez - 201281

## Avances del proyecto

Para llevar a cabo la ejecución correcta del programa se tubo que implementar un Makefile para la correcta ejecución del programa junto con las librerías necesarias.

Se llevo a cabo varios intentos en diferentes ambientes.

==> Se intento ejecutar en windows 10 con el siguiente comando:

```
nmake -f Makefile
```

Pero las maquinas del lab 4 no tienen g++, así que se probo por utilizar cl que es un compilador de c++ de visual studio pero tampoco hubo exito.

![Alt text](./img_try/Make03Windows.jpeg 'Make03Windows')

==> Se intento en MSYS2

![Alt text](./img_try/Make00.jpeg 'Make00')

Pero a la hora de ejecutar el makefile este daba los siguientes problemas

![Alt text](./img_try/Make01.jpeg 'Make01')

==> Y por ultimo en WSL

Pero en este no se pudo ya que se necesitaba permisos de administrador para hacer funcionar la consola, y como las computadoras del lab 4 no tiene dichos permisos entonces no se pudo ejecutar el makefile desde ahí.

==> ¿Entonces que hicimos para lograrlo?

Por el momento logramos ejecutar el programa por medio de Cuda:

![Alt text](./img_try/Make04Final.jpeg 'Make04Final')