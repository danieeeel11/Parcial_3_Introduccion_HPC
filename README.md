# Parcial_3_Introduccion_HPC
Universidad Sergio Arboleda
* Daniel Felipe Velasquez Rincon 

Parcial 3

Link del cuaderno de colab del parcial 3:
https://colab.research.google.com/drive/1fCI6eDlBq23fiif56fmxnecBumSKeil2?usp=sharing

Se puede ingresar al cuaderno de colab con el link mencionado anteriormente.

Si se desean ejecutar el cuaderno de colab con el archivo disponible en el repositorio se debe:
  Para despegar y ejecutar los archivos contenidos en este repositorio, es necesario lo siguiente:
  * Abrir Colaboratory de Google en el navegador

  En caso de desear ejecutar los puntos del parcial: 
  * En el repositorio, ubicarse en la pestaña "Code"
  * Click en "Code" (boton ubicado en color verde)
  * Se despliegan unas opciones, click en "Download ZIP"
  * Se va a descargar un zip con los archivos del repositorio
  * En el explorador de archivos, unzip el arhivo zip recien descargado
  * Abrir Colaboratory
  * Click en 'File'
  * Click en 'Open notebook'
  * Click en 'Upload'
  * Click en 'Choose File'
  * Se abrira el explorador de archivos
  * Seleccionar el archivo 'Parcial 3 - Daniel Velasquez.ipynb'

  Recordar que una vez abierto el notebook:
  * Click en 'Files' en el notebok en la sección lateral izquierda de la pantalla
  * Ubicar en el explorador de archivos el archivo 'diabetes.csv'
  * Arrastrar este archivo a 'Files'
  * Esto es para poder manejar el archivo csv

A final de cuentas, se incluyen los siguientes archivos:
* Parcial 3 - Daniel Velasquez.ipynb (El cual es archivo que se ejecuta en colab)
* Parcial 3 - Daniel Velasquez.py
* diabetes.csv (El dataset seleccionado)
* Carpeta Diabetes
  * Carpeta ClassExtraction (Contiene clase para la extraccion de data y funciones relacionadas)
    * extractiondata.h
    * extractiondata.cpp
  * Carpeta Regression (Contiene clase para la regresion lineal y funciones relacionadas)
    * linearregression.h
    * linearregression.cpp
  * Carpeta DataSets
    * diabetes.csv
  * Carpeta Debug
    * Diabetes (ejecutable)
  * CMakeLists.txt (archivo make del proyecto)
  * CMakeLists.txt.user
  * main.cpp (Clase principal)
  
Nota: Si se desea ejecutar las clases del modelo C++, puede acceder al cuaderno de python, acceder a la tabla de contenidos y acceder a la seccion 6.1.15, donde se especifica como realizar tal proceso. Para esto, se requiere tambien descargar los archivos del repositorio (.zip) y realizar el paso a paso detallado en la seccion mencionada anteriormente (claro esta que la direccion del archivo csv ira ajustado con respecto a donde fue descargado el archivo).
