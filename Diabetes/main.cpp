#include "ClassExtraction/extractiondata.h"
#include "Regression/linearregression.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <list>
#include <iostream>
#include <vector>
#include <fstream>


int main(int argc, char* argv[])
{
    std::cout << "__________________________________________________" << std::endl;
    /*DATOS DE ENTRADA*/
    /*Se necesitan 3 argumentos de entrada
    *   - ruta del dataset (csv)
    *   - separador entre datos (",")
    *   - si tiene cabecera o no (true)
    */
    ExtractionData Obj_extraccion(argv[1],argv[2],argv[3]);
    std::cout << "Archivo csv de entrada: " << argv[1] << std::endl;
    std::cout << "Separador de datos: " << argv[2] << std::endl;
    std::cout << "Cabecera: " << argv[3] << std::endl;

    /*LECTURA*/
    //Se crea un vector de vectores del tipo string para cargar objeto Extraction Data lectura
    std::vector<std::vector<std::string>> lec_datos = Obj_extraccion.LeerCSV();
    //Se establece la cantidad de filas y columnas
    int filas = lec_datos.size();
    int columnas = lec_datos[0].size();
    std::cout << "Numero de filas: " << filas << std::endl;
    std::cout << "Numero de columnas: " << columnas << std::endl;
    std::cout << "__________________________________________________" << std::endl;

    /*ASIGNACION A MATRIZ EIGEN*/
    //Se crea una matriz Eigen, para ingresar los valores a esa matriz: vector de vectores, numero de filas y de columnas
    Eigen::MatrixXd matData = Obj_extraccion.CSVtoEigen(lec_datos, filas, columnas);

    /*CALCULO DEL PROMEDIO Y DESVIACION ESTANDAR*/
    std::cout << "Promedios por columnas: " << std::endl;
    std::cout << "  " << Obj_extraccion.Promedio(matData) << std::endl;
    std::cout << "Desviaciones estandar por columnas: " << std::endl;
    std::cout << "  " << Obj_extraccion.DevStand(matData) << std::endl;

    /*NORMALIZACION DE LOS DATOS*/
    //Se normaliza la matriz de datos y se crea una nueva matriz normData con los datos normalizados
    Eigen::MatrixXd normData = Obj_extraccion.Norm(matData);

    /*DIVISION DE DATOS EN TRAIN Y TEST*/
    //Se divide en datos de entrenamiento y datos de prueba, dados en grupos de entrenamiento de X,y y grupos de test de X,y
    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> tupla_datos = Obj_extraccion.TrainTestSplit(normData, 0.8);
    //Se descomprime la tupla en los cuatro conjuntos requeridos
    std::tie(X_train, y_train, X_test, y_test) = tupla_datos;
    //Inspeccion visual de division conjunto de datos train y test
    std::cout << "__________________________________________________" << std::endl;
    std::cout << "Conjunto de entrenamiento: " << std::endl;
    std::cout << "  X_train columnas: " << X_train.cols() << ", filas: " << X_train.rows() << std::endl;
    std::cout << "  Y_train columnas: " << y_train.cols() << ", filas: " << y_train.rows() << std::endl;
    std::cout << "Conjunto de prueba: " << std::endl;
    std::cout << "  X_test columnas: " << X_test.cols() << ", filas: " << X_test.rows() << std::endl;
    std::cout << "  Y_test columnas: " << y_test.cols() << ", filas: " << y_test.rows() << std::endl;
    std::cout << "__________________________________________________" << std::endl;

    /*CLASE DE REGRESION LINEAL*/
    //Se instancia la clase de regresion lineal en un objeto
    linearregression modeloLR;

    /*PROCESO DE ENTRENAMIENTO Y PRUEBA*/
    //Se crea vectores auxiliares para prueba y entrenamiento, inicializamos en 1
    Eigen::VectorXd vector_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vector_test = Eigen::VectorXd::Ones(X_test.rows());
    //Se redimensiona la matriz de entrenamiento y de prueba para ser ajustada a los vectores auxiliares anteriores
    //Train: Se redimensiona a una columna adicional
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    //Train: Se añade a la nueva columna el vector de unos
    X_train.col(X_train.cols()-1) = vector_train;
    //Test: Se redimendiona a una columna adicional
    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    //Test: Se añade a la nueva columna el vector de unos
    X_test.col(X_test.cols()-1) = vector_test;

    /*PARAMETROS*/
    //Se crea el vector de coeficientes theta
    Eigen::VectorXd thetas = Eigen::VectorXd::Zero(X_train.cols());
    //Se establece el alpha como ratio de aprendizaje de tipo flotante
    float learning_rate = 0.01; //alpha
    int num_iter = 1000; //numero de iteraciones para el gradiente descendiente
    //Se crea un vector para almacenar las thetas de salida (parametros m y b)
    Eigen::VectorXd thetas_salida;
    //Se crea un vector sencillo (std) de flotantes para almacenar los valores del costo
    std::vector<float> costo;

    /*OPTIMIZACION DE PARAMETROS*/
    //Se calcula el gradiente descendiente
    std::tuple<Eigen::VectorXd, std::vector<float>> gradiente = modeloLR.GradientDescent(X_train,y_train,
                                                                                      thetas,
                                                                                      learning_rate,
                                                                                      num_iter);
    //Se desempaqueta el gradiente
    std::tie(thetas_salida,costo) = gradiente;

    /*PROMEDIO Y DESVIACION ESTANDAR PARA Y_HAT, DESNORMALIZACION DE DATOS*/
    //Calculo de Promedio / Desviacion para y_hat, adicional se desnormalizan los datos para calcular la metrica R2_score
    //Se almacenan los valores de thetas y costos en un fichero para posteriormente ser visualizados
    //Se ejecutan una sola vez para generar los archivos
    Obj_extraccion.VectortoFile(costo, "costosModeloC++.txt");
    Obj_extraccion.EigentoFile(thetas_salida, "thetasModeloC++.txt");
    //A continuacion se extrae el promedio de la matriz entrada
    auto prom_data = Obj_extraccion.Promedio(matData);
    //Se extraen los valores de la variable independiente
    auto prom_independientes = prom_data(0,8);
    //Se escalan los datos
    auto escalado = matData.rowwise()-matData.colwise().mean();
    //Se extrae la desviacion estandar de los datos escalados
    auto desv_stand = Obj_extraccion.DevStand(escalado);
    //Se extraen los valores de la variable independiente de la devstand
    auto desv_independientes = desv_stand(0,8);

    /*CALCULO DE PREDICCIONES*/
    //Calculo de valores estimados (predicciones) y_hat, se desnormaliza y "y = mX+b"
    //Se crea una matriz para almacenar los valores estimados de entrenamiento
    Eigen::MatrixXd y_train_hat = (X_train* thetas_salida * desv_independientes).array() + prom_independientes;
    //Matriz para los valores reales de y
    //Valores reales train no normalizados
    //Del total de filas del dataset (168) se saca el 80% para determinar el grupo de entrenamiento de y, es decir, 614 filas
    Eigen::MatrixXd y = matData.col(8).topRows(614);

    /*METRICA DE RENDIMIENTO*/
    //Se revisa que tan bueno fue el modelo a traves de la metrica de rendimiento R² score o coeficiente de determinacion
    float metrica_R2 = modeloLR.R2_Score(y, y_train_hat);
    std::cout << "Metrica R2 conjunto entrenamiento: " << metrica_R2 << std::endl;

    return EXIT_SUCCESS;
}
