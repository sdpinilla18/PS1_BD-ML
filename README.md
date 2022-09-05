## Repositorio Taller 1, 2022-2


**Big Data and Machine Learing for Applied Economics**
**Facultad de Economía**
**Universidad de los Andes**

Integrantes:

[David Santiago Caraballo Candela](https://github.com/scaraballoc), [Sergio David Pinilla Padilla](https://github.com/sdpinilla18) & [Juan Diego Valencia Romero](https://github.com/judval)

En este repositorio se encuentran todos los documentos, bases de datos y códigos utilizados durante el desarrollo del primer taller de la clase *Big Data & Machine Learning for Applied Economics*, del profesor [Ignacio Sarmiento](https://ignaciomsarmiento.github.io/), durante el segundo semestre de 2022.

Este trabajo tenía como objetivo el desarrollo de un modelo de predicción del ingreso de los ciudadanos de Bogotá DC, Colombia, a partir del uso de una [base de datos](https://ignaciomsarmiento.github.io/GEIH2018_sample/) del 2018 de la Gran Encuesta Integrada de Hogares (GEIH) del Departamento Administrativo Nacional de Estadistica (DANE). Esto, con la intención de mejorar el proceso de identificación de fraude fiscal en personas que no reportan la totalidad de sus ingresos a las entidades gubernamentales.

Para organizar y testear la especificacion optima del modelo predictivo, se comenzaron estimando dos modelos estructurales que buscaban identificar si las variables de edad y sexo si eran influyentes a la hora de entender el comportamiento del ingreso laboral de los bogotanos. Posteriormente, a partir de estas especificaciones se fueron agregando variables que podian aumentar el poder predictivo del modelo, y la especificación final se escogió utilizando *Leave-one-out-Cross-validation*.

**1. Data-scraping**

La totalidad de la base de datos fue obtenida mediante un proceso de *data-scraping* realizado en el entorno de programación **R**. Encontramos que, para nosotros esta era la forma más fácil y eficiente de hacerlo dado que teniamos los conocimientos necesarios y en este programa el proceso es más sencillo y directo que en otros.

Para realizar el data-scraping fue necesario tener disponibles la libreria `pacman` y los paquetes `tidyverse`, `data.table`, `plyr`, `rvest`, `XML` y `xml2`.

El código utilizado se encuentra en el *R script* titulado "Datascraping.R". Al utilizar este script se exporta toda la *raw-database* de la GEIH 2018 para Bogotá DC con 32177 observaciones y 178 variables al archivo "bdPS1.Rdata".

**2. Data cleaning & Modeling**

Luego de realizar el *data-scraping* en **R**, migramos a **Python** para realizar la limpieza y organización de la base de datos, y la modelación y estimación de todas las especificaciones propuestas. Se tomó esta decisión dado que, por un lado, poseemos mayores conocimientos técnicos en **Python**, y, por otro lado, consideramos que es un programa mucho más versátil y eficiente a la hora de procesar grandes cantidades de datos (en especial teniendo en cuenta que para este taller se utilizaron estimaciones de errores estándar con *bootstrap* y errores de predicción con LOOCV).

Para poder utilizar nuestro código de **Python**, es necesario tener instalados los paquetes de `pandas`, `numpy`, `pyreadr`, `sklearn`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `bootstrap_stat`. El codigo completo, que incluye todo el proceso de limpieza de datos, extracción de estadisticas descriptivas y estimación de los 10 modelos utilizados se encuentran en orden dentro del notebook titulado "Definitive_code.ipynb".
