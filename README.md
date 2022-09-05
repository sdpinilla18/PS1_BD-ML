## Repositorio Taller 1, BD&ML 2022-2

**Facultad de Economía, Universidad de los Andes**

Integrantes:

David Santiago Caraballo Candela, Sergio David Pinilla Padilla & Juan Diego Valencia Romero

En este repositorio se encuentran todos los documentos, bases de datos y codigos utilizados durante el desarrollo del primer taller de la clase *Big Data & Machine Learning for Applied Economics*, del profesor [Ignacio Sarmiento](https://ignaciomsarmiento.github.io/), durante el segundo semestre de 2022.

Este trabajo tenía como objetivo el desarrollo de un modelo de predición del ingreso de los ciudadanos de Bogotá DC, Colombia, a partir del uso de una [base de datos](https://ignaciomsarmiento.github.io/GEIH2018_sample/) del 2018 de la Gran Encuesta Integrada de Hogares (GEIH) del Departamento Administrativo Nacional de Estadistica (DANE). Esto, con la intención de mejorar el proceso de identificación de fraude fiscal en personas que no reportan la totalidad de sus ingresos a las entidades gubernamentales.

Para organizar y testear la especificacion optima del modelo predictivo, se comenzaron estimando dos modelos estructurales que buscaban identificar si las variables de edad y sexo si eran influyentes a la hora de entender el comportamiento del ingreso laboral de los bogotanos. Posteriormente, a partir de estas especificaciones se fueron agregando variables que podian aumentar el poder predictivo del modelo, y la especificación final se escogió utilizando *Leave-one-out-Cross-validation*.

**1. Data-scraping**

La totalidad de la base de datos fue obtenida mediante un proceso de *data-scraping* realizado en el entorno de programación **R**. Encontramos que, para nosotros esta era la forma mas facil y eficiente de hacerlo dado que teniamos los conocimientos necesarios y en este programa el proceso es mas sencillo y directo que en otros.

Para realizar el data-scraping fue necesario tener disponibles la libreria `pacman` y los paquetes `tidyverse`, `data.table`, `plyr`, `rvest`, `XML` y `xml2`.

El codigo utilizado se encuentra en el `R script` titulado "Datascraping.R". Al utilizar este script se exporta toda la *raw-database* de la GEIH 2018 para Bogotá DC con 32177 observaciones y 178 variables al archivo "bdPS1.Rdata".

**2. Data cleaning**
