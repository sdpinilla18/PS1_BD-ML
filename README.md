## Repositorio Taller 1

**Facultad de Economía**

**Universidad de los Andes**

Integrantes: [David Santiago Caraballo Candela](https://github.com/scaraballoc), [Sergio David Pinilla Padilla](https://github.com/sdpinilla18) y [Juan Diego Valencia Romero](https://github.com/judval).

En el presente repositorio se encuentran todos los documentos, bases de datos y códigos utilizados durante el desarrollo del Taller 1 de la clase *Big Data & Machine Learning for Applied Economics*, del profesor [Ignacio Sarmiento-Barbieri](https://ignaciomsarmiento.github.io/igaciomsarmiento) durante el segundo semestre del año 2022.

Este trabajo tenía como objetivo el desarrollo de un modelo de predicción del ingreso de los ciudadanos de Bogotá D.C., Colombia, a partir del uso de una [base de datos](https://ignaciomsarmiento.github.io/GEIH2018_sample/) del 2018 de la Gran Encuesta Integrada de Hogares (GEIH) recolectada por el Departamento Administrativo Nacional de Estadistica (DANE). Tal insumo, con la intención de mejorar el proceso de identificación de fraude fiscal en personas que no reportan la totalidad de sus ingresos a las entidades gubernamentales.

Para organizar y *testear* la especificacion optima del modelo predictivo, se comenzó estimando dos (2) modelos estructurales que buscaban identificar si las variables de edad y género eran determinantes a la hora de entender el comportamiento del ingreso laboral de los ciudadanos. Posteriormente, a partir de estas especificaciones se fueron agregando regresores y controles que pretendían aumentar el poder predictivo del modelo, y la especificación final que se escogió utilizando el proceso de *Leave-One-Out-Cross-Validation (LOOCV)*.

**1. *Data scraping***

La totalidad de la base de datos fue obtenida mediante un proceso de *data-scraping* realizado en un entorno de desarrollo integrado para el lenguaje de programación **R**. Encontramos que, para nosotros esta era la forma más fácil y eficiente de hacerlo dado que se proveyó una explicación específica en la clase complementeria y en este programa el proceso es más sencillo y directo. Para realizar el *data-scraping* fue necesario tener disponibles la libreria `pacman` y los paquetes comop `tidyverse`, `data.table`, `plyr`, `rvest`, `XML` y `xml2`. El código utilizado se encuentra en el *R script* titulado "Datascraping.R". Al utilizar este script se exporta toda la *raw-database* de la GEIH 2018 para Bogotá D.C. con 32.177 observaciones y 178 variables al archivo "bdPS1.Rdata".

**2. *Data cleaning & Modelling***

Luego de realizar el *data-scraping* en **R**, migramos al lenguaje de programación **Python** para realizar la limpieza y organización de la base de datos, y la modelación y estimación de todas las especificaciones propuestas. Se tomó esta decisión dado que, por un lado, poseemos mayores conocimientos técnicos en **Python**, y, por otro lado, consideramos que es un programa con importantes ventajas absolutas en temas de versatilidad y eficiencia al momento de manejar grandes volúmenes de datos (en especial teniendo en cuenta que para este trabajo se utilizaron estimaciones de errores estándar con *bootstrap* y errores de predicción con validación cruzada). Para poder utilizar nuestro código de **Python**, es necesario tener instalados los paquetes de `pandas`, `numpy`, `pyreadr`, `sklearn`, `scipy`, `statsmodels`, `matplotlib`, `seaborn` y `bootstrap_stat`. El código completo, que incluye todo el proceso de limpieza de datos, extracción de estadísticas descriptivas y estimación de los diez (10) modelos utilizados para responder a las preguntas del *problem set* se encuentran en orden dentro del notebook de Jupyter titulado "PS1_BD.ipynb". El *Python script* asociado al notebook esta titulado como "T1Script.py".

***Nota:*** *Este archivo debería correr a la perfección siempre y cuando se sigan las instrucciones y comentarios del código (en orden y forma). Es altamente recomendable que antes de verificar la replicabilidad del código, se asegure de tener **todos** los requerimientos informáticos previamente mencionados. Además, la velocidad de ejecución dependerá de las características propias de su máquina, por lo que deberá (o no) tener paciencia mientras se procesa.*
