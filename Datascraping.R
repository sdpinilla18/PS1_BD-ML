#=============================================================================#
#      Problem Set 1 Big Data and Machine Learning for Applied Economics      #
#           David Santiago Caraballo Candela             201813007            #
#           Sergio David Pinilla Padilla                 201814755            #
#           Juan Diego Valencia Romero                   201815561            #
#=============================================================================#

# 0. DATA-SCRAPING

#Initial workspace configuration
rm(list=ls()) 
Sys.setlocale("LC_CTYPE","en_US.UTF-8")
require(pacman)
p_load(tidyverse,data.table,plyr,rvest,XML,xml2)

#Read pages
dc_url<-paste0("https://ignaciomsarmiento.github.io/GEIH2018_sample/pages/geih_page_", 1:10, ".html")
df<-data.frame()
for (url in dc_url){
  print(url)
  temp<-read_html(url) %>% html_table() # Extrae la tabla de la URL
  temp<-as.data.frame(temp[[1]]) # Selecciona la tabla de interés 
  df<-rbind(df, temp) # Une (append) la tabla con las previamente seleccionadas
}

#Save database
save(df, file="bdPS1.Rdata")

#Load data
db <- load("bdPS1.Rdata")

