# Application of tree-based models for weather forecasting

This application enables creation of a structured dataset from a number of text files containing historical weather
variables across the Netherlands that are provided by Koninklijk Nederlands Meteorologisch Instituut (KNMI). The tree-based
models include a single decision tree model as well as two types of tree ensemble models including the Random Forest and
Gradient-boosted models. The forecast variable is a discrete variable representing the severity of precipitation.<br> 
Detailed descriptions and an example for continues time series forecasting using ensemble of models can be found at: 
https://www.sciencedirect.com/science/article/pii/S0957417410010328  


## Build and dependency management
SBT 1.1.X and Java 8+ are needed to build. For a list of dependencies and Scala version, please see [build.sbt](build.sbt).    


## Data source
The daily weather data are maintained by the Royal Netherlands Meteorological Institute and available for download at:
http://projects.knmi.nl/klimatologie/daggegevens/selectie.cgi. 


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.