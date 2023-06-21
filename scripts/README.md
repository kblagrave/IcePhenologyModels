# Predicting ice phenology
### Script for predicting ice phenology using random forest model trained on lakes in the northeastern US.
Ice-on prediction relies on lake morphology (depth, area), elevation, and seasonal mean temperatures (summer, fall, and winter).
Ice-off prediction relies on lake morphology (shoreline development), elevation, and seasonal mean temperatures (summer, fall, winter, and spring).

**Requirements**
- Python 3.7+ 
- pandas 1.3.4+

From the command line, run:

`python phenology_models.py -y [YEAR] -c [LATITUDE] [LONGITUDE] -d [DEPTH] -e [ELEVATION] -s [SHORELINE_DEVELOPMENT] -a [AREA] -f [FILENAME] (optional) -t [SEASONAL_TEMPERATURES (summer fall winter spring)] (optional)`

- **YEAR**: end of winter season (i.e., 1990 for winter of 1989-1990), used to match year in `filename`
- **LATITUDE**, **LONGITUDE**: coordinates of lake, used to match lake with coordinates in `filename` (degrees)
- **DEPTH**: Average depth (m)
- **ELEVATION**: elevation (m)
- **SHORELINE_DEVELOPMENT**: A measure of the complexity of the shoreline (from HydroLAKES documentation: "Shoreline development, measured as the ratio between shoreline length and the circumference of a circle with the same area.
A lake with the shape of a perfect circle has a shoreline development of 1, while higher values indicate increasing shoreline complexity.")
- **AREA**: Lake area (sq km)

One of the following must also be included:
- **FILENAME**: file containing these columns: Latitude, Longitude, Year, Tave01, Tave02, Tave03, ... Tave12 (monthly average temperatures)
- **SEASONAL_TEMPERATURES**: mean summer, fall, winter, and spring temperatures. Enter these as four numbers after the "-t". (degrees Celsius)



