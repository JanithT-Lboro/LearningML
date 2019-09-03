# Problem  
Problem is to try to predict the burned area of forest firest in the northeast region of Portugal. 
Data was found [here](https://archive.ics.uci.edu/ml/datasets/forest+fires)  

## Citation Request: 
This dataset is public available for research.  
The details are described in [Cortez and Morais, 2007].  
Please include this citation if you plan to use this database:  
P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data.  
In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. 
Available at: http://www.dsi.uminho.pt/~pcortez/fires.pdf

'Compressive strength or compression strength is the capacity of a material or structure to withstand loads tending to reduce size, as opposed to tensile strength, which withstands loads tending to elongate.  

compressive strength is one of the most important engineering properties of concrete. It is a standard industrial practice that the concrete is classified based on grades. This grade is nothing but the Compressive Strength of the concrete cube or cylinder. Cube or Cylinder samples are usually tested under a compression testing machine to obtain the compressive strength of concrete. The test requisites differ country to country based on the design code.  

The concrete compressive strength is a highly nonlinear function of age and ingredients .These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.'    

## Attribute Information
For more information, read [Cortez and Morais, 2007]. 
1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9 
2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9 
3. month - month of the year: 'jan' to 'dec' 
4. day - day of the week: 'mon' to 'sun' 
5. FFMC - FFMC index from the FWI system: 18.7 to 96.20 
6. DMC - DMC index from the FWI system: 1.1 to 291.3 
7. DC - DC index from the FWI system: 7.9 to 860.6 
8. ISI - ISI index from the FWI system: 0.0 to 56.10 
9. temp - temperature in Celsius degrees: 2.2 to 33.30 
10. RH - relative humidity in %: 15.0 to 100 
11. wind - wind speed in km/h: 0.40 to 9.40 
12. rain - outside rain in mm/m2 : 0.0 to 6.4 
13. area - the burned area of the forest (in ha): 0.00 to 1090.84 
(this output variable is very skewed towards 0.0, thus it may make 
sense to model with the logarithm transform).

## Approach 
The model created attempts to predict the compressive strength of concrete using the listed features. I made use of sklearn library and imported multiple algorithms to fit the [data](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/Concrete_Data.xls) to, in an attempt to find the one which performs best.    
The workflow for this problem was as follows:
- load dataset
- visualise dataset
- split dataset in training and test
- fit multiple models to training data
- predict for test inputs and compare results against test output    

This is easier recognised within the [source code.](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/concreteCompression2.py)  
The various outputs from this workflow can be found below... 

## Variables Summary
### Distribution of inputs
![cement](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/cement_hist.png)
![furnaceSlag](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/furnaceSlag_hist.png)
![flyAsh](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/flyAsh_hist.png)
![water](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/water_hist.png)
![superPlasticizer](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/superPlasticizer_hist.png)
![coarseAgg](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/coarseAgg_hist.png)
![fineAgg](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/fineAgg_hist.png)
![age](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/age_hist.png)
### Distribution of Compressive Strength of Concrete
![compressiveStr](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/compressiveStr_hist.png)

### Pair plot of all variables
![pairplot](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/pairplot.png)
### Corellation of data heatmap
![heatmap](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/heatmap.png)

## Regressor Model Summaries
### Linear 
RMSE: 9.795398721673989  
cross validation scores:  
cv_mean: -114.2803289093508  
cv_std : 30.97472155939001  
cv_max : -69.11350348829288  
cv_min : -182.32786747085345  
### Ridge
RMSE: 9.795403258998174  
cross validation scores:  
cv_mean: -114.28027376941168  
cv_std : 30.974720390144906  
cv_max : -69.11358504057787  
cv_min : -182.32776523171808  
### Lasso 
RMSE: 9.814158430974311  
cross validation scores:  
cv_mean: -114.33989273214044  
cv_std : 30.988222580936736  
cv_max : -69.62466384275456  
cv_min : -181.5355275143075  
### AdaBoost 
RMSE: 7.463366760984524  
cross validation scores:  
cv_mean: -62.034275033019696  
cv_std : 11.502743280075848  
cv_max : -35.2872339112232  
cv_min : -84.45680821179937  
### Extra Trees
RMSE: 4.393527569553418  
cross validation scores:  
cv_mean: -27.783247529989797  
cv_std : 10.976242557468428  
cv_max : -15.157219985958205  
cv_min : -54.73096984426451  
### Random Forest 
RMSE: 4.85921342620648  
cross validation scores:  
cv_mean: -28.92654077558577  
cv_std : 11.463643581743373  
cv_max : -10.814747898017886  
cv_min : -62.783547469457  
### Gradient Boosting  
RMSE: 4.516740235579305  
cross validation scores:  
cv_mean: -28.796978089462964  
cv_std : 10.105333195100439  
cv_max : -17.25217971564181  
cv_min : -59.305702049837336  
### KNN  
RMSE: 8.889407344918  
cross validation scores:  
cv_mean: -88.80697042800375  
cv_std : 22.50831845644143  
cv_max : -42.70991345411688  
cv_min : -131.89289369501964  
![]() 
![]()
![]()
![]()


