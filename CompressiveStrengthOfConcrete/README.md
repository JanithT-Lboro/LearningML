# Problem  
Problem defined on [Kaggle website:](https://www.kaggle.com/pavanraj159/concrete-compressive-strength-data-set)  

'Compressive strength or compression strength is the capacity of a material or structure to withstand loads tending to reduce size, as opposed to tensile strength, which withstands loads tending to elongate.  

compressive strength is one of the most important engineering properties of concrete. It is a standard industrial practice that the concrete is classified based on grades. This grade is nothing but the Compressive Strength of the concrete cube or cylinder. Cube or Cylinder samples are usually tested under a compression testing machine to obtain the compressive strength of concrete. The test requisites differ country to country based on the design code.  

The concrete compressive strength is a highly nonlinear function of age and ingredients .These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.'    

The model created attempts to predict the compressive strength of concrete using the listed features. I made use of sklearn library and imported multiple algorithms to fit the data to, in an attempt to find the one which performs best.    
The workflow for this problem was as follows:
- load dataset
- visualise dataset
- split dataset in training and test
- fit multiple models to training data
- predict for test inputs and compare results against test output

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
### Heatmap of all variables
![heatmap](https://github.com/JanThan/LearningML/blob/master/CompressiveStrengthOfConcrete/figures/heatmap.png)

![]()
![]()
![]()
![]()


