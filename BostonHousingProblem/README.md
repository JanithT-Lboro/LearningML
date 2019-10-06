See [BostonHousing.py](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/BostonHousing.py) to see my approach to this common problem.  


![MEDV_LSTAT](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/MEDV_LSTAT.png)
# Visualisation of dataset
A general description of the dataset:  
![description](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/descr.png)
What the first 5 data points looks like:  
![head](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/head.PNG)  

## Heatmap plot
I used a heatmap of the dataset to quickly identify strong correlations between features:
![heat_map](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/heatmap.png)
Strong correlations can be found between the output MEDV (Median value of owner-occupied homes in $1000's) and the two features:   LSTAT (% lower status of the population) and RM (average number of rooms per dwelling)
## MEDV and LSTAT correlation 
![MEDV_LSTAT](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/MEDV_LSTAT.png)
## MEDV and RM correlation 
![MEDV_RM](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/MEDV_RM.png)


# Comparison of different methods
A comparison of the different accuracies achieved via the different methods of classification attempted:  
![comparison of methods](https://github.com/JanThan/LearningML/blob/master/IRIS/images/comparison_methods2.png)

# Evaluation
From the above results, it can be seen that any of the listed methods would perform well.  
I chose to use the SVM method going forward.  
I was able to achieve an accuracy of 93% with very good f1, prescision and recall results...  
![results](https://github.com/JanThan/LearningML/blob/master/IRIS/images/results.PNG)

