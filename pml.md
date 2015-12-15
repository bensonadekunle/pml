# Human Activity Recognition



##  Predicting Weight-lifting Styles Determined by Measurement of Exercise Form

### Synopsis

This report describes building a prediction model to identify how participants performed during various weight-lifting routines. Our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to make these predictions. A subset of the original data set was obtained from the Johns Hopkins University "Reproducible Research" course project. This dataset was originally obtained from the [*Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements*](http://groupware.les.inf.puc-rio.br/har#ixzz34dpS6oks).

For the original study 6 healthy participants performed 5 variations for 1 set of 10 repetitions of unilateral dumbbell bicep curls over the course of 8 hours. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Class A corresponds to the specified execution of the exercise, while Classes B through E correspond to common mistakes during weight-lifting.

1. exactly according to the specification (Class A),
2. throwing the elbows to the front (Class B),
3. lifting the dumbbell only halfway (Class C),
4. lowering the dumbbell only halfway (Class D), and
5. throwing the hips to the front (Class E). 

Our model accurately predicts the correct classification for each of the twenty test cases (pml-testing.csv) provided by the instructor. When the model was applied to the full data set (n observations) collected by the researchers an accuracy rate of 100% (95% CI: ) was achieved. The outcomes support the hypothesis that the data provided for the course project was collected under highly-controlled circumstances.

### Introduction

The necessary external R package(s) to perform the data analysis presented in this report:

- randomForest
- knitr

The randomForest package has been chosen to train the predictive model. A custom 
implementation of the confusion matrix function has been developed as a learning
aid in addition to removing any dependency on the caret package.

The R [script](https://github.com/gdhorne/pml/blob/master/pml.r) which produced the results shown in this report is available. The source code in its entirety is embedded in this document to facilitate on-demand calculation. Source code fragments presented in this report focus only on predictor selection, model building and prediction.

### Building the Prediction Model

**Selecting the Predictors**

After reading the data, not shown here, from the comma-separated-values (CSV) file, pml-training.csv, select only those features with fewer NA values than non-NA values and whose names include arm or belt or dumbbell. This strategy reduces the number of predictors and eliminates any predictor with mostly NA values.

    nonmissing <- which(sapply(training, function (x) sum(is.na(x)) < sum(!is.na(x))))

    training.predictors <- 
    names((training[ , nonmissing])[ , grep("belt|arm|dumbbell", names(training[, nonmissing]))])

    training <- training[, c('classe', training.predictors)]
    training[, 'classe'] <- as.factor(training$classe)

**Partitioning the Training Data Set to Allow Cross-Validation**

The training data is divided such that 75% reserved for training the model and 25% reserved for cross-validation of the model.

    training.indices <- sample(nrow(training), as.integer(floor(0.75 * nrow(training))))
    training.train <- training[training.indices, ]
    training.test  <- training[-training.indices, ]

**Building and Training the Model**

Building the prediction model using the Random Forest machine learning algorithm applied to the training dataset, training.train, enables efficient training and accurate predictions. Missing values in any of the predictor features are imputted by calculating the mean of that predictor.

    model <- randomForest(formula = classe ~ ., data = training.train, na.action = na.roughfix, replace = FALSE)

<u>Model</u>


```

Call:
 randomForest(formula = classe ~ ., data = training.train, replace = FALSE,      na.action = na.roughfix) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 7

        OOB estimate of  error rate: 0.42%
Confusion matrix:
     A    B    C    D    E  class.error
A 4161    2    0    0    0 0.0004804228
B   10 2900    6    0    0 0.0054869684
C    0   12 2529    2    0 0.0055053087
D    0    0   23 2393    0 0.0095198675
E    0    0    1    6 2671 0.0026138910
```

With the model the manner in which the weight-lifting exercises were performed by the participants can be predicted. As a base-line the model is applied to the training data set.

    training.train.prediction <- predict(model, training.train, type = 'response')
    confusionMatrix(training.train.prediction, training.train$classe)

<u>Confusion Matrix and Statistics</u>


```
          Reference
Prediction    A    B    C    D    E
         A 4163    0    0    0    0
         B    0 2916    0    0    0
         C    0    0 2543    0    0
         D    0    0    0 2416    0
         E    0    0    0    0 2678
```

The in-sample accuracy rate is 1.0000 with a 95% confidence interval of [ 0.9997, 1.0000].


                            Class A   Class B   Class C   Class D   Class E 
--------------------------  --------  --------  --------  --------  --------
Sensitivity                 1.0000    1.0000    1.0000    1.0000    1.0000  
Specificity                 1.0000    1.0000    1.0000    1.0000    1.0000  
Positive Predictive Value   1.0000    1.0000    1.0000    1.0000    1.0000  
Negative Predictive Value   1.0000    1.0000    1.0000    1.0000    1.0000  

These results are within expectations because the model is making predictions using the same data with which it was trained.

**Cross-Validation of the Model**

The prediction model using the Random Forest machine learning algorithm applied to the testing dataset, training.test, yields surpringly accurate classification of the manner in which the weight-lifting exercises were performed by the participants. Considering the controlled environment during the weight-lifting exercises we surmise that each of the five styles of activity performance might have been deliberately exaggerated.

    training.test.prediction <- predict(model, training.test, type = 'response')
    confusionMatrix(training.test.prediction, training.test$classe)

<u>Confusion Matrix and Statistics</u>


```
          Reference
Prediction    A    B    C    D    E
         A 1416    2    0    0    0
         B    1  879   10    0    0
         C    0    0  869    5    0
         D    0    0    0  792    3
         E    0    0    0    3  926
```

The out-of-sample accuracy rate is 0.9951 with a 95% confidence interval of [0.9926, 0.9968].


                            Class A   Class B   Class C   Class D   Class E 
--------------------------  --------  --------  --------  --------  --------
Sensitivity                 0.9986    0.9876    0.9943    0.9962    0.9968  
Specificity                 0.9997    0.9995    0.9975    0.9981    0.9992  
Positive Predictive Value   0.9993    0.9977    0.9886    0.9900    0.9968  
Negative Predictive Value   0.9994    0.9973    0.9988    0.9993    0.9992  

### Making Predictions

**Model Applied to the Held-out Data Set**

The ultimate test of our model's predictive accuracy hinges on correctly classifying the twenty observations held-back. These predictions will be submitted separately for comparison with the reference predictions so the actual predictions are withheld from this report. To view the predictions you can execute the R [script](https://github.com/gdhorne/pml/blob/master/pml.R) and examine this statement: predict <- testing.prediction.

<u>Confusion Matrix and Statistics</u>


```
          Reference
Prediction A B C D E
         A 7 0 0 0 0
         B 0 8 0 0 0
         C 0 0 1 0 0
         D 0 0 0 1 0
         E 0 0 0 0 3
```

The accuracy rate is 1.0000 with a 95% confidence interval of [0.7995, 1.0000]. The wider confidence interval can be attributed to the sample size (20 observations).


                            Class A   Class B   Class C   Class D   Class E 
--------------------------  --------  --------  --------  --------  --------
Sensitivity                 1.0000    1.0000    1.0000    1.0000    1.0000  
Specificity                 1.0000    1.0000    1.0000    1.0000    1.0000  
Positive Predictive Value   1.0000    1.0000    1.0000    1.0000    1.0000  
Negative Predictive Value   1.0000    1.0000    1.0000    1.0000    1.0000  

**Model Applied to the Master Weight-Lifting Data Set Collected by the Researchers**

To further test the predictive capability of our model it is applied to the original data set available from the *Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements* project [website](http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip).

<u>Confusion Matrix and Statistics</u>


```
          Reference
Prediction     A     B     C     D     E
         A 11175     0     0     0     0
         B     0  7609     0     0     0
         C     0     0  6854     0     0
         D     0     0     0  6390     0
         E     0     0     0     0  7213
```

The accuracy rate is 1.0000 with a 95% confidence interval of [0.9999, 1.0000].


                            Class A   Class B   Class C   Class D   Class E 
--------------------------  --------  --------  --------  --------  --------
Sensitivity                 1.0000    1.0000    1.0000    1.0000    1.0000  
Specificity                 1.0000    1.0000    1.0000    1.0000    1.0000  
Positive Predictive Value   1.0000    1.0000    1.0000    1.0000    1.0000  
Negative Predictive Value   1.0000    1.0000    1.0000    1.0000    1.0000  

### Analysis Results

The predictions achieved 100% accuracy during submission to the automatic grader. This supports the hypothesis that the data provided for the course project was collected under controlled circumstances. The perfect accuracy for the pml-testing.csv data set was expected as a requirement to pass each of the 20 performance classification test cases.



### Bibliography

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers’ Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.
