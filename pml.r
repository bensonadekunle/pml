require(randomForest)

confusionMatrix <- function(p, c) {
    
    cm <- table(p, c, dnn = c("Prediction", "Reference"))
    
    sum.diagonal <- sum(diag(cm))
    sum.off.diagonal <- sum(cm[which(row(cm) != col(cm))])
    total <- sum.diagonal + sum.off.diagonal
    
    accuracy <- sprintf("%.4f", sum.diagonal / total)
    
    model.stats <- prop.test(sum.diagonal, total, 0.95)
    ci <- format(prop.test(sum.diagonal, total, 0.95)[["conf.int"]][1:2], digits = 4, nsmall = 4)
    
    cm.statistics <- matrix(rep(0.0), 4, 5)
    
    rownames(cm.statistics) <- c('Sensitivity', 'Specificity', 
                                 'Positive Predictive Value', 
                                 'Negative Predictive Value')
    
    colnames(cm.statistics) <- colnames(cm)
    
    sapply(colnames(cm), 
           function(c) {
               sensitivity <- cm[c, c] / (cm[c, c] + sum(cm[c, ]) - cm[c, c])
               specificity <- (total - cm[c, c]) / 
                   (sum(cm[ , c]) - 2 * cm[c, c] + total)
               positive.predictive.value <- cm[c, c] / sum(cm[ , c])
               negative.predictive.value <- (total - cm[c, c]) / 
                   (sum(cm[c, ]) - 2 * cm[c, c] + total)
               
               cm.statistics['Sensitivity', c] <<- 
                   format(sensitivity, digits = 4, nsmall = 4)
               cm.statistics['Specificity', c] <<- 
                   format(specificity, digits = 4, nsmall = 4)
               cm.statistics['Positive Predictive Value', c] <<- 
                   format(positive.predictive.value, digits = 4, nsmall = 4)
               cm.statistics['Negative Predictive Value', c] <<- 
                   format(negative.predictive.value, digits = 4, nsmall = 4)
           })
    
    colnames(cm.statistics) <- sapply(colnames(cm), 
                                      function(c) paste("Class", c))
    
    return(list(cm = cm, accuracy = accuracy, ci = ci, statistics = cm.statistics))
}

# Obtain the source data for subsequent analysis
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
filename <- "pml-training.csv"

if (!file.exists(filename)) {
    download.file(url = url, destfile = filename, method = "curl", quiet = TRUE)
}

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
filename <- "pml-testing.csv"

if (!file.exists(filename)) {
    download.file(url = url, destfile = filename, method = "curl", quiet = TRUE)
}

training <- read.csv(file = "pml-training.csv", na.strings = c('NA',''), 
                     stringsAsFactors = FALSE)

nonmissing <- which(!sapply(training, function (x) any(is.na(x))))
nonmissing <- which(sapply(training, function (x) sum(is.na(x)) < sum(!is.na(x))))

training.predictors <- 
    names((training[ , nonmissing])[ , grep("belt|arm|dumbbell", 
                                            names(training[, nonmissing]))])

training <- training[, c('classe', training.predictors)]
training[, 'classe'] <- as.factor(training$classe)

set.seed(floor(2.718281828))

training.partition <- 0.75
training.indices <- sample(nrow(training), 
                           as.integer(0.75 * nrow(training)))
training.train <- training[training.indices, ]
training.test  <- training[-training.indices, ]

model <- randomForest(formula = classe~., data = training.train, 
                      na.action = na.roughfix, replace = FALSE)

training.train.prediction <- predict(model, training.train, type = 'response')

cm0 <- confusionMatrix(training.train.prediction, training.train$classe)

training.test.prediction <- predict(model, training.test, type = 'response')

cm1 <- confusionMatrix(training.test.prediction, training.test$classe)

testing <- read.csv("pml-testing.csv", na.strings=c('NA',''), 
                    stringsAsFactors = FALSE)

testing <- testing[ , training.predictors]
testing.prediction <- predict(model, testing, type = 'response')
testing$classe <- as.character(testing.prediction)

cm2 <- confusionMatrix(testing.prediction, testing$classe)
predict <- testing.prediction

url <- "http://groupware.les.inf.puc-rio.br/static/WLE/WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv"
filename <- "pml-master.csv"

if (!file.exists(filename)) {
    download.file(url = url, destfile = filename, method = "curl", TRUE)
}

master.testing <- read.csv("pml-master.csv", na.strings=c('NA',''), 
                           stringsAsFactors = FALSE)

master.testing <- master.testing[ , c('classe', training.predictors)]
#testing[, 'classe'] <- as.factor(testing$classe)
master.testing.prediction <- predict(model, master.testing[, -which(colnames(master.testing) == 'classe')], type = 'response')
master.testing$classe <- as.character(master.testing.prediction)

cm3 <- confusionMatrix(master.testing.prediction, master.testing$classe)

