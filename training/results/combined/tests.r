library(twowaytests)
library(phia)
data = read.csv("accuracies_all.csv", header = TRUE, sep = ",",  stringsAsFactors = TRUE) 
accuracy = data$accuracy
classifier = data$classifier
out <- tmeanTwoWay(accuracy ~ dataset*classifier, data = data)
paircompTwoWay(out)

model = lm(accuracy ~ dataset*classifier, data=data)

IM = interactionMeans(model)
plot(IM, atx = "dataset", traces = "classifier")