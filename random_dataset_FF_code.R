library(dplyr)
set.seed(123)

n <- 1000
x1 <- runif(n, -1, 1)
x2 <- runif(n, -1, 1)
y <- as.factor(ifelse(x1^2 + x2^2 > 0.5, "A", "B"))
data <- data.frame(x1 = x1, x2 = x2, y = y)

train <- sample_frac(data, 0.7)
test <- data[!(rownames(data) %in% rownames(train)),]

input_size <- 2
hidden_size <- 5
output_size <- 2

W1 <- matrix(rnorm(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
W2 <- matrix(rnorm(hidden_size * output_size), nrow = hidden_size, ncol = output_size)
b1 <- matrix(rnorm(hidden_size), nrow = 1, ncol = hidden_size)
b2 <- matrix(rnorm(output_size), nrow = 1, ncol = output_size)

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}
softmax <- function(x) {
  exp(x) / rowSums(exp(x))
}

cross_entropy <- function(y_pred, y_true) {
  -sum(ifelse(y_true == "A", log(y_pred[,1]), log(y_pred[,2])))
}

forward_forward <- function(X, W1, W2, b1, b2) {
  # Positive pass
  z1 <- X %*% W1 + b1
  a1 <- sigmoid(z1)
  z2 <- a1 %*% W2 + b2
  y_pred <- softmax(z2)
  
  # Negative pass
  z1_neg <- -z1
  a1_neg <- sigmoid(z1_neg)
  z2_neg <- a1_neg %*% W2 - b2
  y_pred_neg <- softmax(z2_neg)
  
  return(list(y_pred = y_pred, y_pred_neg = y_pred_neg, a1 = a1))
}


# Train the neural network
learning_rate <- 0.05
epochs <- 120

for (i in 1:epochs) {
  for (j in 1:nrow(train)) {
    
    # Forward-forward pass
    X <- as.matrix(train[j, 1:2])
    y_true <- as.numeric(train[j, 3] == "A") + 1
    result <- forward_forward(X, W1, W2, b1, b2)
    
    # Compute gradients
    dL_dz2 <- result$y_pred
    dL_dz2[1, y_true] <- dL_dz2[1, y_true] - 1
    dL_da1 <- dL_dz2 %*% t(W2)
    dL_dz1 <- dL_da1 * result$a1 * (1 - result$a1)
    dL_dz1_neg <- dL_dz1[, 1] - dL_dz1[, 2]
    
    # Update weights and biases
    W2 <- W2 - learning_rate * t(result$a1) %*% dL_dz2  # update W2 using positive pass
    b2 <- b2 - learning_rate * dL_dz2 # update b2 using positive pass
    W1 <- W1 - learning_rate * t(as.matrix(train[j, 1:2])) %*% dL_dz1
    b1 <- b1 - learning_rate * dL_dz1_neg  # update b1 using positive pass
  }
}

# Compute and print the training loss
if (j == nrow(train)) {
  y_preds <- apply(test[, 1:2], 1, function(x) {
    forward_forward(t(as.matrix(x)), W1, W2, b1, b2)$y_pred
  })
  y_preds <- apply(y_preds, 2, function(x) {
    ifelse(x == max(x), "A", "B")
  })
  test_loss <- mean(y_preds != test$y)
  
  cat(sprintf("Epoch %d - Training Loss: %.4f, Test Loss: %.4f\n", i, cross_entropy(result$y_pred, train[j, 3]), test_loss))
}