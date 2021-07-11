function accuracy = binary_classifier_accuracy(theta, X, y)
  fx = sigmoid(theta' * X);
  correct = sum(y == ( fx > 0.5));
  accuracy = correct / length(y);
end
