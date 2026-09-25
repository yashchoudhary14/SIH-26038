function y = sigmoid(x)
%SIGMOID Logistic function.
y = 1 ./ (1 + exp(-x));
end
