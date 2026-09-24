function y = silu(x)
%SILU x * sigmoid(x) (a.k.a. swish), the activation in both networks.
y = x ./ (1 + exp(-x));
end
