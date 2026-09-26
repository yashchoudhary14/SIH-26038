function X = squeezeExcite(X, rw, rb, ew, eb)
%SQUEEZEEXCITE Channel attention: X .* sigmoid(expand(silu(reduce(mean(X))))).
%
%   rw : [1 1 C r] reduce weights, rb : r biases
%   ew : [1 1 r C] expand weights, eb : C biases

C = size(X, 3);
r = size(rw, 4);
z = reshape(mean(X, [1 2]), 1, C);
z = drscreen.nn.silu(z * reshape(rw, C, r) + reshape(rb, 1, r));
z = drscreen.nn.sigmoid(z * reshape(ew, r, C) + reshape(eb, 1, C));
X = X .* reshape(z, 1, 1, C);
end
