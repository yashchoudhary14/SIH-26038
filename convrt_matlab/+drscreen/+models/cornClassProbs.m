function P = cornClassProbs(z)
%CORNCLASSPROBS CORN logits (K-1 x S) -> class probabilities (K x S).
%
%   P(y > k) = prod_{j <= k} sigmoid(z_j), monotone non-increasing by
%   construction; P(y = k) = P(y > k-1) - P(y > k).
S = size(z, 2);
cum = cumprod(drscreen.nn.sigmoid(z), 1);
upper = [ones(1, S, 'like', cum); cum];
lower = [cum; zeros(1, S, 'like', cum)];
P = max(upper - lower, 0);
end
