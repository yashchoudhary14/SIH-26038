function cum = cumulativeFromClassProbs(p)
%CUMULATIVEFROMCLASSPROBS P(y > k), k = 0..K-2, from a class-probability vector.
%   Lets the ordinal grade rule run on an MC-averaged posterior, which no
%   single logit vector corresponds to.
p = p(:);
K = numel(p);
cum = zeros(K - 1, 1);
for k = 1:K-1
    cum(k) = sum(p(k + 1:K));
end
end
