function X = groupNorm(X, G, gamma, beta, epsilon)
%GROUPNORM torch.nn.GroupNorm for a single H x W x C sample.
%
%   Channels are split into G contiguous groups; each group is normalised with
%   its own mean and biased variance over (channels-in-group, H, W), then a
%   per-channel affine is applied. Statistics are accumulated in double: at
%   1024 x 1024 a single-precision running sum over eight million values
%   drifts enough to move the result. Two passes for the statistics (mean,
%   then squared deviations), and normalisation plus affine fused into one.

if nargin < 5, epsilon = 1e-5; end
[H, W, C] = size(X);
k = C / G;
n = H * W * k;
Xc = reshape(X, H * W, C);
mu = repelem(sum(reshape(sum(Xc, 1, 'double'), k, G), 1) / n, k);
v = sum(reshape(sum((Xc - single(mu)) .^ 2, 1, 'double'), k, G), 1) / n;
a = reshape(gamma, 1, C) ./ single(repelem(sqrt(v + epsilon), k));
X = reshape(Xc .* a + (reshape(beta, 1, C) - single(mu) .* a), H, W, C);
end
