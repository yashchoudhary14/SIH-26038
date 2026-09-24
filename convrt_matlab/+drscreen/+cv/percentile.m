function q = percentile(x, p)
%PERCENTILE numpy.percentile with the default 'linear' method.
%
%   MATLAB's prctile uses a midpoint rule and returns different values, and
%   several quality and landmark thresholds are defined as percentiles.

x = sort(double(x(:)));
n = numel(x);
q = zeros(size(p));
if n == 0
    q(:) = NaN;
    return
end
for i = 1:numel(p)
    hpos = (n - 1) * p(i) / 100;
    lo = floor(hpos);
    hi = min(lo + 1, n - 1);
    q(i) = x(lo + 1) + (hpos - lo) * (x(hi + 1) - x(lo + 1));
end
end
