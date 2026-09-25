function g = gradeFromCumulative(cum, thr)
%GRADEFROMCUMULATIVE Ordinal grade, stopping at the first failed boundary.
%
%   thr is a scalar or one fitted cut-point per boundary. With per-boundary
%   cut-points, counting passes independently can yield a grade no chain of
%   conditionals supports (a low later cut-point passing after an earlier one
%   failed); stopping at the first failure keeps the prediction ordinal.
cum = cum(:);
if isscalar(thr)
    thr = repmat(thr, size(cum));
end
thr = thr(:);
if numel(thr) ~= numel(cum)
    error('drscreen:grade', 'expected %d grade thresholds, got %d', numel(cum), numel(thr));
end
g = 0;
for k = 1:numel(cum)
    if cum(k) > thr(k) && g == k - 1
        g = k;
    end
end
end
