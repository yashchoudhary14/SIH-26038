function q = calibrate(p, cal)
%CALIBRATE Isotonic recalibration of P(referable), as the bundle stores it.
%
%   The referral threshold was selected on this recalibrated scale, so the two
%   ship together in pipeline.json; applying the threshold to the raw
%   probability would put the operating point on a different number line.
%
%   Isotonic regression is a step function and creates ties -- on Messidor-2 it
%   pinned 10% of true positives to exactly 0.0, unreachable by any threshold.
%   A sliver (tie_break) of the raw score is blended back in to restore a
%   strict order. Interpolation follows numpy.interp, clamped at both ends.

if nargin < 2 || isempty(cal) || ~isfield(cal, 'kind') || ~strcmp(cal.kind, 'isotonic')
    q = p;
    return
end
x = cal.x(:);
y = cal.y(:);
tb = 1e-3;
if isfield(cal, 'tie_break'), tb = cal.tie_break; end

q = zeros(size(p));
for i = 1:numel(p)
    pi_ = p(i);
    if pi_ <= x(1)
        v = y(1);
    elseif pi_ >= x(end)
        v = y(end);
    else
        j = find(x <= pi_, 1, 'last');
        v = y(j) + (pi_ - x(j)) * (y(j + 1) - y(j)) / (x(j + 1) - x(j));
    end
    q(i) = v;
end
q = (1 - tb) * q + tb * p;
q = min(max(q, 0), 1);
end
