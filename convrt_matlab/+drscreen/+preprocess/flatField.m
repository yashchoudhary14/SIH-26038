function out = flatField(gray, mask, order, iterations)
%FLATFIELD Divide out the illumination field, modelled as a low-order polynomial.
%
%   A wide Gaussian would be the usual choice, but its kernel is of the same
%   spatial scale as the macula, so it absorbs the macular depression into the
%   "illumination" and the fovea detector is left nothing to find. A degree-2
%   surface has six free parameters: enough for vignetting and a tilted flash,
%   structurally incapable of a localised dip. The fit is re-weighted with the
%   brightest and darkest residuals trimmed so the disc and large haemorrhages
%   do not drag it.
%
%   Points are gathered in row-major order and subsampled with a fixed stride,
%   exactly as numpy's boolean indexing and [::step] do in the Python pipeline,
%   so both fit the same points.

if nargin < 3, order = 2; end
if nargin < 4, iterations = 3; end
g = single(gray);
[h, w] = size(g);
m = mask > 0;
if nnz(m) < 50
    out = g / max(mean(g(:)), 1);
    return
end

[Xn, Yn] = meshgrid(single((0:w-1) / w - 0.5), single((0:h-1) / h - 0.5));

% row-major gather: index the transposes
Xt = Xn.'; Yt = Yn.'; Gt = g.'; Mt = m.';
xs = double(Xt(Mt)); ys = double(Yt(Mt)); z = double(Gt(Mt));
maxPts = 20000;
if numel(z) > maxPts
    step = ceil(numel(z) / maxPts);
    xs = xs(1:step:end); ys = ys(1:step:end); z = z(1:step:end);
end

terms = zeros(0, 2);
for d = 1:order
    for i = 0:d
        terms(end + 1, :) = [d - i, i]; %#ok<AGROW>
    end
end
A = ones(numel(z), 1 + size(terms, 1));
for t = 1:size(terms, 1)
    A(:, t + 1) = (xs .^ terms(t, 1)) .* (ys .^ terms(t, 2));
end

keep = true(size(z));
coef = zeros(size(A, 2), 1);
for it = 1:max(1, iterations)
    coef = A(keep, :) \ z(keep);
    resid = z - A * coef;
    q = drscreen.cv.percentile(resid(keep), [10 90]);
    keep = resid >= q(1) & resid <= q(2);
    if nnz(keep) < size(A, 2) * 10
        break
    end
end

coef = single(coef);
field = coef(1) * ones(h, w, 'single');
for t = 1:size(terms, 1)
    field = field + coef(t + 1) * (Xn .^ terms(t, 1)) .* (Yn .^ terms(t, 2));
end
out = g ./ max(field, 1);
out(~m) = 1;
end
