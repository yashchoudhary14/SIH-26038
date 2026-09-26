function [img, msk, fov] = standardize(raw, sz, fov)
%STANDARDIZE Full geometric normalisation: detect -> crop -> square-pad -> resize.
%
%   [img, mask, fov] = drscreen.preprocess.standardize(rgb, 512)
%
%   Square padding matters clinically: optic-disc diameter is the unit the
%   severe-NPDR and CSME criteria are written in, so a stretched image corrupts
%   the geometry the grading rules are defined in. The resized mask is eroded
%   by ~1% to drop the aliased rim, a classic source of false exudates.

if nargin < 2 || isempty(sz), sz = 512; end
if nargin < 3 || isempty(fov)
    fov = drscreen.preprocess.detectFov(raw);
end
[h, w, ~] = size(raw);

% crop to the retina with a 2% margin
x0 = fov.bbox(1); y0 = fov.bbox(2); x1 = fov.bbox(3); y1 = fov.bbox(4);
px = fix(0.02 * (x1 - x0));
py = fix(0.02 * (y1 - y0));
x0 = max(0, x0 - px); y0 = max(0, y0 - py);
x1 = min(w, x1 + px); y1 = min(h, y1 + py);
cropped = raw(y0+1:y1, x0+1:x1, :);
mask = fov.mask(y0+1:y1, x0+1:x1);

% zero-pad to a square
[ch, cw, nc] = size(cropped);
side = max(ch, cw);
top = floor((side - ch) / 2);
left = floor((side - cw) / 2);
sq = zeros(side, side, nc, 'like', cropped);
sq(top+1:top+ch, left+1:left+cw, :) = cropped;
sm = zeros(side, side, 'uint8');
sm(top+1:top+ch, left+1:left+cw) = mask;

if side > sz
    method = 'area';
else
    method = 'cubic';
end
img = drscreen.cv.resize(sq, [sz sz], method);
msk = drscreen.cv.resize(sm, [sz sz], 'nearest');

er = max(3, bitor(floor(sz / 100), 1));
msk = drscreen.cv.morph(msk, 'erode', er);
img(repmat(msk == 0, 1, 1, size(img, 3))) = 0;
end
