function out = annotateLesions(base, probs, lm, threshold, minArea)
%ANNOTATELESIONS Lesion outlines in the fixed colour code, plus disc and fovea.
%
%   A reviewer needs to know *where* to look before *why*. Each detected
%   lesion is outlined in its class colour (small ones are ringed, since a
%   3-pixel outline is invisible at review scale), the optic disc is circled,
%   the fovea marked, and a one-disc-diameter ring is drawn around the fovea --
%   the CSME boundary a reviewer checks first.

if nargin < 4 || isempty(threshold), threshold = 0.5; end
if nargin < 5 || isempty(minArea), minArea = 3; end
C = drscreen.constants();
out = base;
[h, w, ~] = size(out);

for ci = 1:numel(C.LESION_CLASSES)
    name = C.LESION_CLASSES{ci};
    colour = C.LESION_COLORS.(name);
    binary = probs(:, :, ci) >= threshold;
    if ~any(binary(:))
        continue
    end
    S = drscreen.cv.conncomp(binary);
    keep = false(h, w);
    for k = 1:numel(S)
        if S(k).Area >= minArea
            if S(k).Area < 12
                r = max(4, ceil(sqrt(S(k).Area / pi)) + 3);
                out = drawCircle(out, S(k).Centroid(1), S(k).Centroid(2), r, colour);
            else
                keep(S(k).PixelIdxList) = true;
            end
        end
    end
    if any(keep(:))
        out = paint(out, bwperim(keep, 8), colour);
    end
end

if nargin >= 3 && ~isempty(lm)
    green = [0 255 0];
    out = drawCircle(out, lm.disc_xy(1), lm.disc_xy(2), lm.disc_radius, green);
    out = drawCross(out, lm.fovea_xy(1), lm.fovea_xy(2), 8, green);
    out = drawCircle(out, lm.fovea_xy(1), lm.fovea_xy(2), lm.disc_diameter_px, [0 200 0]);
end
end


function img = paint(img, mask, colour)
for c = 1:3
    ch = img(:, :, c);
    ch(mask) = colour(c);
    img(:, :, c) = ch;
end
end


function img = drawCircle(img, cx, cy, r, colour)
% 1-px circle outline; (cx, cy) are 0-based pixel coordinates.
[h, w, ~] = size(img);
n = max(32, ceil(2 * pi * r * 1.5));
t = linspace(0, 2 * pi, n);
x = round(cx + r * cos(t)) + 1;
y = round(cy + r * sin(t)) + 1;
ok = x >= 1 & x <= w & y >= 1 & y <= h;
m = false(h, w);
m(sub2ind([h w], y(ok), x(ok))) = true;
img = paint(img, m, colour);
end


function img = drawCross(img, cx, cy, half, colour)
[h, w, ~] = size(img);
m = false(h, w);
xs = round(cx) + (-half:half) + 1;
ys = round(cy) + (-half:half) + 1;
xr = round(cx) + 1; yr = round(cy) + 1;
if yr >= 1 && yr <= h
    m(yr, xs(xs >= 1 & xs <= w)) = true;
end
if xr >= 1 && xr <= w
    m(ys(ys >= 1 & ys <= h), xr) = true;
end
img = paint(img, m, colour);
end
