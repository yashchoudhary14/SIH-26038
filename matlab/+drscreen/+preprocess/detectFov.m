function fov = detectFov(img, threshold)
%DETECTFOV Locate the circular retinal aperture in an RGB fundus frame.
%
%   Fundus cameras image a circular aperture onto a rectangular sensor, so
%   every raw frame carries a black surround whose thickness depends on the
%   camera. Feeding that surround to a CNN lets it identify the camera rather
%   than the pathology -- the usual cause of the APTOS-to-Messidor collapse --
%   so the disc is recovered analytically and the same code path serves every
%   camera.
%
%   Strategy: threshold the per-pixel channel maximum (the retina is bright in
%   at least one channel even under-exposed), clean with elliptical opening and
%   closing, keep the largest blob, fill its holes, and fit a circle from its
%   area and centroid (more stable than a Hough circle when the sensor clips
%   the aperture).
%
%   All coordinates are 0-based, exactly as the Python pipeline reports them:
%   cx, cy, radius; bbox = [x0 y0 x1 y1] with x1, y1 exclusive.

if size(img, 3) == 1
    gray = img;
else
    gray = max(img, [], 3);
end
gray = uint8(gray);
[h, w] = size(gray);

if nargin < 2 || isempty(threshold)
    % Otsu on a blurred copy, floored: a nearly black frame would otherwise get
    % a threshold of ~2 and "detect" sensor noise.
    blur = drscreen.cv.gaussianBlur(gray, max(1.0, min(h, w) / 200.0));
    t = drscreen.cv.otsu(blur);
    threshold = fix(min(max(t * 0.45, 8), 60));
end

binary = gray > threshold;
k = max(3, bitor(floor(min(h, w) / 100), 1));
binary = drscreen.cv.morph(binary, 'open', k);
binary = drscreen.cv.morph(binary, 'close', k);

% Largest 8-connected component, holes filled (a dark choroid or a large
% haemorrhage must not punch a hole in the FOV mask).
S = drscreen.cv.conncomp(binary);
if ~isempty(S)
    [~, big] = max([S.Area]);
    binary = false(h, w);
    binary(S(big).PixelIdxList) = true;
    binary = imfill(binary, 'holes');
end

area = nnz(binary);
if area < 0.01 * h * w
    % Degenerate: treat the whole frame as retina rather than fail the case.
    fov = struct('cx', w / 2, 'cy', h / 2, 'radius', min(h, w) / 2, ...
                 'mask', 255 * ones(h, w, 'uint8'), 'bbox', [0 0 w h], ...
                 'coverage', 1.0, 'fill_ratio', 1.0, ...
                 'clipped_sides', [true true true true]);
    return
end

[ys, xs] = find(binary);
xs = xs - 1;                       % 0-based, as numpy.nonzero reports
ys = ys - 1;
x0 = min(xs); x1 = max(xs) + 1;
y0 = min(ys); y1 = max(ys) + 1;
cx = mean(xs);
cy = mean(ys);

% Radius from area (robust to clipping) blended with the half-span of the
% widest extent (robust to a partially imaged pupil).
rArea = sqrt(area / pi);
rSpan = max(x1 - x0, y1 - y0) / 2;
radius = max(rArea, 0.85 * rSpan);

% How much of the ideal circle landed on the sensor?
[XX, YY] = meshgrid(0:w-1, 0:h-1);
inFrame = nnz((XX - cx) .^ 2 + (YY - cy) .^ 2 <= radius ^ 2);
coverage = min(max(inFrame / max(pi * radius ^ 2, 1), 0), 1);

margin = max(2, floor(0.005 * min(h, w)));
clipped = [x0 <= margin, y0 <= margin, x1 >= w - margin, y1 >= h - margin];

fov = struct('cx', cx, 'cy', cy, 'radius', radius, ...
             'mask', uint8(binary) * 255, 'bbox', [x0 y0 x1 y1], ...
             'coverage', coverage, 'fill_ratio', area / (h * w), ...
             'clipped_sides', clipped);
end
