function vd = vesselDensity(img, mask, scale)
%VESSELDENSITY Cheap vesselness: bottom-hat on green, smoothed to a density field.

if nargin < 3, scale = 1.0; end
if size(img, 3) == 3
    g = single(img(:, :, 2));
else
    g = single(img);
end
k = max(3, bitor(fix(9 * scale), 1));
bg = drscreen.cv.morph(g, 'close', k);
resp = max(bg - g, 0);
resp(mask == 0) = 0;
vd = drscreen.cv.gaussianBlur(resp, max(3.0, 0.04 * max(size(img, 1), size(img, 2))));
end
