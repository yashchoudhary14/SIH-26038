function out = overlayCam(base, cam, alpha, threshold)
%OVERLAYCAM Blend a Grad-CAM map onto the image, suppressing low attention.
%
%   Thresholding matters: an un-thresholded jet map tints the whole retina,
%   which both looks alarming and hides the actual peak. The cut-off edge is
%   feathered so the overlay does not look like a stencil.

if nargin < 3 || isempty(alpha), alpha = 0.42; end
if nargin < 4 || isempty(threshold), threshold = 0.25; end
cam = min(max(double(cam), 0), 1);
cmap = jet(256);
idx = floor(cam * 255) + 1;
heat = reshape(cmap(idx(:), :), [size(cam) 3]) * 255;
m = double(cam >= threshold);
m = drscreen.cv.gaussianBlur(single(m), 3);
a = alpha * min(max(double(m), 0), 1);
out = uint8(min(max(double(base) .* (1 - a) + heat .* a, 0), 255));
end
