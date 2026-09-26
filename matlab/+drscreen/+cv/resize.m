function out = resize(img, outHW, method)
%RESIZE cv2.resize equivalent: out = resize(img, [H W], method).
%
%   method: 'area' | 'cubic' | 'linear' | 'nearest'.
%
%   Written as two sparse interpolation matrices, out = Wy * img * Wx', so every
%   method shares one code path and the coordinate conventions are explicit.
%   MATLAB's imresize is not used: its bicubic kernel is Keys a=-0.5 where
%   OpenCV uses a=-0.75, and its antialiased 'box' includes whole pixels by
%   centre where OpenCV's INTER_AREA weights fractional overlap. Either would
%   hand the networks a different image from the one they were trained on.
%
%   'linear' matches both cv2 INTER_LINEAR and PyTorch bilinear with
%   align_corners=False (used to upsample the Grad-CAM map).

[h, w, c] = size(img);
H = outHW(1);
W = outHW(2);

% INTER_AREA downscale by a non-integer factor, on 8-bit or float32 images:
% OpenCV's resizeArea accumulates in float32 in a fixed order, and a few
% outputs per photograph sit close enough to .5 that the order decides the
% rounding. Reproduce it (see areaFloat32). The geometry stage depends on this.
if strcmpi(method, 'area') && h >= H && w >= W && (isa(img, 'uint8') || isa(img, 'single')) ...
        && ~(mod(h, H) == 0 && mod(w, W) == 0)
    out = areaFloat32(img, H, W);
    return
end

Wy = drscreen.cv.resizeWeights(h, H, method);
Wx = drscreen.cv.resizeWeights(w, W, method);

cls = class(img);
out = zeros(H, W, c);
for ch = 1:c
    out(:, :, ch) = full(Wy * double(img(:, :, ch)) * Wx');
end

switch cls
    case 'uint8'
        out = drscreen.cv.saturateU8(out);
    case 'single'
        out = single(out);
    case 'logical'
        out = out > 0.5;
end
end


function out = areaFloat32(img, H, W)
% cv::resizeArea (ResizeArea_Invoker) for a non-integer downscale. Weights are
% the computeResizeAreaTab fractions rounded to float32. Each source row is
% first reduced horizontally -- buf[dx] += S[sx] * alpha, sequentially over sx
% in float32 -- then rows are accumulated vertically in source order,
% sum[dx] += beta * buf[dx], and the result is rounded half-to-even. Padding
% entries (index 1, weight 0) add an exact zero and leave the sums unchanged.
[h, w, c] = size(img);
[xi, xa] = areaTable(w, W);
[yi, ya] = areaTable(h, H);
out = zeros(H, W, c, 'single');
for ch = 1:c
    S = single(img(:, :, ch));
    buf = zeros(h, W, 'single');
    for k = 1:size(xi, 1)
        buf = buf + S(:, xi(k, :)) .* xa(k, :);
    end
    acc = zeros(H, W, 'single');
    for k = 1:size(yi, 1)
        acc = acc + ya(k, :)' .* buf(yi(k, :), :);
    end
    out(:, :, ch) = acc;
end
if isa(img, 'uint8')
    out = drscreen.cv.saturateU8(out);
end
end


function [idx, alpha] = areaTable(n, N)
% K x N source indices and float32 weights, each column in increasing source
% order (the order OpenCV adds them in).
[r, col, v] = find(drscreen.cv.resizeWeights(n, N, 'area'));
[~, o] = sortrows([r, col]);
r = r(o); col = col(o); v = v(o);
cnt = accumarray(r, 1, [N 1]);
first = cumsum(cnt) - cnt;
pos = (1:numel(r))' - first(r);
K = max(cnt);
idx = ones(K, N);
alpha = zeros(K, N, 'single');
idx(sub2ind([K N], pos, r)) = col;
alpha(sub2ind([K N], pos, r)) = single(v);
end
