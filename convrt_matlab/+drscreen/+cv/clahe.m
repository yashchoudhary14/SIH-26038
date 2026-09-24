function out = clahe(I, clipLimit, tiles)
%CLAHE cv2.createCLAHE(clipLimit, (tiles, tiles)).apply(I) for uint8 I.
%
%   MATLAB's adapthisteq is a different algorithm (different clip-limit scale,
%   different distribution, different tile interpolation). CLAHE builds two of
%   the three planes the grader consumes, so its output has to be OpenCV's,
%   not merely similar to it. This follows OpenCV's CLAHE_CalcLut_Body and
%   CLAHE_Interpolation_Body step for step, including two quirks:
%
%   * When the image is not a whole number of tiles, OpenCV pads bottom/right
%     with REFLECT_101 by tiles - mod(size, tiles) on BOTH axes -- a full extra
%     tile on an axis that did divide evenly. LUTs are built on that padded
%     image; interpolation reads the original.
%   * The LUT scale and the interpolation weights are float32, and ties round
%     to even.

if nargin < 3, tiles = 8; end
I = uint8(I);
[h, w] = size(I);
tx = tiles; ty = tiles;

if mod(w, tx) == 0 && mod(h, ty) == 0
    src = I;
else
    ri = drscreen.cv.reflect101(h, 0, h + (ty - mod(h, ty)) - 1);
    ci = drscreen.cv.reflect101(w, 0, w + (tx - mod(w, tx)) - 1);
    src = I(ri, ci);
end
tileH = floor(size(src, 1) / ty);
tileW = floor(size(src, 2) / tx);
tileArea = tileH * tileW;

clip = 0;
if clipLimit > 0
    clip = max(fix(clipLimit * tileArea / 256), 1);
end
lutScale = single(255) / single(tileArea);

luts = zeros(256, ty, tx);
for j = 0:ty-1
    for i = 0:tx-1
        block = src(j*tileH + (1:tileH), i*tileW + (1:tileW));
        hist = accumarray(double(block(:)) + 1, 1, [256 1]);
        if clip > 0
            excess = sum(max(hist - clip, 0));
            hist = min(hist, clip);
            redist = floor(excess / 256);
            residual = excess - redist * 256;
            hist = hist + redist;
            if residual > 0
                step = max(floor(256 / residual), 1);
                bins = 1:step:256;
                bins = bins(1:min(residual, numel(bins)));
                hist(bins) = hist(bins) + 1;
            end
        end
        cs = cumsum(hist);
        luts(:, j + 1, i + 1) = double(drscreen.cv.saturateU8(double(single(cs) * lutScale)));
    end
end

% Bilinear interpolation between the four surrounding tile LUTs, in float32.
invTh = single(1) / single(tileH);
invTw = single(1) / single(tileW);
tyf = single(0:h-1)' * invTh - single(0.5);
txf = single(0:w-1) * invTw - single(0.5);
ty1 = floor(tyf); ya = tyf - ty1; ya1 = 1 - ya;
tx1 = floor(txf); xa = txf - tx1; xa1 = 1 - xa;
ty2 = min(ty1 + 1, ty - 1); ty1 = max(ty1, 0);
tx2 = min(tx1 + 1, tx - 1); tx1 = max(tx1, 0);

p = double(I);                                   % h x w, 0..255
lin = @(TY, TX) p + 1 + 256 * (double(TY) + ty * double(TX));   % into luts(:)
L = single(luts(:));
v11 = L(lin(repmat(ty1, 1, w), repmat(tx1, h, 1)));
v12 = L(lin(repmat(ty1, 1, w), repmat(tx2, h, 1)));
v21 = L(lin(repmat(ty2, 1, w), repmat(tx1, h, 1)));
v22 = L(lin(repmat(ty2, 1, w), repmat(tx2, h, 1)));

XA = repmat(xa, h, 1); XA1 = repmat(xa1, h, 1);
YA = repmat(ya, 1, w); YA1 = repmat(ya1, 1, w);
res = (v11 .* XA1 + v12 .* XA) .* YA1 + (v21 .* XA1 + v22 .* XA) .* YA;
out = drscreen.cv.saturateU8(double(res));
end
