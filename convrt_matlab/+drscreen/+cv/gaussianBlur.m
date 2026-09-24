function out = gaussianBlur(img, sigma)
%GAUSSIANBLUR cv2.GaussianBlur(img, (0,0), sigma) equivalent.
%
%   Kernel size follows OpenCV's rule, which depends on the input type:
%   round(sigma*6+1)|1 for uint8, round(sigma*8+1)|1 otherwise. That rule is
%   why the same sigma blurs an 8-bit and a float image differently in the
%   Python pipeline, and the port keeps the distinction. Border is
%   BORDER_REFLECT_101.
%
%   uint8 is bit-exact with OpenCV: its 8-bit path quantises the normalised
%   kernel to 8 fractional bits with error diffusion (so the taps sum to
%   exactly 256), convolves in exact integer arithmetic, and rounds once at 16
%   fractional bits. A double-precision blur rounded at the end differs from
%   that on ~1% of pixels by one grey level -- and the Ben-Graham plane the
%   grader reads multiplies the blur by 4, turning one level into four.
%   Verified identical to cv2 at four sigmas (tools/verify_cv_algorithms.py).
%
%   Float inputs are filtered in double precision (within 1e-4 of OpenCV's
%   float32 result) and returned as single.

isU8 = isa(img, 'uint8');
if isU8
    k = drscreen.cv.cvRound(sigma * 3 * 2 + 1);
else
    k = drscreen.cv.cvRound(sigma * 4 * 2 + 1);
end
k = bitor(k, 1);
x = (0:k-1) - (k - 1) / 2;
g = exp(-(x .^ 2) / (2 * sigma ^ 2));
g = g(:) / sum(g);

if isU8
    % getGaussianKernelFixedPoint_ED: 8 fractional bits, error diffusion,
    % centre tap absorbs the remainder so the sum is exactly 1.0.
    mult = 256;
    kf = zeros(k, 1);
    err = 0;
    s = 0;
    half = floor(k / 2);
    for i = 1:half
        adj = g(i) * mult + err;
        v0 = drscreen.cv.cvRound(adj);
        err = adj - v0;
        kf(i) = v0;
        kf(k + 1 - i) = v0;
        s = s + v0;
    end
    kf(half + 1) = mult - 2 * s;
    g = kf;
end

r = (k - 1) / 2;
[h, w, c] = size(img);
ri = drscreen.cv.reflect101(h, -r, h - 1 + r);
ci = drscreen.cv.reflect101(w, -r, w - 1 + r);
out = zeros(h, w, c);
for ch = 1:c
    P = double(img(ri, ci, ch));
    out(:, :, ch) = conv2(g, g', P, 'valid');
end

if isU8
    % Exact integers at 16 fractional bits; one rounding, as ufixedpoint32 -> uint8.
    out = uint8(min(max(floor((out + 32768) / 65536), 0), 255));
elseif isa(img, 'single')
    out = single(out);
end
end
