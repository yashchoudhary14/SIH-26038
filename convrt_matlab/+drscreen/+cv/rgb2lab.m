function lab = rgb2lab(rgb)
%RGB2LAB cv2.cvtColor(BGR2LAB) for uint8, via OpenCV's integer lookup tables.
%
%   OpenCV's 8-bit path does not evaluate the Lab formula in floating point.
%   It gamma-expands through a 256-entry table (x8 fixed point), mixes to XYZ
%   with 12-bit integer coefficients, takes the cube root through a second
%   table (15-bit), and descales with rounding. This reproduces that path.
%
%   Verified bit-exact against cv2 5.0 on all 16.7 million 8-bit colours, in
%   all three channels. L is the third plane of the grader's input, and a, b
%   survive clahe_lab into the enhanced image, so exactness here keeps MATLAB's
%   model input identical to Python's; a floating-point Lab differs from
%   OpenCV on 12% of pixels.
%
%   Output: L scaled to 0..255 (L*255/100), a and b offset by 128.

persistent tab cb C
if isempty(tab)
    x = (0:255)' / 255;
    gam = x / 12.92;
    hi = x > 0.04045;
    gam(hi) = ((x(hi) + 0.055) / 1.055) .^ 2.4;
    tab = drscreen.cv.cvRound(255 * 8 * gam);               % gamma_shift = 3

    n = 256 * 3 / 2 * 8;                                     % LAB_CBRT_TAB_SIZE_B
    xs = (0:n-1)' / (255 * 8);
    f = xs * 7.787 + 16 / 116;
    hi = xs >= 0.008856;
    f(hi) = nthroot(xs(hi), 3);
    cb = min(max(drscreen.cv.cvRound(32768 * f), 0), 65535);  % lab_shift2 = 15
    % OpenCV evaluates this cube root in its own float32 "softfloat" library.
    % Four of the 3072 entries fall within 1e-4 of a rounding tie (entries 49,
    % 324, 628, 2079), where that library and IEEE double can land on opposite
    % sides; two of them do. Set to OpenCV's values -- with them the whole
    % conversion is bit-exact on all 16.7M colours, a and b included.
    cb(49 + 1) = 9454;
    cb(628 + 1) = 22126;

    M = [0.412453 0.357580 0.180423;
         0.212671 0.715160 0.072169;
         0.019334 0.119193 0.950227];
    wp = [0.950456; 1.0; 1.088754];
    C = drscreen.cv.cvRound(4096 * M ./ wp);                % lab_shift = 12
end

desc = @(v, s) floor((v + 2 ^ (s - 1)) / 2 ^ s);            % CV_DESCALE

R = tab(double(rgb(:, :, 1)) + 1);
G = tab(double(rgb(:, :, 2)) + 1);
B = tab(double(rgb(:, :, 3)) + 1);
fX = cb(desc(R * C(1, 1) + G * C(1, 2) + B * C(1, 3), 12) + 1);
fY = cb(desc(R * C(2, 1) + G * C(2, 2) + B * C(2, 3), 12) + 1);
fZ = cb(desc(R * C(3, 1) + G * C(3, 2) + B * C(3, 3), 12) + 1);

Lscale = floor((116 * 255 + 50) / 100);
Lshift = -floor((16 * 255 * 2 ^ 15 + 50) / 100);
L = desc(Lscale * fY + Lshift, 15);
a = desc(500 * (fX - fY) + 128 * 2 ^ 15, 15);
b = desc(200 * (fY - fZ) + 128 * 2 ^ 15, 15);

sz = size(rgb);
lab = cat(3, reshape(uint8(min(max(L, 0), 255)), sz(1:2)), ...
             reshape(uint8(min(max(a, 0), 255)), sz(1:2)), ...
             reshape(uint8(min(max(b, 0), 255)), sz(1:2)));
end
