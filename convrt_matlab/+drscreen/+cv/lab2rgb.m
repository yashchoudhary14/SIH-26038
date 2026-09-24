function rgb = lab2rgb(lab)
%LAB2RGB cv2.cvtColor(LAB2BGR) for uint8, via OpenCV's integer lookup tables.
%
%   The inverse of drscreen.cv.rgb2lab. OpenCV's 8-bit path is fixed point
%   throughout: L maps to Y and f(Y) through a 256-entry table, a and b are
%   scaled by multiply-and-shift approximations of /500 and /200, f -> X, Z goes
%   through an integer table, XYZ -> linear RGB uses 12-bit integer
%   coefficients with a 14-bit descale, and a 4096-entry table applies the sRGB
%   gamma. The tables are built with the same rational constants OpenCV uses
%   (7.787 is 841/108, 903.3 is (29/3)^3), in float32 where OpenCV uses
%   softfloat.
%
%   Verified bit-exact against cv2 5.0 on all 16,777,216 8-bit Lab values. It
%   matters: clahe_lab runs on about a third of real photographs (anything the
%   gate flags for focus or contrast), and a float Lab->RGB is one level off on
%   3% of values -- enough to move lesion counts downstream.
%
%   Input: L scaled to 0..255 (L*255/100), a and b offset by 128. Output RGB.

persistent YF XZ GAM C
if isempty(YF)
    base = 2 ^ 14;

    % L -> (y, f(y)), both in 2^14 units
    i = (0:255)';
    yLo = drscreen.cv.cvRound(double(single(i * base * 20 * 9) / single(17 * 29 ^ 3)));
    fLo = drscreen.cv.cvRound(double(single(base) * ...
              (single(16) / single(116) + single(i * 5) / single(3 * 17 * 29))));
    fy = single(i * 100 * base) / single(255 * 116) + single(16 * base) / single(116);
    fHi = drscreen.cv.cvRound(double(fy));
    yHi = drscreen.cv.cvRound(double(fy .* fy .* fy / single(base * base)));
    lo = i <= 20;                                   % 8 * 255 / 100 = 20.4
    YF = [yLo .* lo + yHi .* ~lo, fLo .* lo + fHi .* ~lo];

    % f -> X or Z (before the white point), integer arithmetic as in C
    j = (-8145:(-8145 + base * 9 / 4 - 1))';
    lin = fix(j * 108 / 841) - fix(fix(base * 16 / 116) * 108 / 841);
    cub = fix(fix(fix(j .* j / base) .* j) / base);
    XZ = lin .* (j <= 3390) + cub .* (j > 3390);

    % inverse sRGB gamma: double pow, rounded to float32, x255 in float32
    x = (0:4095)' / 4096;
    g = x * 12.92;
    hi = x > 0.0031308;
    g(hi) = x(hi) .^ (1 / 2.4) * 1.055 - 0.055;
    GAM = drscreen.cv.cvRound(double(single(255) * single(g)));

    M = [ 3.240479 -1.53715  -0.498535;
         -0.969256  1.875991  0.041556;
          0.055648 -0.204043  1.057311];
    C = drscreen.cv.cvRound(4096 * M .* [0.950456 1.0 1.088754]);
end

sz = size(lab);
L = double(reshape(lab(:, :, 1), [], 1));
a = double(reshape(lab(:, :, 2), [], 1));
b = double(reshape(lab(:, :, 3), [], 1));

y = YF(L + 1, 1);
f = YF(L + 1, 2);
adiv = floor((5 * a * 53687 + 2 ^ 7) / 2 ^ 13) - 4194;      % ~ (a - 128) * 2^14 / 500
bdiv = floor((b * 41943 + 2 ^ 4) / 2 ^ 9) - 10485 + 1;      % ~ (b - 128) * 2^14 / 200
X = XZ(f + adiv + 8145 + 1);
Z = XZ(f - bdiv + 8145 + 1);

out = zeros(numel(L), 3);
for c = 1:3
    v = floor((C(c, 1) * X + C(c, 2) * y + C(c, 3) * Z + 2 ^ 13) / 2 ^ 14);   % CV_DESCALE
    out(:, c) = GAM(min(max(v, 0), 4095) + 1);
end
rgb = reshape(uint8(out), sz(1), sz(2), 3);
end
