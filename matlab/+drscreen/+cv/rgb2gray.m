function g = rgb2gray(rgb)
%RGB2GRAY OpenCV's 8-bit grey conversion, bit-exact.
%
%   OpenCV 5 computes (R*9798 + G*19235 + B*3735 + 16384) >> 15 in integers --
%   verified identical to cv2.cvtColor(BGR2GRAY) over all 16.7 million 8-bit
%   colours (tools/verify_cv_algorithms.py). The older 14-bit coefficients
%   (4899, 9617, 1868) disagree on 0.26% of colours, and MATLAB's rgb2gray on
%   more; the quality gate's exposure and illumination criteria read this plane.

if size(rgb, 3) == 1
    g = uint8(rgb);
    return
end
R = uint32(rgb(:, :, 1));
G = uint32(rgb(:, :, 2));
B = uint32(rgb(:, :, 3));
g = uint8(bitshift(R * 9798 + G * 19235 + B * 3735 + 16384, -15));
end
