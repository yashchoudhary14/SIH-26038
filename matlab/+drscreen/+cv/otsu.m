function t = otsu(I)
%OTSU Threshold returned by cv2.threshold(I, 0, 255, THRESH_BINARY|THRESH_OTSU).
%
%   OpenCV's getThreshVal_Otsu_8u, reproduced so the returned integer is the
%   same one (graythresh normalises differently and can land one level off).
%   Pixels strictly greater than t are foreground.

h = accumarray(double(I(:)) + 1, 1, [256 1]) / numel(I);
mu = sum((0:255)' .* h);
q1 = 0; mu1 = 0; maxSigma = 0; t = 0;
for i = 0:255
    p = h(i + 1);
    mu1 = mu1 * q1;
    q1 = q1 + p;
    q2 = 1 - q1;
    if min(q1, q2) < eps('single') || max(q1, q2) > 1 - eps('single')
        continue
    end
    mu1 = (mu1 + i * p) / q1;
    mu2 = (mu - q1 * mu1) / q2;
    sigma = q1 * q2 * (mu1 - mu2) ^ 2;
    if sigma > maxSigma
        maxSigma = sigma;
        t = i;
    end
end
end
