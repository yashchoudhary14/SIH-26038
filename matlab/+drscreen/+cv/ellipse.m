function se = ellipse(k)
%ELLIPSE cv2.getStructuringElement(MORPH_ELLIPSE, (k, k)) as a logical k-by-k.
%
%   Not strel('disk'): MATLAB's disk is a different digital approximation, and
%   every morphological criterion in the quality gate and the landmark
%   detector was tuned with OpenCV's shape.

if numel(k) > 1, k = k(1); end
r = floor(k / 2);
c = floor(k / 2);
se = false(k, k);
invR2 = 0;
if r > 0, invR2 = 1 / (r * r); end
for i = 0:k-1
    dy = i - r;
    if abs(dy) <= r
        dx = drscreen.cv.cvRound(c * sqrt((r * r - dy * dy) * invR2));
        j1 = max(c - dx, 0);
        j2 = min(c + dx + 1, k);
        se(i + 1, j1 + 1:j2) = true;
    end
end
end
