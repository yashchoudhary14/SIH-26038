function [S, L] = conncomp(bw)
%CONNCOMP 8-connected components with OpenCV-style 0-based centroids.
%
%   S(i).Area, S(i).Centroid ([x y], 0-based, as cv2 reports them) and
%   S(i).PixelIdxList. Label order differs from OpenCV's raster order; nothing
%   downstream depends on it.

cc = bwconncomp(logical(bw), 8);
S = regionprops(cc, 'Area', 'Centroid', 'PixelIdxList');
for i = 1:numel(S)
    S(i).Centroid = S(i).Centroid - 1;
end
if nargout > 1
    L = labelmatrix(cc);
end
end
