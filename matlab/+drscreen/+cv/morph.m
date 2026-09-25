function out = morph(img, op, k)
%MORPH OpenCV-compatible morphology with an elliptical element of size k.
%
%   op: 'erode' | 'dilate' | 'open' | 'close'. Works on uint8, single, double
%   and logical. Border handling matches OpenCV's default (the border never
%   wins an erosion or a dilation), which is also what imerode/imdilate do.

se = strel('arbitrary', drscreen.cv.ellipse(k));
switch lower(op)
    case 'erode',  out = imerode(img, se);
    case 'dilate', out = imdilate(img, se);
    case 'open',   out = imdilate(imerode(img, se), se);
    case 'close',  out = imerode(imdilate(img, se), se);
    otherwise
        error('drscreen:cv:morph', 'Unknown morphology op "%s".', op);
end
end
