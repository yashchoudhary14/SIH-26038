function out = filter2d(img, K)
%FILTER2D cv2.filter2D(img, -1, K): correlation with a REFLECT_101 border.

img = double(img);
[kh, kw] = size(K);
ah = floor((kh - 1) / 2);
aw = floor((kw - 1) / 2);
[h, w] = size(img);
P = img(drscreen.cv.reflect101(h, -ah, h - 1 + (kh - 1 - ah)), ...
        drscreen.cv.reflect101(w, -aw, w - 1 + (kw - 1 - aw)));
out = conv2(P, rot90(K, 2), 'valid');
end
