function out = toModelInput(img, mask)
%TOMODELINPUT The three planes the networks consume -- not an RGB image.
%
%   plane 1  CLAHE(green, clip 3)      lesion and vessel contrast
%   plane 2  Ben-Graham green          local contrast, illumination-free
%   plane 3  L* of OpenCV 8-bit Lab    absolute luminance, keeps the exudate cue
%
%   The order is fixed by training. It was once stored reversed on disk
%   (a stray colour conversion before imwrite), so training and serving saw
%   mirror-image inputs; the Python test_train_serve_channel_parity guards the
%   same thing this function has to get right.

green = img(:, :, 2);
gClahe = drscreen.cv.clahe(green, 3.0, 8);

% Ben Graham: 4*img - 4*blur(img) + 128, sigma a fraction of the WIDTH so the
% operator is resolution-invariant. Only the green plane is used.
sigma = max(1.0, 0.033 * size(img, 2));
blurG = drscreen.cv.gaussianBlur(green, sigma);
bg = drscreen.cv.saturateU8(4 * double(green) - 4 * double(blurG) + 128);
if ~isempty(mask)
    bg(mask == 0) = 0;
end

lab = drscreen.cv.rgb2lab(img);
out = cat(3, gClahe, bg, lab(:, :, 1));
if ~isempty(mask)
    out(repmat(mask == 0, 1, 1, 3)) = 0;
end
end
