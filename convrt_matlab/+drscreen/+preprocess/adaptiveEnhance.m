function [out, applied] = adaptiveEnhance(img, mask, issues, always)
%ADAPTIVEENHANCE Apply only the corrections the quality gate says are needed.
%
%   [enhanced, applied] = drscreen.preprocess.adaptiveEnhance(rgb, mask, issues)
%
%   Unconditional enhancement is harmful: CLAHE on an already well-exposed
%   image amplifies sensor noise into structures the microaneurysm detector
%   then reports, and Ben-Graham normalisation on a clean image destroys the
%   absolute colour cue that separates hard exudates (yellow, sharp) from
%   cotton-wool spots (pale, fuzzy). So each operator runs only when a
%   specific quality criterion asked for it, and the list of what ran is
%   returned for the audit trail.
%
%   issues -> operators:
%     (always)                                      grey_world
%     illumination, under_exposure, over_exposure  illumination_normalize
%     under_exposure, over_exposure                 auto_exposure
%     contrast, focus                               clahe_lab
%     noise                                         denoise

if nargin < 3, issues = {}; end
if nargin < 4, always = {'grey_world'}; end
applied = {};
out = img;

if ismember('grey_world', always)
    out = greyWorld(out, mask);
    applied{end + 1} = 'grey_world';
end
if any(ismember({'illumination', 'under_exposure', 'over_exposure'}, issues))
    out = illuminationNormalize(out, mask);
    applied{end + 1} = 'illumination_normalize';
end
if any(ismember({'under_exposure', 'over_exposure'}, issues))
    out = autoExposure(out, mask);
    applied{end + 1} = 'auto_exposure';
end
if any(ismember({'contrast', 'focus'}, issues))
    out = claheLab(out, 2.5);
    applied{end + 1} = 'clahe_lab';
end
if ismember('noise', issues)
    out = denoise(out);
    applied{end + 1} = 'denoise';
end
out = applyMask(out, mask);
end


% ========================================================================
function out = applyMask(img, mask)
out = img;
if ~isempty(mask)
    out(repmat(mask == 0, 1, 1, size(img, 3))) = 0;
end
end


function out = greyWorld(img, mask)
% Shades-of-grey colour constancy (p = 6). Different cameras have very
% different white balance; without this a classifier can learn "this hue
% implies this hospital implies this prevalence" -- the shortcut that
% collapses on Messidor-2.
%
% Reproduces numpy's float32 arithmetic exactly, because it is ill-conditioned:
% the sum of pixel^6 over ~200k pixels reaches 1e19, where float32 rounding
% moves the channel norms by ~2e-5 relative -- enough to change 0.3% of pixels
% by one level, which CLAHE downstream amplifies to several. numpy computes
% powf(x, 6) correctly rounded and sums sequentially in row-major pixel order;
% cumsum reproduces that order (sum() does not -- it is more accurate).
x = single(img);
sel = reshape(permute(x, [2 1 3]), [], 3);          % row-major, as numpy indexes
if ~isempty(mask)
    mt = mask.';
    sel = sel(mt(:) > 0, :);
end
if isempty(sel)
    out = img;
    return
end
pw = single(double(max(sel, single(1e-6))) .^ 6);   % correctly rounded x^6
total = cumsum(pw, 1);
m = total(end, :) ./ single(size(pw, 1));
norms = single(double(m) .^ double(single(1 / 6)));
nmean = ((norms(1) + norms(2)) + norms(3)) / single(3);
scale = nmean ./ max(norms, single(1e-6));
out = uint8(floor(min(max(x .* reshape(scale, 1, 1, 3), 0), 255)));
out = applyMask(out, mask);
end


function out = illuminationNormalize(img, mask)
% Flat-field correction: divide out the low-frequency illumination field.
% Multiplicative, which is physically right for a reflectance model -- the
% subtractive Ben-Graham variant also removes large blot haemorrhages. Uses a
% normalised convolution so the black surround does not drag the background
% estimate down near the rim.
[h, w, ~] = size(img);
sigma = max(3.0, 0.05 * max(h, w));
x = single(img);
m = single(mask > 0);
num = drscreen.cv.gaussianBlur(x .* m, sigma);
den = drscreen.cv.gaussianBlur(m, sigma);
background = num ./ max(den, single(1e-3));
target = zeros(1, 1, 3, 'single');
for c = 1:3
    ch = x(:, :, c);
    target(c) = single(median(double(ch(mask > 0))));
end
out = x ./ max(background, single(1.0)) .* target;
out = uint8(floor(min(max(out, 0), 255)));
out = applyMask(out, mask);
end


function out = autoExposure(img, mask)
% Push the retinal median luminance to 110 via gamma. Gain would clip the
% optic disc; gamma keeps the highlight roll-off that distinguishes a bright
% disc from a hard exudate.
target = 110.0;
gray = drscreen.cv.rgb2gray(img);
if isempty(mask)
    sel = double(gray(:));
else
    sel = double(gray(mask > 0));
end
sel = sel(sel > 5);
if numel(sel) < 100
    out = img;
    return
end
med = median(sel);
if med < 1
    out = img;
    return
end
g = log(max(med, 1) / 255) / log(max(target, 1) / 255);
g = min(max(g, 0.4), 2.5);
lut = uint8(floor(min(max(((0:255) / 255) .^ (1 / max(g, 1e-3)) * 255, 0), 255)));
out = lut(double(img) + 1);
out = reshape(out, size(img));
end


function out = claheLab(img, clip)
% CLAHE on L* only, so hue and chroma -- and therefore lesion colour -- survive.
lab = drscreen.cv.rgb2lab(img);
lab(:, :, 1) = drscreen.cv.clahe(lab(:, :, 1), clip, 8);
out = drscreen.cv.lab2rgb(lab);
end


function out = denoise(img)
% Edge-preserving denoise. Non-local means rather than a Gaussian, because
% microaneurysms are 3-8 px blobs and any isotropic smoothing wide enough to
% suppress read noise erases them too.
%
% APPROXIMATION: OpenCV's fastNlMeansDenoisingColored (h = hColor = 3, 7x7
% template, 21x21 search) has no bit-compatible MATLAB counterpart. imnlmfilt
% with the same windows is the nearest equivalent. This branch only runs when
% the gate flags sensor noise, which is rare on fundus cameras.
% (try/catch, not exist(): exist misreports P-coded and compiled functions.)
persistent warned
out = img;
try
    lab = rgb2lab(img);
    for c = 1:3
        lab(:, :, c) = imnlmfilt(lab(:, :, c), 'DegreeOfSmoothing', 3, ...
                                 'SearchWindowSize', 21, 'ComparisonWindowSize', 7);
    end
    out = im2uint8(lab2rgb(lab));
catch err
    if isempty(warned)
        warned = true;
        warning('drscreen:denoise', 'Denoise skipped (%s); image passed through.', err.message);
    end
end
end
