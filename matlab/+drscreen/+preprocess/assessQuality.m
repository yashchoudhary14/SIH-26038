function Q = assessQuality(img, mask, fov, lm, thresholds)
%ASSESSQUALITY Interpretable image-quality gate with recapture advice.
%
%   Q = drscreen.preprocess.assessQuality(rgb512, mask, fov, landmarks)
%
%   Physics-first rather than a learned quality classifier, for three reasons:
%   the problem statement wants *recapture feedback*, which a logit cannot give
%   ("the macula is out of frame" needs a per-criterion measure); it runs on
%   an edge device before any heavy model, so it must be milliseconds on CPU;
%   and real field failures (media opacity, miosis, flare, dirty optics) do not
%   look like the Gaussian blur a synthetic quality model is trained on.
%
%   Correctable defects (illumination, exposure, contrast, noise) are reasons
%   to enhance, never to send a patient away; only non-correctable ones (focus,
%   field of view, macula, artifact) can make the first pass ungradeable. The
%   pipeline re-runs this gate after enhancement and rejects only what survived.
%
%   Q has fields scores, verdicts, overall, gradeable, needs_enhancement,
%   issues, advice, confidence -- the Python QualityReport, field for field.

C = drscreen.constants();
T = drscreen.preprocess.qualityThresholds();
if nargin >= 5 && ~isempty(thresholds)
    f = fieldnames(thresholds);
    for i = 1:numel(f)
        T.(f{i}) = thresholds.(f{i});
    end
end
if nargin < 3 || isempty(fov)
    fov = drscreen.preprocess.detectFov(img);
end
if nargin < 2 || isempty(mask)
    mask = fov.mask;
end
if nargin < 4 || isempty(lm)
    lm = drscreen.preprocess.locateLandmarks(img, mask);
end

[under, over] = exposureScores(img, mask);
scores = struct( ...
    'focus',          focusScore(img, mask), ...
    'illumination',   illuminationScore(img, mask), ...
    'contrast',       contrastScore(img, mask), ...
    'fov',            fovScore(fov), ...
    'macula',         maculaVisibility(img, mask, lm), ...
    'artifact',       artifactScore(img, mask), ...
    'under_exposure', under, ...
    'over_exposure',  over, ...
    'noise',          noiseScore(img, mask));

correctable = {'illumination', 'under_exposure', 'over_exposure', 'contrast', 'noise'};
names = fieldnames(scores);
verdicts = struct();
issues = {};
advice = {};
hardFail = {};
softFail = {};
for i = 1:numel(names)
    k = names{i};
    v = scores.(k);
    th = T.(k);
    if v < th(1)
        verdicts.(k) = 'fail';
        issues{end + 1} = k; %#ok<AGROW>
        if ~ismember(k, correctable)
            hardFail{end + 1} = k; %#ok<AGROW>
            if isfield(C.RECAPTURE_ADVICE, k)
                advice{end + 1} = C.RECAPTURE_ADVICE.(k); %#ok<AGROW>
            else
                advice{end + 1} = sprintf('Recapture: %s inadequate.', k); %#ok<AGROW>
            end
        else
            softFail{end + 1} = k; %#ok<AGROW>
        end
    elseif v < th(2)
        verdicts.(k) = 'borderline';
        issues{end + 1} = k; %#ok<AGROW>
    else
        verdicts.(k) = 'pass';
    end
end

nBorder = sum(strcmp(struct2cell(verdicts), 'borderline'));
if ~isempty(hardFail)
    overall = 'ungradeable'; gradeable = false;
elseif ~isempty(softFail) || nBorder > 0
    overall = 'borderline'; gradeable = true;
else
    overall = 'good'; gradeable = true;
end

% Soft-min of the criteria: quality is a conjunction, so one bad axis dominates.
vals = cellfun(@(k) scores.(k), names);
beta = 8.0;
conf = -log(sum(exp(-beta * vals)) / numel(vals)) / beta;

rounded = struct();
for i = 1:numel(names)
    rounded.(names{i}) = round(scores.(names{i}) * 1e4) / 1e4;
end

Q = struct( ...
    'scores', rounded, ...
    'verdicts', verdicts, ...
    'overall', overall, ...
    'gradeable', gradeable, ...
    'needs_enhancement', nBorder > 0 || ~isempty(softFail) || ~isempty(hardFail), ...
    'issues', {issues}, ...
    'advice', {advice}, ...
    'confidence', round(min(max(conf, 0), 1) * 1e4) / 1e4);
end


% ========================================================================
% Individual criteria. Each returns a score in [0, 1], higher is better.
% ========================================================================
function sel = retinaPixels(plane, mask)
sel = plane(mask > 0);
if isempty(sel)
    sel = plane(:);
end
end


function s = focusScore(img, mask)
% Two conjoined terms. The ratio of high- to mid-band energy is content
% independent but NOT monotone in blur: both bands collapse under heavy defocus
% and their ratio climbs back up, so a retina blurred to sigma = 0.08*W scored
% better than the same retina in focus. Absolute high-band energy relative to
% the retina's own intensity spread IS monotone, and fixes it.
m = mask > 0;
g = single(img(:, :, 2));
g(~m) = 0;
if nnz(m) < 100
    s = 0;
    return
end
sc = max(size(img, 1), size(img, 2)) / 512.0;
b1 = drscreen.cv.gaussianBlur(g, 1.0 * sc);
b4 = drscreen.cv.gaussianBlur(g, 4.0 * sc);
hi = g - b1;
mid = b1 - b4;
eHi = mean(abs(double(hi(m))));
eMid = mean(abs(double(mid(m))));
ratio = eHi / max(eMid, 1e-3);
ratioScore = 1 / (1 + exp(-(ratio - 0.45) / 0.12));

sel = double(g(m));
q = drscreen.cv.percentile(sel, [5 95]);
density = eHi / max(q(2) - q(1), 1e-3);
densityScore = 1 / (1 + exp(-(density - 0.014) / 0.005));
s = min(ratioScore, densityScore);
end


function s = illuminationScore(img, mask)
% Uniformity of the illumination field: block-wise median luminance inside the
% aperture; 1 minus a robust coefficient of variation.
gray = drscreen.cv.rgb2gray(img);
[h, w] = size(gray);
blocks = 8;
bh = floor(h / blocks); bw = floor(w / blocks);
meds = [];
for i = 0:blocks-1
    for j = 0:blocks-1
        rr = i*bh + (1:bh); cc = j*bw + (1:bw);
        subM = mask(rr, cc);
        if mean(subM(:) > 0) < 0.6
            continue
        end
        sub = gray(rr, cc);
        meds(end + 1) = median(double(sub(subM > 0))); %#ok<AGROW>
    end
end
if numel(meds) < 4
    s = 0;
    return
end
center = median(meds);
if center < 1
    s = 0;
    return
end
q = drscreen.cv.percentile(meds, [90 10]);
cvr = (q(1) - q(2)) / max(center, 1);
s = min(max(1 - cvr / 0.9, 0), 1);
end


function s = contrastScore(img, mask)
% Vessel visibility: a multi-scale dark-structure response relative to its
% own spread. A retina you cannot grade is usually one whose vasculature has
% vanished (opacity, cataract, miosis).
m = mask > 0;
if nnz(m) < 100
    s = 0;
    return
end
g = single(img(:, :, 2));
sc = max(size(img, 1), size(img, 2)) / 512.0;
resp = zeros(size(g), 'single');
for sigma = [1.0 2.0 3.5]
    k = bitor(fix(2 * sigma * 3 * sc), 1);
    bg = drscreen.cv.morph(g, 'close', k);
    resp = max(resp, bg - g);
end
r = double(resp(m));
signal = drscreen.cv.percentile(r, 99);
noise = std(r, 1);
snr = signal / max(noise, 1e-3);
s = min(max((snr - 1.5) / 4.0, 0), 1);
end


function s = fovScore(fov)
% Fraction of the fitted aperture circle that landed on the sensor. Edge
% contact only counts when coverage is already poor: a correctly captured
% fundus touches the top and bottom of the sensor, and pre-cropped corpora
% touch all four edges.
s = fov.coverage;
if s < 0.80
    s = s * (1 - 0.05 * sum(fov.clipped_sides));
end
if fov.fill_ratio < 0.15
    s = s * 0.4;
end
s = min(max(s, 0), 1);
end


function [under, over] = exposureScores(img, mask)
gray = drscreen.cv.rgb2gray(img);
sel = double(retinaPixels(gray, mask));
if numel(sel) < 100
    under = 0; over = 0;
    return
end
med = median(sel);
fracDark = mean(sel < 25);
fracBlown = mean(sel > 245);
under = min(max((med - 25) / 55, 0), 1) * min(max(1 - fracDark / 0.45, 0), 1);
over = min(max((215 - med) / 55, 0), 1) * min(max(1 - fracBlown / 0.12, 0), 1);
end


function s = artifactScore(img, mask)
% Large specular reflections, dust arcs, lens flare: saturated blobs too large
% to be exudate clusters.
gray = drscreen.cv.rgb2gray(img);
m = mask > 0;
area = max(nnz(m), 1);
blown = (gray > 250) & m;
if ~any(blown(:))
    s = 1.0;
    return
end
blown = drscreen.cv.morph(blown, 'open', 5);
S = drscreen.cv.conncomp(blown);
bad = 0;
for i = 1:numel(S)
    if S(i).Area > 0.004 * area
        bad = bad + S(i).Area;
    end
end
s = min(max(1 - (bad / area) / 0.05, 0), 1);
end


function s = noiseScore(img, mask)
% Sensor noise in flat retinal regions (Immerkaer estimator).
g = single(img(:, :, 2));
K = [1 -2 1; -2 4 -2; 1 -2 1];
lap = drscreen.cv.filter2d(g, K);
m = mask > 0;
if nnz(m) < 100
    s = 0;
    return
end
sigma = sqrt(pi / 2) / (6 * max(nnz(m), 1)) * sum(abs(lap(m)));
s = min(max(1 - (sigma - 1.5) / 5.0, 0), 1);
end


function s = maculaVisibility(img, mask, lm)
if sum(double(mask(:))) < 100
    s = 0;
    return
end
s = drscreen.preprocess.maculaScore(lm, size(img));
end
