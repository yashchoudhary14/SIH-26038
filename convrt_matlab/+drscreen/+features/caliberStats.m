function [cvAll, nBeaded] = caliberStats(vesselMask, lm, minSegmentPx, minCaliberPx, beadingRatio)
%CALIBERSTATS Venous-beading proxy, measured within individual vessel segments.
%
%   Beading is a focal dilatation along one venule. Pooling calibre over a
%   quadrant is wrong -- its spread is dominated by arcade-versus-capillary
%   differences, which are normal anatomy -- so each segment is compared only
%   with its own median, and only segments thick enough to be venules count.
%   A screening cue; the report never grades on it alone.

if nargin < 3, minSegmentPx = 40; end
if nargin < 4, minCaliberPx = 1.8; end
if nargin < 5, beadingRatio = 1.9; end
cvAll = 0;
nBeaded = 0;

binary = vesselMask > 0;
if nnz(binary) < 50
    return
end
dist = drscreen.cv.distanceTransform(binary);
dil = drscreen.cv.morph(dist, 'dilate', 3);
skel = (dist >= dil - 1e-3) & (dist > 1.0);
if nnz(skel) < 30
    return
end
d = double(dist(skel));
cvAll = std(d, 1) / max(mean(d), 1e-6);

S = drscreen.cv.conncomp(skel);
[h, ~] = size(skel);
beaded = {};
for i = 1:numel(S)
    if S(i).Area < minSegmentPx
        continue
    end
    idx = S(i).PixelIdxList;
    c = double(dist(idx));
    med = median(c);
    if med < minCaliberPx
        continue
    end
    focal = drscreen.cv.percentile(c, 95) / max(med, 1e-6);
    if focal < beadingRatio
        continue
    end
    % Attribute the segment to the quadrant holding most of its length.
    % Pixels are walked in row-major order and ties keep the first quadrant
    % seen, exactly as the Python dict/max does.
    ys = mod(idx - 1, h);
    xs = floor((idx - 1) / h);
    [~, order] = sortrows([ys xs]);
    ys = ys(order); xs = xs(order);
    step = max(1, floor(numel(xs) / 60));
    names = {};
    counts = [];
    for j = 1:step:numel(xs)
        q = drscreen.preprocess.quadrantOf(xs(j), ys(j), lm);
        k = find(strcmp(names, q), 1);
        if isempty(k)
            names{end + 1} = q; %#ok<AGROW>
            counts(end + 1) = 1; %#ok<AGROW>
        else
            counts(k) = counts(k) + 1;
        end
    end
    if ~isempty(counts)
        [~, k] = max(counts);
        if ~ismember(names{k}, beaded)
            beaded{end + 1} = names{k}; %#ok<AGROW>
        end
    end
end
nBeaded = numel(beaded);
end
