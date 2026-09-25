function panel = buildReviewPanel(res, art, panelWidth)
%BUILDREVIEWPANEL Three-up review image: enhanced | lesions | attention.
%
%   Target: an ophthalmologist reaches an agree/disagree decision in under 30
%   seconds. The enhanced fundus orients them, the lesion tile shows *where*,
%   the attention tile is a cross-check that the network looked at the lesions
%   and not at the vignette. Rendered through an off-screen figure so the
%   titles and legend are real text; falls back to a text-free composite if
%   graphics are unavailable (e.g. a headless server without a renderer).

if nargin < 3 || isempty(panelWidth), panelWidth = 1440; end
C = drscreen.constants();
if isfield(art, 'enhanced')
    base = art.enhanced;
elseif isfield(art, 'standardized')
    base = art.standardized;
else
    error('drscreen:report:noImage', 'No image available for the review panel.');
end

tiles = {base};
titles = {'Enhanced fundus'};
if isfield(art, 'lesion_probs')
    lm = [];
    if isfield(art, 'landmarks'), lm = art.landmarks; end
    tiles{end + 1} = drscreen.report.annotateLesions(base, art.lesion_probs, lm);
    titles{end + 1} = 'Detected lesions (ICDR evidence)';
end
if isfield(art, 'cam')
    tiles{end + 1} = drscreen.report.overlayCam(base, art.cam);
    titles{end + 1} = 'Model attention (Grad-CAM++)';
end

n = numel(tiles);
tileW = floor(panelWidth / n);
header = 30;
legendH = 34;
H = tileW + header + legendH;
W = tileW * n;

try
    panel = renderWithText(tiles, titles, tileW, header, legendH, C);
catch
    panel = uint8(18 * ones(H, W, 3));
    for i = 1:n
        t = drscreen.cv.resize(tiles{i}, [tileW tileW], 'area');
        panel(header + (1:tileW), (i - 1) * tileW + (1:tileW), :) = t;
    end
    x = 10;
    for i = 1:numel(C.LESION_CLASSES)
        col = C.LESION_COLORS.(C.LESION_CLASSES{i});
        yy = H - legendH + (11:23);
        xx = x + (0:13);
        for c = 1:3, panel(yy, xx, c) = col(c); end
        x = x + 26 + round(numel(C.LESION_CLASSES{i}) * 7.2);
    end
end
end


function img = renderWithText(tiles, titles, tileW, header, legendH, C)
n = numel(tiles);
W = tileW * n;
H = tileW + header + legendH;
bg = [18 18 18] / 255;
fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100 100 W H], ...
             'Color', bg, 'MenuBar', 'none', 'ToolBar', 'none', ...
             'InvertHardcopy', 'off');
cleanup = onCleanup(@() close(fig));
for i = 1:n
    ax = axes(fig, 'Units', 'pixels', 'Position', [(i - 1) * tileW + 1, legendH + 1, tileW, tileW]);
    image(ax, tiles{i});
    axis(ax, 'image', 'off');
    annotation(fig, 'textbox', 'Units', 'pixels', ...
        'Position', [(i - 1) * tileW + 8, H - header + 2, tileW - 16, header - 4], ...
        'String', titles{i}, 'Color', [0.94 0.94 0.94], 'EdgeColor', 'none', ...
        'FontSize', 10, 'VerticalAlignment', 'middle', 'Interpreter', 'none');
end
x = 10;
for i = 1:numel(C.LESION_CLASSES)
    name = strrep(C.LESION_CLASSES{i}, '_', ' ');
    col = C.LESION_COLORS.(C.LESION_CLASSES{i}) / 255;
    annotation(fig, 'rectangle', 'Units', 'pixels', 'Position', [x, 11, 14, 12], ...
        'FaceColor', col, 'Color', col);
    annotation(fig, 'textbox', 'Units', 'pixels', 'Position', [x + 18, 4, 170, 26], ...
        'String', name, 'Color', [0.92 0.92 0.92], 'EdgeColor', 'none', ...
        'FontSize', 8, 'VerticalAlignment', 'middle', 'Interpreter', 'none');
    x = x + 26 + round(numel(name) * 7.2);
end
drawnow;
img = print(fig, '-RGBImage', '-r96');
if size(img, 1) ~= H || size(img, 2) ~= W
    img = drscreen.cv.resize(img, [H W], 'area');
end
end
