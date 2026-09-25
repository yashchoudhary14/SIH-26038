function paths = saveReport(res, art, outDir, provenance)
%SAVEREPORT Write <id>_report.html, <id>_panel.png and <id>_result.json.
%
%   The PNG is what a low-bandwidth review client receives; the HTML is the
%   record; the JSON is the machine-readable result, schema-compatible with
%   the Python API.

if nargin < 3 || isempty(outDir), outDir = drscreen.paths().reports; end
if nargin < 4, provenance = ''; end
if ~isfolder(outDir), mkdir(outDir); end
paths = struct();

paths.html = fullfile(outDir, [res.image_id '_report.html']);
drscreen.io.writeText(paths.html, drscreen.report.renderHtml(res, art, provenance));

try
    p = fullfile(outDir, [res.image_id '_panel.png']);
    imwrite(drscreen.report.buildReviewPanel(res, art), p);
    paths.panel = p;
catch
end

paths.json = fullfile(outDir, [res.image_id '_result.json']);
drscreen.io.writeText(paths.json, jsonencode(res, 'PrettyPrint', true));
end
