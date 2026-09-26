function html = renderHtml(res, art, provenance, title)
%RENDERHTML Standalone HTML screening report (the same layout as the Python one).
%
%   provenance is a one-line statement of where the pixels came from. A
%   reader's first question about any screening output is "what was this run
%   on?", and a report that does not answer it lets a demonstration be
%   mistaken for a clinical record.

if nargin < 3, provenance = ''; end
if nargin < 4 || isempty(title), title = 'DR Screening Report'; end
C = drscreen.constants();
E = @drscreen.report.htmlEscape;

switch res.urgency
    case 'urgent', colour = '#7f1d1d'; bg = '#fecaca'; utext = 'URGENT REFERRAL';
    case 'soon',   colour = '#78350f'; bg = '#fed7aa'; utext = 'REFER';
    otherwise,     colour = '#14532d'; bg = '#bbf7d0'; utext = 'ROUTINE';
end
decisionLabel = struct( ...
    'auto_report', 'Auto-reported (model confident, no referral)', ...
    'refer', 'Referral recommended', ...
    'defer_to_human', 'Deferred for human grading (model uncertain)', ...
    'recapture', 'Image ungradeable - recapture required');
if isfield(decisionLabel, res.decision)
    dlabel = decisionLabel.(res.decision);
else
    dlabel = res.decision;
end

panelImg = '';
try
    panelImg = ['data:image/png;base64,' drscreen.io.base64Image( ...
        drscreen.report.buildReviewPanel(res, art), 'png')];
catch
end

bars = '';
for g = 1:numel(res.class_probabilities)
    p = res.class_probabilities(g);
    cls = '';
    if g - 1 == res.grade, cls = ' pred'; end
    bars = [bars sprintf(['<div class="bar-row"><span class="bar-label">%d &middot; %s</span>' ...
        '<span class="bar-track"><span class="bar-fill%s" style="width:%.1f%%"></span></span>' ...
        '<span class="bar-val">%.1f%%</span></div>'], g - 1, E(C.ICDR_GRADES{g}), cls, ...
        max(0.4, p * 100), p * 100)]; %#ok<AGROW>
end

ev = '';
for i = 1:numel(res.evidence)
    e = res.evidence{i};
    if isfield(e, 'status') && strcmp(e.status, 'not assessed')
        ev = [ev sprintf('<li class=''caution''><b>%s</b>: NOT ASSESSED &mdash; %s</li>', ...
            E(e.finding), E(e.detail))]; %#ok<AGROW>
    elseif isfield(e, 'finding')
        qs = fieldnames(e.per_quadrant);
        parts = cellfun(@(k) sprintf('%s %d', k, e.per_quadrant.(k)), qs, 'UniformOutput', false);
        q = strjoin(parts, ', ');
        extra = '';
        if ~isempty(q), extra = [' (' E(q) ')']; end
        ev = [ev sprintf('<li><b>%s</b>: %d detected%s &mdash; %s%% of retinal area</li>', ...
            E(e.finding), e.count, extra, num2str(e.area_percent))]; %#ok<AGROW>
    elseif isfield(e, 'criterion')
        ev = [ev sprintf('<li class=''criterion''>%s</li>', E(e.criterion))]; %#ok<AGROW>
    elseif isfield(e, 'macular_assessment')
        ev = [ev sprintf('<li class=''macula''>%s</li>', E(e.macular_assessment))]; %#ok<AGROW>
    elseif isfield(e, 'caution')
        ev = [ev sprintf('<li class=''caution''>%s</li>', E(e.caution))]; %#ok<AGROW>
    end
end
if isempty(ev), ev = '<li>No lesions detected.</li>'; end

qrows = '';
if isfield(res.quality, 'scores')
    ks = fieldnames(res.quality.scores);
    for i = 1:numel(ks)
        v = res.quality.verdicts.(ks{i});
        qrows = [qrows sprintf('<tr><td>%s</td><td class=''num''>%.2f</td><td class=''v-%s''>%s</td></tr>', ...
            E(strrep(ks{i}, '_', ' ')), res.quality.scores.(ks{i}), v, v)]; %#ok<AGROW>
    end
end

warns = {};
if isfield(res.quality, 'overall') && strcmp(res.quality.overall, 'borderline')
    warns{end + 1} = ['Image quality is borderline; enhancement was applied. ' ...
                      'Interpret subtle findings with caution.'];
end
if strcmp(res.agreement, 'disagree')
    warns{end + 1} = sprintf('Deep model and rule-based criteria disagree (model %d, rules %d).', ...
        res.grade, res.rule_based_grade);
end
if isfield(res.uncertainty, 'epistemic_variance') && res.uncertainty.epistemic_variance > 0.05
    warns{end + 1} = 'High model uncertainty on this image.';
end
if ~res.gradeable
    warns{end + 1} = 'Image was rejected by the quality gate.';
end
warnHtml = strjoin(cellfun(@(w) ['<div class=''warn''>' E(w) '</div>'], warns, ...
    'UniformOutput', false), '');

provHtml = '';
if ~isempty(provenance)
    provHtml = ['<div class=''prov''><span class=''k''>Image of eye</span><span class=''v''>' ...
                E(provenance) '</span></div>'];
end

adviceHtml = '';
if ~isempty(res.recapture_advice)
    items = strjoin(cellfun(@(a) ['<li>' E(a) '</li>'], res.recapture_advice, ...
        'UniformOutput', false), '');
    adviceHtml = ['<div class="card"><h2>Recapture instructions</h2><ol>' items '</ol></div>'];
end

trows = '';
tk = fieldnames(res.timing_ms);
for i = 1:numel(tk)
    trows = [trows sprintf('<tr><td>%s</td><td class=''num''>%.1f ms</td></tr>', ...
        E(tk{i}), res.timing_ms.(tk{i}))]; %#ok<AGROW>
end
total = 0;
if isfield(res.timing_ms, 'total'), total = res.timing_ms.total; end

if isempty(panelImg)
    panelHtml = '<p class="meta">No image panel available.</p>';
else
    panelHtml = ['<img class="panel" src="' panelImg '" alt="review panel">'];
end

ent = 0; epi = 0;
if isfield(res.uncertainty, 'entropy'), ent = res.uncertainty.entropy; end
if isfield(res.uncertainty, 'epistemic_variance'), epi = res.uncertainty.epistemic_variance; end
overall = '';
if isfield(res.quality, 'overall'), overall = res.quality.overall; end
applied = 'none';
if ~isempty(res.enhancement_applied), applied = strjoin(res.enhancement_applied, ', '); end
stamp = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd HH:mm ''UTC'''));

css = fileread(fullfile(fileparts(mfilename('fullpath')), 'report.css'));

html = strjoin({
    '<!doctype html>'
    '<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    sprintf('<title>%s &mdash; %s</title>', E(title), E(res.image_id))
    ['<style>' strrep(strrep(css, '{{GRADE_BG}}', bg), '{{GRADE_FG}}', colour) '</style></head><body><div class="wrap">']
    '<header><div><h1>Diabetic Retinopathy Screening Report</h1>'
    sprintf(['<div class="meta">Case <b>%s</b> &middot; %s &middot; model %s &middot; ' ...
             '%.0f ms total &middot; MATLAB</div></div>'], E(res.image_id), stamp, ...
             E(res.model_version), total)
    sprintf(['<div class="verdict"><div class="u">%s</div><div class="g">Grade %d &mdash; %s</div>' ...
             '<div class="d">%s<br>P(referable) = <b>%.3f</b> &middot; confidence %.1f%%</div></div>'], ...
             utext, res.grade, E(res.grade_label), E(dlabel), res.referable_probability, ...
             res.confidence * 100)
    '</header>'
    provHtml
    warnHtml
    '<div class="grid"><div>'
    ['<div class="card"><h2>Image review</h2>' panelHtml '</div>']
    ['<div class="card"><h2>Clinical evidence &mdash; ICDR criteria</h2><ul>' ev '</ul></div>']
    adviceHtml
    '</div><div>'
    ['<div class="card"><h2>Severity distribution</h2>' bars '</div>']
    '<div class="card"><h2>Cross-check</h2>'
    sprintf('<div class="kv"><span>Deep model grade</span><b>%d</b></div>', res.grade)
    sprintf('<div class="kv"><span>Rule-based grade</span><b>%d</b></div>', res.rule_based_grade)
    sprintf('<div class="kv"><span>Agreement</span><b>%s</b></div>', E(res.agreement))
    sprintf('<div class="kv"><span>Macular oedema risk</span><b>%d</b></div>', res.dme_risk)
    sprintf('<div class="kv"><span>Predictive entropy</span><b>%.3f</b></div>', ent)
    sprintf('<div class="kv"><span>Epistemic variance</span><b>%.4f</b></div></div>', epi)
    ['<div class="card"><h2>Image quality</h2><table>' qrows '</table>']
    sprintf('<div class="kv" style="margin-top:8px"><span>Overall</span><b>%s</b></div>', E(overall))
    sprintf('<div class="kv"><span>Enhancement applied</span><b>%s</b></div></div>', E(applied))
    ['<div class="card"><h2>Latency breakdown</h2><table>' trows '</table></div>']
    '</div></div>'
    ['<footer>Decision-support output. Not a diagnosis. Every referable and every ' ...
     'sight-threatening finding is reviewed by a qualified ophthalmologist before any ' ...
     'clinical action. Grading follows the International Clinical Diabetic Retinopathy ' ...
     'severity scale; referable DR is grade &ge; 2.</footer>']
    '</div></body></html>'}, newline);
end
