function results = run_demo(varargin)
%RUN_DEMO Screen images from the command line and write clinical reports.
%
%   run_demo                                % the 12 committed verification photographs
%   run_demo('Image', 'fundus.jpg')
%   run_demo('Folder', 'data/showcase/grade_3')
%   run_demo('Bundle', 'pooled', 'Out', 'outputs/reports', 'NoReport', true)
%
%   Prints one line per image and, where a reference grade is known, the exact
%   grade match. Mirrors scripts/run_demo.py in the Python repository.

ip = inputParser;
ip.addParameter('Image', '');
ip.addParameter('Folder', '');
ip.addParameter('Bundle', '');
ip.addParameter('Out', '');
ip.addParameter('NoReport', false);
ip.addParameter('McSamples', 8);
ip.parse(varargin{:});
o = ip.Results;

root = fileparts(mfilename('fullpath'));
addpath(root);
if isempty(o.Bundle), o.Bundle = drscreen.constants().DEFAULT_BUNDLE; end
P = drscreen.paths();
if isempty(o.Out), o.Out = P.reports; end
pipe = drscreen.Pipeline.load(o.Bundle, 'McSamples', o.McSamples);
fprintf('Loaded bundle %s: threshold %.4f, T %.3f, segmentation %d, grader %d\n\n', ...
    o.Bundle, pipe.Cfg.referral_threshold, pipe.Cfg.temperature, ...
    ~isempty(pipe.Segmenter), ~isempty(pipe.Grader));

jobs = struct('name', {}, 'file', {}, 'truth', {});
if ~isempty(o.Image)
    [~, n] = fileparts(o.Image);
    jobs(end + 1) = struct('name', n, 'file', o.Image, 'truth', []);
end
if ~isempty(o.Folder)
    files = [dir(fullfile(o.Folder, '*.jpg')); dir(fullfile(o.Folder, '*.png')); ...
             dir(fullfile(o.Folder, '*.jpeg')); dir(fullfile(o.Folder, '*.tif'))];
    for i = 1:numel(files)
        [~, n] = fileparts(files(i).name);
        tok = regexp(n, '^grade(\d)', 'tokens', 'once');
        t = [];
        if ~isempty(tok), t = str2double(tok{1}); end
        jobs(end + 1) = struct('name', n, 'file', fullfile(files(i).folder, files(i).name), 'truth', t); %#ok<AGROW>
    end
end
if isempty(jobs)
    list = drscreen.samples.list();
    for i = 1:numel(list)
        [~, n] = fileparts(list{i});
        tok = regexp(n, '^grade(\d)', 'tokens', 'once');
        jobs(end + 1) = struct('name', n, 'file', list{i}, 'truth', str2double(tok{1})); %#ok<AGROW>
    end
end

C = drscreen.constants();
results = cell(1, numel(jobs));
correct = 0; graded = 0;
t0 = tic;
for i = 1:numel(jobs)
    [res, art] = pipe.run(jobs(i).file, jobs(i).name);
    results{i} = res;
    if ~res.gradeable
        fprintf('%-28s UNGRADEABLE (%s)\n', res.image_id, res.quality.overall);
        for k = 1:numel(res.recapture_advice)
            fprintf('%28s   -> %s\n', '', res.recapture_advice{k});
        end
    else
        fprintf('%-28s grade %d (%-16s) P(ref)=%.3f conf=%.2f %-15s %-7s rule=%d %6.0fms\n', ...
            res.image_id, res.grade, C.ICDR_GRADES{res.grade + 1}, res.referable_probability, ...
            res.confidence, res.decision, res.urgency, res.rule_based_grade, res.timing_ms.total);
        if ~isempty(jobs(i).truth)
            graded = graded + 1;
            correct = correct + (res.grade == jobs(i).truth);
            if res.grade ~= jobs(i).truth
                fprintf('%28s   (reference grade %d)\n', '', jobs(i).truth);
            end
        end
    end
    if ~o.NoReport
        p = drscreen.report.saveReport(res, art, o.Out);
        fprintf('%28s   report: %s\n', '', p.html);
    end
end
dt = toc(t0);
fprintf('\n%d images in %.1f s (%.0f ms/image)\n', numel(jobs), dt, dt / numel(jobs) * 1000);
if graded
    fprintf('exact grade match on images with a reference grade: %d/%d\n', correct, graded);
end
if ~isfolder(o.Out), mkdir(o.Out); end
drscreen.io.writeText(fullfile(o.Out, 'batch_results.json'), ...
    jsonencode(results, 'PrettyPrint', true));
if nargout == 0, clear results; end
end
