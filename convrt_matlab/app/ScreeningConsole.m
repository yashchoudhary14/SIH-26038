classdef ScreeningConsole < handle
    %SCREENINGCONSOLE Desktop screening console for the MATLAB pipeline.
    %
    %   app = ScreeningConsole();            % or run launch_console
    %
    %   A reviewer workstation, built so an ophthalmologist can reach an
    %   agree/disagree decision in under 30 seconds:
    %
    %   * left    pick a case -- the committed showcase and verification
    %             photographs, any image file, or a whole folder (batch)
    %   * centre  enhanced fundus | lesion outlines | Grad-CAM++ attention
    %   * right   the verdict first (grade, referral, urgency, P(referable)),
    %             then the severity distribution, the ICDR evidence in clinical
    %             language, the quality criteria and the model/rule cross-check
    %   * bottom  record the human decision to the audit log, save the HTML
    %             report, or open the web console served from this session
    %
    %   Compiles to a standalone desktop app with deploy/build_standalone.m.

    properties (SetAccess = private)
        Pipe
        Server
        Res
        Art
        CurrentFile = ''
        TrueGrade = []
        BatchRows = {}
        ReviewStart
    end

    properties (Access = private)
        Fig
        BundleDrop
        McSpin
        Status
        CaseTree
        ImgTabs
        AxEnh
        AxLes
        AxCam
        AxRaw
        Verdict
        VerdictSub
        ProbAx
        Evidence
        QualityTable
        CrossCheck
        Timing
        ReviewerName
        ReviewerGrade
        Notes
        BatchTable
        BatchSummary
        BatchTab
        FirstTab
    end

    methods
        function app = ScreeningConsole(bundle)
            if nargin < 1 || isempty(bundle), bundle = drscreen.constants().DEFAULT_BUNDLE; end
            app.build(bundle);
            app.loadPipeline(bundle);
            app.populateCases();
        end

        function delete(app)
            if ~isempty(app.Server), app.Server.stop(); end
            if ~isempty(app.Fig) && isvalid(app.Fig), delete(app.Fig); end
        end

        function waitUntilClosed(app)
            %WAITUNTILCLOSED Block until the console window closes (compiled app).
            if ~isempty(app.Fig) && isvalid(app.Fig)
                waitfor(app.Fig);
            end
        end

        function screenFile(app, file, trueGrade)
            %SCREENFILE Screen one image file and show the result.
            if nargin < 3, trueGrade = []; end
            d = uiprogressdlg(app.Fig, 'Title', 'Screening', ...
                'Message', 'Running the pipeline...', 'Indeterminate', 'on');
            cleaner = onCleanup(@() close(d));
            [~, name] = fileparts(file);
            [app.Res, app.Art] = app.Pipe.run(file, name);
            app.CurrentFile = file;
            app.TrueGrade = trueGrade;
            app.ReviewStart = tic;
            app.show();
        end
    end

    % ==================================================================
    % Layout
    % ==================================================================
    methods (Access = private)
        function build(app, bundle)
            app.Fig = uifigure('Name', 'DR Screening Console (MATLAB)', ...
                'Position', [60 60 1480 900], 'Color', [0.043 0.059 0.078]);
            app.Fig.CloseRequestFcn = @(~, ~) delete(app);
            g = uigridlayout(app.Fig, [3 3], 'RowHeight', {44, '1x', 150}, ...
                'ColumnWidth', {260, '1x', 400}, 'Padding', [10 10 10 10], ...
                'RowSpacing', 8, 'ColumnSpacing', 10, 'BackgroundColor', app.Fig.Color);

            % ---- top bar ----
            top = uigridlayout(g, [1 7], 'ColumnWidth', {'1x', 60, 120, 90, 60, 130, 150}, ...
                'Padding', [0 0 0 0], 'BackgroundColor', app.Fig.Color);
            top.Layout.Row = 1; top.Layout.Column = [1 3];
            uilabel(top, 'Text', 'Diabetic Retinopathy Screening Console', ...
                'FontSize', 17, 'FontWeight', 'bold', 'FontColor', [0.9 0.93 0.95]);
            uilabel(top, 'Text', 'Model', 'FontColor', [0.6 0.65 0.7], 'HorizontalAlignment', 'right');
            app.BundleDrop = uidropdown(top, 'Items', unique([{'prepool', 'pooled'}, {bundle}], 'stable'), ...
                'Value', bundle, ...
                'ValueChangedFcn', @(s, ~) app.loadPipeline(s.Value));
            uilabel(top, 'Text', 'MC samples', 'FontColor', [0.6 0.65 0.7], 'HorizontalAlignment', 'right');
            app.McSpin = uispinner(top, 'Limits', [0 32], 'Value', 8, 'Step', 1, ...
                'ValueChangedFcn', @(s, ~) app.setMc(s.Value));
            uibutton(top, 'Text', 'Open web console', 'ButtonPushedFcn', @(~, ~) app.openWeb());
            app.Status = uilabel(top, 'Text', 'Loading models...', 'FontColor', [0.6 0.65 0.7]);

            % ---- left: case picker ----
            left = uigridlayout(g, [4 1], 'RowHeight', {'1x', 30, 30, 30}, 'Padding', [0 0 0 0], ...
                'BackgroundColor', app.Fig.Color);
            left.Layout.Row = [2 3]; left.Layout.Column = 1;
            app.CaseTree = uitree(left, 'SelectionChangedFcn', @(~, e) app.onCase(e));
            uibutton(left, 'Text', 'Open image...', 'ButtonPushedFcn', @(~, ~) app.openImage());
            uibutton(left, 'Text', 'Batch screen a folder...', 'ButtonPushedFcn', @(~, ~) app.batchFolder());
            uibutton(left, 'Text', 'Batch screen the showcase set', ...
                'ButtonPushedFcn', @(~, ~) app.batchFolder(drscreen.paths().showcase));

            % ---- centre: images + batch table ----
            app.ImgTabs = uitabgroup(g);
            app.ImgTabs.Layout.Row = 2; app.ImgTabs.Layout.Column = 2;
            app.AxEnh = app.imageTab('Enhanced fundus');
            app.FirstTab = app.AxEnh.Parent.Parent;
            app.AxLes = app.imageTab('Detected lesions');
            app.AxCam = app.imageTab('Attention (Grad-CAM++)');
            app.AxRaw = app.imageTab('Original');
            bt = uitab(app.ImgTabs, 'Title', 'Batch results');
            app.BatchTab = bt;
            bg = uigridlayout(bt, [2 1], 'RowHeight', {'1x', 60});
            app.BatchTable = uitable(bg, 'ColumnName', {'image', 'true', 'grade', 'decision', ...
                'urgency', 'P(ref)', 'quality', 'ms'}, 'RowName', {}, ...
                'CellSelectionCallback', @(~, e) app.onBatchRow(e));
            app.BatchSummary = uitextarea(bg, 'Editable', 'off', 'FontName', 'Consolas');

            % ---- right: verdict and evidence ----
            right = uigridlayout(g, [7 1], 'RowHeight', {108, 150, 110, '1x', 170, 30, 30}, ...
                'Padding', [0 0 0 0], 'RowSpacing', 6, 'BackgroundColor', app.Fig.Color);
            right.Layout.Row = [2 3]; right.Layout.Column = 3;
            vp = uipanel(right, 'BackgroundColor', [0.73 0.97 0.82], 'BorderType', 'none');
            vg = uigridlayout(vp, [2 1], 'RowHeight', {40, '1x'}, 'Padding', [12 6 12 6], ...
                'BackgroundColor', vp.BackgroundColor);
            app.Verdict = uilabel(vg, 'Text', 'No case screened', 'FontSize', 21, ...
                'FontWeight', 'bold', 'FontColor', [0.08 0.33 0.18]);
            app.VerdictSub = uilabel(vg, 'Text', '', 'FontSize', 12, 'WordWrap', 'on', ...
                'FontColor', [0.08 0.33 0.18]);
            app.ProbAx = uiaxes(right);
            app.CrossCheck = uitextarea(right, 'Editable', 'off', 'FontName', 'Consolas', ...
                'Value', {''});
            app.Evidence = uitextarea(right, 'Editable', 'off', 'Value', {'Evidence appears here.'});
            app.QualityTable = uitable(right, 'ColumnName', {'criterion', 'score', 'verdict'}, ...
                'RowName', {}, 'ColumnWidth', {'1x', 70, 110});
            uibutton(right, 'Text', 'Save HTML report and open', ...
                'ButtonPushedFcn', @(~, ~) app.saveReport());
            app.Timing = uilabel(right, 'Text', '', 'FontColor', [0.6 0.65 0.7]);

            % ---- bottom centre: human review ----
            rv = uipanel(g, 'Title', 'Reviewer decision (audit log)', ...
                'BackgroundColor', [0.075 0.1 0.133], 'ForegroundColor', [0.8 0.84 0.88]);
            rv.Layout.Row = 3; rv.Layout.Column = 2;
            rg = uigridlayout(rv, [2 5], 'RowHeight', {30, '1x'}, ...
                'ColumnWidth', {150, 190, 120, 120, '1x'}, 'BackgroundColor', rv.BackgroundColor);
            app.ReviewerName = uieditfield(rg, 'text', 'Placeholder', 'Reviewer name');
            app.ReviewerGrade = uidropdown(rg, 'Items', {'0 No apparent DR', '1 Mild NPDR', ...
                '2 Moderate NPDR', '3 Severe NPDR', '4 Proliferative DR'});
            uibutton(rg, 'Text', 'Agree with model', 'ButtonPushedFcn', @(~, ~) app.record(true));
            uibutton(rg, 'Text', 'Record my grade', 'ButtonPushedFcn', @(~, ~) app.record(false));
            uilabel(rg, 'Text', '');
            app.Notes = uitextarea(rg, 'Placeholder', 'Notes (optional)');
            app.Notes.Layout.Column = [1 5];
        end

        function ax = imageTab(app, title)
            t = uitab(app.ImgTabs, 'Title', title, 'BackgroundColor', [0.07 0.07 0.07]);
            tg = uigridlayout(t, [1 1], 'Padding', [0 0 0 0], 'BackgroundColor', [0.07 0.07 0.07]);
            ax = uiaxes(tg, 'Color', [0.07 0.07 0.07]);
            axis(ax, 'off');
            disableDefaultInteractivity(ax);
        end

        % ==============================================================
        % Actions
        % ==============================================================
        function loadPipeline(app, bundle)
            d = uiprogressdlg(app.Fig, 'Title', 'Loading', ...
                'Message', sprintf('Loading the %s model bundle...', bundle), 'Indeterminate', 'on');
            cleaner = onCleanup(@() close(d));
            mc = 8;
            if ~isempty(app.McSpin), mc = app.McSpin.Value; end
            app.Pipe = drscreen.Pipeline.load(bundle, 'McSamples', mc);
            if ~isempty(app.Server)
                app.Server.stop();
                app.Server = [];
            end
            cfg = app.Pipe.Cfg;
            app.Status.Text = sprintf('%s | thr %.3f | T %.2f', bundle, ...
                cfg.referral_threshold, cfg.temperature);
        end

        function setMc(app, v)
            app.Pipe.Cfg.mc_samples = v;
        end

        function populateCases(app)
            P = drscreen.paths();
            delete(app.CaseTree.Children);
            sc = uitreenode(app.CaseTree, 'Text', 'Showcase set (60)');
            man = fullfile(P.showcase, 'manifest.json');
            if isfile(man)
                m = jsondecode(drscreen.io.readText(man));
                rows = m.cases;
                for g = 0:4
                    gn = uitreenode(sc, 'Text', sprintf('Grade %d', g));
                    for i = 1:numel(rows)
                        if iscell(rows), r = rows{i}; else, r = rows(i); end
                        if r.true_grade ~= g, continue; end
                        [~, stem] = fileparts(r.file);
                        uitreenode(gn, 'Text', sprintf('%s  (%s)', stem, r.source), ...
                            'NodeData', struct('file', fullfile(P.showcase, strrep(r.file, '/', filesep)), ...
                                               'grade', r.true_grade));
                    end
                end
            end
            vs = uitreenode(app.CaseTree, 'Text', 'Verification set (12)');
            f = fullfile(P.vset, 'verification_summary.json');
            if isfile(f)
                s = jsondecode(drscreen.io.readText(f));
                rows = s.cases;
                for i = 1:numel(rows)
                    if iscell(rows), r = rows{i}; else, r = rows(i); end
                    uitreenode(vs, 'Text', sprintf('%s  (%s)', r.case, r.source), ...
                        'NodeData', struct('file', fullfile(P.vset, strrep(r.image, '/', filesep)), ...
                                           'grade', r.true_grade));
                end
            end
            expand(sc);
        end

        function onCase(app, e)
            n = e.SelectedNodes;
            if isempty(n) || isempty(n.NodeData), return; end
            app.screenFile(n.NodeData.file, n.NodeData.grade);
        end

        function openImage(app)
            [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp', 'Fundus images'}, ...
                'Choose a fundus photograph');
            figure(app.Fig);
            if isequal(f, 0), return; end
            app.screenFile(fullfile(p, f), []);
        end

        function batchFolder(app, folder)
            if nargin < 2
                folder = uigetdir(pwd, 'Folder of fundus photographs');
                figure(app.Fig);
                if isequal(folder, 0), return; end
            end
            files = [];
            for ext = {'*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp'}
                files = [files; dir(fullfile(folder, '**', ext{1}))]; %#ok<AGROW>
            end
            if isempty(files)
                uialert(app.Fig, 'No images found in that folder.', 'Batch');
                return
            end
            truth = batchTruth(folder);
            d = uiprogressdlg(app.Fig, 'Title', 'Batch screening', 'Cancelable', 'on');
            cleaner = onCleanup(@() close(d));
            rows = cell(numel(files), 8);
            keep = false(numel(files), 1);
            for i = 1:numel(files)
                if d.CancelRequested, break; end
                d.Value = (i - 1) / numel(files);
                d.Message = sprintf('%d / %d  %s', i, numel(files), files(i).name);
                f = fullfile(files(i).folder, files(i).name);
                [~, stem] = fileparts(f);
                t = tic;
                res = app.Pipe.run(f, stem, false);
                tg = NaN;
                if isKey(truth, stem), tg = truth(stem); end
                q = '';
                if isfield(res.quality, 'overall'), q = res.quality.overall; end
                rows(i, :) = {f, tg, res.grade, res.decision, res.urgency, ...
                              round(res.referable_probability, 4), q, round(toc(t) * 1000)};
                keep(i) = true;
            end
            rows = rows(keep, :);
            app.BatchRows = rows;
            shown = rows;
            for i = 1:size(shown, 1)
                [~, shown{i, 1}] = fileparts(shown{i, 1});
            end
            app.BatchTable.Data = shown;
            app.BatchSummary.Value = batchSummary(rows);
            app.ImgTabs.SelectedTab = app.BatchTab;
            out = fullfile(drscreen.paths().reports, 'batch_results.csv');
            if ~isfolder(fileparts(out)), mkdir(fileparts(out)); end
            T = cell2table(rows, 'VariableNames', {'file', 'true_grade', 'grade', 'decision', ...
                'urgency', 'p_referable', 'quality', 'ms'});
            writetable(T, out);
            app.Status.Text = sprintf('Batch: %d images -> %s', size(rows, 1), out);
        end

        function onBatchRow(app, e)
            if isempty(e.Indices), return; end
            r = e.Indices(1, 1);
            tg = app.BatchRows{r, 2};
            if isnan(tg), tg = []; end
            app.screenFile(app.BatchRows{r, 1}, tg);
        end

        function saveReport(app)
            if isempty(app.Res), return; end
            prov = sprintf('Local file %s', app.CurrentFile);
            p = drscreen.report.saveReport(app.Res, app.Art, drscreen.paths().reports, prov);
            web(p.html, '-browser');
            app.Status.Text = ['Saved ' p.html];
        end

        function openWeb(app)
            if isempty(app.Server)
                s = drscreen.server.Server('Pipeline', app.Pipe, 'Port', 8000);
                try
                    s.start();
                catch err
                    uialert(app.Fig, sprintf(['Could not start the web console on port 8000; ' ...
                        'another program is probably using it.\n\n%s'], err.message), 'Web console');
                    return
                end
                app.Server = s;
            end
            web(app.Server.url(), '-browser');
            app.Status.Text = ['Web console at ' app.Server.url()];
        end

        function record(app, agree)
            if isempty(app.Res) || app.Res.grade < 0
                uialert(app.Fig, 'Screen a gradeable image first.', 'Review');
                return
            end
            if agree
                rg = app.Res.grade;
            else
                rg = str2double(app.ReviewerGrade.Value(1));
            end
            mg = app.Res.grade;
            if mg == rg
                ag = 'exact';
            elseif abs(mg - rg) == 1
                ag = 'within_one';
            else
                ag = 'disagree';
            end
            secs = 0;
            if ~isempty(app.ReviewStart), secs = toc(app.ReviewStart); end
            name = strtrim(app.ReviewerName.Value);
            if isempty(name), name = 'unknown'; end
            rec = struct('timestamp', char(datetime('now', 'TimeZone', 'UTC', ...
                    'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSSxxx')), ...
                'image_id', app.Res.image_id, 'model_grade', mg, 'reviewer_grade', rg, ...
                'reviewer', name, 'review_seconds', round(secs, 1), ...
                'notes', strjoin(app.Notes.Value, ' '), 'agreement', ag);
            drscreen.io.writeText(drscreen.paths().audit, [jsonencode(rec) newline], 'a');
            app.Status.Text = sprintf('Recorded: reviewer %d vs model %d (%s), %.0f s', ...
                rg, mg, ag, secs);
        end

        % ==============================================================
        % Rendering a result
        % ==============================================================
        function show(app)
            r = app.Res;
            a = app.Art;
            C = drscreen.constants();
            imshow(a.raw, 'Parent', app.AxRaw);
            if isfield(a, 'enhanced')
                base = a.enhanced;
            else
                base = a.standardized;
            end
            imshow(base, 'Parent', app.AxEnh);
            if isfield(a, 'lesion_probs')
                imshow(drscreen.report.annotateLesions(base, a.lesion_probs, a.landmarks), ...
                    'Parent', app.AxLes);
            else
                cla(app.AxLes);
            end
            if isfield(a, 'cam')
                imshow(drscreen.report.overlayCam(base, a.cam), 'Parent', app.AxCam);
            else
                cla(app.AxCam);
            end

            % verdict card, coloured by urgency
            switch r.urgency
                case 'urgent', bg = [0.996 0.792 0.792]; fg = [0.498 0.114 0.114]; u = 'URGENT REFERRAL';
                case 'soon',   bg = [0.996 0.843 0.667]; fg = [0.471 0.208 0.059]; u = 'REFER';
                otherwise,     bg = [0.733 0.969 0.816]; fg = [0.078 0.325 0.176]; u = 'ROUTINE';
            end
            app.Verdict.Parent.Parent.BackgroundColor = bg;
            app.Verdict.Parent.BackgroundColor = bg;
            app.Verdict.FontColor = fg;
            app.VerdictSub.FontColor = fg;
            if ~r.gradeable
                app.Verdict.Text = 'Ungradeable - recapture';
                app.VerdictSub.Text = strjoin(r.recapture_advice, ' ');
            else
                app.Verdict.Text = sprintf('Grade %d - %s', r.grade, r.grade_label);
                truth = '';
                if ~isempty(app.TrueGrade)
                    truth = sprintf('   |   reference grade %d', app.TrueGrade);
                end
                app.VerdictSub.Text = sprintf('%s  |  %s  |  P(referable) %.3f  |  confidence %.0f%%%s', ...
                    u, strrep(r.decision, '_', ' '), r.referable_probability, r.confidence * 100, truth);
            end

            cla(app.ProbAx);
            if ~isempty(r.class_probabilities)
                p = r.class_probabilities * 100;
                bh = barh(app.ProbAx, 0:4, p, 'FaceColor', 'flat');
                bh.CData = repmat([0.23 0.43 0.65], 5, 1);
                if r.grade >= 0, bh.CData(r.grade + 1, :) = [0.18 0.62 0.42]; end
                app.ProbAx.YTickLabel = C.ICDR_GRADES;
                app.ProbAx.XLim = [0 100];
                app.ProbAx.XLabel.String = '% probability';
                app.ProbAx.Title.String = 'Severity distribution';
            end

            if r.gradeable
                app.CrossCheck.Value = {
                    sprintf('Deep model grade      %d', r.grade)
                    sprintf('Rule-based grade      %d  (%s)', r.rule_based_grade, r.agreement)
                    sprintf('Macular oedema risk   %d', r.dme_risk)
                    sprintf('Entropy / epistemic   %.3f / %.4f', r.uncertainty.entropy, ...
                            r.uncertainty.epistemic_variance)
                    sprintf('Enhancement           %s', strjoin(r.enhancement_applied, ', '))};
            else
                app.CrossCheck.Value = {'Rejected by the quality gate.'};
            end

            lines = {};
            for i = 1:numel(r.evidence)
                e = r.evidence{i};
                if isfield(e, 'status')
                    lines{end + 1} = sprintf('! %s: NOT ASSESSED', e.finding); %#ok<AGROW>
                elseif isfield(e, 'finding')
                    lines{end + 1} = sprintf('- %s: %d detected (%.3f%% of retina)', ...
                        e.finding, e.count, e.area_percent); %#ok<AGROW>
                elseif isfield(e, 'criterion')
                    lines{end + 1} = ['> ' e.criterion]; %#ok<AGROW>
                elseif isfield(e, 'macular_assessment')
                    lines{end + 1} = ['> ' e.macular_assessment]; %#ok<AGROW>
                elseif isfield(e, 'caution')
                    lines{end + 1} = ['! ' e.caution]; %#ok<AGROW>
                end
            end
            if isempty(lines), lines = {'No evidence (image not graded).'}; end
            app.Evidence.Value = lines;

            q = r.quality;
            if isfield(q, 'scores')
                ks = fieldnames(q.scores);
                data = cell(numel(ks), 3);
                for i = 1:numel(ks)
                    data(i, :) = {strrep(ks{i}, '_', ' '), sprintf('%.2f', q.scores.(ks{i})), ...
                                  q.verdicts.(ks{i})};
                end
                app.QualityTable.Data = data;
            end
            tk = fieldnames(r.timing_ms);
            if ~isempty(tk)
                app.Timing.Text = sprintf('%s | %.0f ms total | %s', r.image_id, ...
                    r.timing_ms.total, strjoin(cellfun(@(k) sprintf('%s %.0f', k, r.timing_ms.(k)), ...
                    tk(1:end-1), 'UniformOutput', false), ', '));
            end
            app.ImgTabs.SelectedTab = app.FirstTab;
        end
    end
end


% ======================================================================
function truth = batchTruth(folder)
% Reference grades, when the folder carries them (the showcase manifest), so a
% batch run doubles as a quick check.
truth = containers.Map();
m = fullfile(folder, 'manifest.json');
if ~isfile(m), return; end
s = jsondecode(drscreen.io.readText(m));
rows = s.cases;
for i = 1:numel(rows)
    if iscell(rows), r = rows{i}; else, r = rows(i); end
    [~, stem] = fileparts(r.file);
    truth(stem) = r.true_grade;
end
end


function lines = batchSummary(rows)
n = size(rows, 1);
dec = rows(:, 4);
lines = {sprintf('%d images | refer %d | defer %d | auto-report %d | recapture %d', n, ...
    sum(strcmp(dec, 'refer')), sum(strcmp(dec, 'defer_to_human')), ...
    sum(strcmp(dec, 'auto_report')), sum(strcmp(dec, 'recapture')))};
tg = cell2mat(rows(:, 2));
has = ~isnan(tg);
if any(has)
    g = cell2mat(rows(has, 3));
    t = tg(has);
    refTrue = t >= 2;
    refPred = ~strcmp(dec(has), 'auto_report');
    lines{end + 1} = sprintf(['With reference grades (%d): exact %.1f%% | within one %.1f%% | ' ...
        'referable sensitivity %.1f%% | specificity %.1f%%'], nnz(has), ...
        100 * mean(g == t), 100 * mean(abs(g - t) <= 1), ...
        100 * mean(refPred(refTrue)), 100 * mean(~refPred(~refTrue)));
end
end
