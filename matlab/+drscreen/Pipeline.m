classdef Pipeline < handle
    %PIPELINE End-to-end diabetic retinopathy screening: image in, auditable result out.
    %
    %   pipe = drscreen.Pipeline.load();            % the default bundle (constants.m)
    %   pipe = drscreen.Pipeline.load('pooled');    % trained on all four corpora
    %   [res, art] = pipe.run('fundus.jpg');
    %   disp(jsonencode(res, 'PrettyPrint', true))
    %
    %   Stages, each timed:
    %    1 geometry        FOV detection, tight crop, square pad, resize to 512
    %    2 landmarks       optic disc and fovea -- the clinical coordinate frame
    %    3 quality gate    interpretable criteria; ungradeable images stop here
    %                      with recapture instructions, never a guessed grade
    %    4 enhancement     only what the gate asked for, then the gate is re-run
    %    5 segmentation    five lesion classes at 1024 px
    %    6 features        lesion counts per quadrant, distance from the fovea
    %    7 rule grade      ICDR criteria applied directly (the audit trail)
    %    8 grading         CORN ordinal network, temperature + isotonic calibrated
    %    9 decision        referral against a frozen operating point, with a
    %                      defer-to-human band and a fitted urgency tier
    %   10 explanation     Grad-CAM++ over the referable log-odds
    %
    %   The result struct has the fields of the Python ScreeningResult, in the
    %   same order and with the same rounding, so its JSON is interchangeable
    %   with the Python API's -- the web console reads either.

    properties
        Cfg             % operating point and options (see defaultConfig)
        Grader          % drscreen.models.Grader, or [] for rule-based only
        Segmenter       % drscreen.models.Segmenter, or []
        Bundle = ''     % bundle name or directory the config came from
    end

    methods (Static)
        function obj = load(bundle, varargin)
            %LOAD Build a pipeline from an exported bundle.
            %   drscreen.Pipeline.load(bundle, 'McSamples', 8, 'EnableCam', true)
            %   bundle: 'prepool', 'pooled', or a directory holding
            %           pipeline.json + grader.mat. Default: constants DEFAULT_BUNDLE.
            if nargin < 1 || isempty(bundle), bundle = drscreen.constants().DEFAULT_BUNDLE; end
            P = drscreen.paths();
            if isfolder(bundle)
                bdir = bundle;
            else
                bdir = fullfile(P.models, bundle);
            end
            cfg = drscreen.Pipeline.defaultConfig();
            meta = fullfile(bdir, 'pipeline.json');
            if isfile(meta)
                m = jsondecode(fileread(meta));
                cfg = drscreen.Pipeline.applyMeta(cfg, m);
            end
            ip = inputParser;
            ip.addParameter('McSamples', cfg.mc_samples);
            ip.addParameter('EnableCam', cfg.enable_cam);
            ip.parse(varargin{:});
            cfg.mc_samples = ip.Results.McSamples;
            cfg.enable_cam = ip.Results.EnableCam;

            seg = [];
            if isfile(fullfile(P.models, 'segmentation.mat'))
                seg = drscreen.models.Segmenter(P.models);
                cfg.seg_size = seg.Size;
                cfg.supervised_lesion_classes = seg.SupervisedClasses;
            end
            grader = [];
            if isfile(fullfile(bdir, 'grader.mat'))
                grader = drscreen.models.Grader(bdir);
            end
            obj = drscreen.Pipeline(grader, seg, cfg);
            obj.Bundle = bundle;
        end

        function cfg = defaultConfig()
            cfg = struct( ...
                'size', 512, ...
                'seg_size', 1024, ...
                'lesion_threshold', 0.5, ...
                'lesion_thresholds', [], ...
                'supervised_lesion_classes', {{}}, ...
                'grade_thresholds', [], ...
                'referral_threshold', 0.5, ...
                'urgent_threshold', [], ...
                'defer_band', [0.35 0.65], ...
                'temperature', 1.0, ...
                'mc_samples', 8, ...
                'uncertainty_defer', 0.05, ...
                'enable_cam', true, ...
                'cam_method', 'gradcam++', ...
                'model_version', 'drscreen-1.0.0', ...
                'calibrator', []);
        end

        function cfg = applyMeta(cfg, m)
            keys = {'referral_threshold', 'temperature', 'size', 'seg_size', ...
                    'calibrator', 'lesion_thresholds', 'grade_thresholds', ...
                    'urgent_threshold', 'defer_band', 'model_version'};
            for i = 1:numel(keys)
                if isfield(m, keys{i})
                    cfg.(keys{i}) = m.(keys{i});
                end
            end
            cfg.defer_band = double(cfg.defer_band(:))';
        end
    end

    methods
        function obj = Pipeline(grader, segmenter, cfg)
            if nargin < 3 || isempty(cfg), cfg = drscreen.Pipeline.defaultConfig(); end
            obj.Grader = grader;
            obj.Segmenter = segmenter;
            obj.Cfg = cfg;
        end

        function u = unassessedLesions(obj)
            %UNASSESSEDLESIONS Classes the lesion model cannot see, because it
            %never saw one. Kept distinct from "detected none" everywhere.
            C = drscreen.constants();
            sup = obj.Cfg.supervised_lesion_classes;
            if isempty(sup), sup = C.PIXEL_ANNOTATED_LESION_CLASSES; end
            u = C.LESION_CLASSES(~ismember(C.LESION_CLASSES, sup));
        end

        function [res, art] = run(obj, image, imageId, explain)
            %RUN Screen one image. Returns the JSON-ready result and the arrays
            %a report needs (standardised, enhanced, lesion maps, CAM...).
            C = drscreen.constants();
            cfg = obj.Cfg;
            if nargin < 3 || isempty(imageId)
                if ischar(image) || isstring(image)
                    [~, imageId] = fileparts(char(image));
                else
                    imageId = 'case';
                end
            end
            if nargin < 4, explain = true; end
            tAll = tic;
            timing = struct();
            res = drscreen.Pipeline.emptyResult(char(imageId), cfg.model_version);
            art = struct();

            raw = drscreen.io.readImage(image);
            art.raw = raw;

            % 1. geometry
            t = tic;
            [img, mask, fov] = drscreen.preprocess.standardize(raw, cfg.size);
            timing.geometry = ms(t);
            art.standardized = img;
            art.fov_mask = mask;

            % 2. landmarks (the quality gate needs them)
            t = tic;
            lm = drscreen.preprocess.locateLandmarks(img, mask);
            timing.landmarks = ms(t);
            res.landmarks = lm;
            art.landmarks = lm;

            % 3. quality gate
            t = tic;
            Q = drscreen.preprocess.assessQuality(img, mask, fov, lm);
            timing.quality = ms(t);
            res.quality = Q;
            res.gradeable = Q.gradeable;
            if ~Q.gradeable
                res.decision = 'recapture';
                res.recapture_advice = Q.advice;
                res.grade_label = 'Ungradeable';
                timing.total = ms(tAll);
                res.timing_ms = roundTiming(timing);
                return
            end

            % 4. adaptive enhancement, then re-check the correctable criteria
            t = tic;
            [enhanced, applied] = drscreen.preprocess.adaptiveEnhance(img, mask, Q.issues);
            timing.enhancement = ms(t);
            res.enhancement_applied = applied;
            art.enhanced = enhanced;
            if ~isempty(applied)
                t = tic;
                Q2 = drscreen.preprocess.assessQuality(enhanced, mask, fov, lm);
                timing.quality_recheck = ms(t);
                correctable = {'illumination', 'under_exposure', 'over_exposure', 'contrast', 'noise'};
                vn = fieldnames(Q2.verdicts);
                still = {};
                for i = 1:numel(vn)
                    if strcmp(Q2.verdicts.(vn{i}), 'fail') && ismember(vn{i}, correctable)
                        still{end + 1} = vn{i}; %#ok<AGROW>
                    end
                end
                res.quality = Q2;
                res.quality.first_pass = struct('overall', Q.overall, ...
                    'issues', {Q.issues}, 'scores', Q.scores);
                if ~isempty(still)
                    res.gradeable = false;
                    res.decision = 'recapture';
                    adv = cell(1, numel(still));
                    for i = 1:numel(still)
                        if isfield(C.RECAPTURE_ADVICE, still{i})
                            adv{i} = C.RECAPTURE_ADVICE.(still{i});
                        else
                            adv{i} = sprintf('Recapture: %s inadequate.', still{i});
                        end
                    end
                    res.recapture_advice = [{sprintf(['Enhancement was applied (%s) but the ' ...
                        'image is still not gradeable.'], strjoin(applied, ', '))}, adv];
                    res.grade_label = 'Ungradeable';
                    timing.total = ms(tAll);
                    res.timing_ms = roundTiming(timing);
                    return
                end
            end

            modelIn = drscreen.preprocess.toModelInput(enhanced, mask);
            art.model_input = modelIn;

            % 5. segmentation
            t = tic;
            if ~isempty(obj.Segmenter)
                probs = obj.Segmenter.segment(modelIn, cfg.size);
            else
                probs = zeros(cfg.size, cfg.size, C.NUM_LESION_CLASSES, 'single');
            end
            timing.segmentation = ms(t);
            art.lesion_probs = probs;

            % 6. clinical features
            t = tic;
            vessel = drscreen.Pipeline.vesselProxy(enhanced, mask);
            thr = cfg.lesion_thresholds;
            if isempty(thr), thr = cfg.lesion_threshold; end
            F = drscreen.features.extract(probs, lm, mask, vessel, thr, obj.unassessedLesions());
            timing.clinical_features = ms(t);
            res.clinical_features = struct( ...
                'counts', F.counts, ...
                'per_quadrant', F.per_quadrant, ...
                'quadrants_with_hemorrhage', F.quadrants_with_hemorrhage, ...
                'quadrants_with_beading', F.quadrants_with_beading, ...
                'nv_at_disc', F.nv_at_disc, ...
                'nv_elsewhere', F.nv_elsewhere, ...
                'unassessed', {F.unassessed}, ...
                'lesions_within_1dd_of_fovea', F.lesions_within_1dd_of_fovea, ...
                'nearest_lesion_dd', round(F.nearest_lesion_dd * 100) / 100);
            art.features = F;
            art.vessel_mask = vessel;

            % 7. rule-based grade
            [rg, reasons] = drscreen.features.ruleGrade(F);
            res.rule_based_grade = rg;
            res.rule_based_reasons = reasons;
            [dme, dmeReason] = drscreen.features.dmeRisk(F);
            res.dme_risk = dme;

            % 8. neural grading
            t = tic;
            if ~isempty(obj.Grader)
                gthr = cfg.grade_thresholds;
                if isempty(gthr), gthr = 0.5; end
                pred = obj.Grader.predict(modelIn, cfg.mc_samples, cfg.temperature, gthr);
                p = pred.class_probs(:)';
                res.grade = pred.grade;
                res.class_probabilities = round(p * 1e4) / 1e4;
                res.sight_threatening_probability = sum(p(C.SIGHT_THREATENING_THRESHOLD + 1:end));
                res.referable_probability = drscreen.models.calibrate(pred.referable_prob, cfg.calibrator);
                res.confidence = max(p);
                res.uncertainty = struct( ...
                    'entropy', round(pred.entropy * 1e4) / 1e4, ...
                    'epistemic_variance', round(pred.epistemic * 1e5) / 1e5);
                art.activation = pred.activation;
            else
                % No trained grader: fall back to the rule engine so the service
                % still returns a defensible answer.
                res.grade = rg;
                p = zeros(1, C.NUM_GRADES); p(rg + 1) = 1;
                res.class_probabilities = p;
                res.referable_probability = double(rg >= C.REFERABLE_THRESHOLD);
                res.confidence = 0.5;
                res.uncertainty = struct('entropy', 0, 'epistemic_variance', 0);
            end
            timing.grading = ms(t);
            res.grade_label = C.ICDR_GRADES{res.grade + 1};
            res.referable = res.referable_probability >= cfg.referral_threshold;

            % 9. decision and urgency
            res.agreement = drscreen.Pipeline.agreement(res.grade, rg);
            [res.decision, res.urgency] = obj.decide(res, F);

            % 10. evidence
            res.evidence = drscreen.Pipeline.buildEvidence(F, reasons, dmeReason, res);

            % 11. explanation -- never allowed to break screening
            if explain && cfg.enable_cam && ~isempty(obj.Grader)
                t = tic;
                try
                    art.cam = obj.Grader.gradCamPP(modelIn, mask, art.activation);
                catch err
                    art.cam_error = err.message;
                end
                timing.explanation = ms(t);
            end

            timing.total = ms(tAll);
            res.timing_ms = roundTiming(timing);
        end

        function [decision, urgency] = decide(obj, res, F)
            %DECIDE Combine the calibrated verdict with lesion-based escalation.
            %
            %   Which signals may change a decision is a question of their
            %   measured reliability, not their clinical severity. The grader is
            %   calibrated and validated; the lesion rules run on a segmentation
            %   head trained on a few dozen images, so they escalate only when
            %   the calibrated model is not *confidently* negative. Wired
            %   unconditionally they once marked every case urgent.
            cfg = obj.Cfg;
            lo = cfg.defer_band(1);
            hi = cfg.defer_band(2);
            p = res.referable_probability;
            epiVar = 0;
            if isfield(res.uncertainty, 'epistemic_variance')
                epiVar = res.uncertainty.epistemic_variance;
            end
            confidentlyNegative = p < lo;

            % Neovascularisation escalates on sight -- but only if the channel
            % reporting it was ever supervised. Unsupervised, it returns a blob
            % on nearly every retina and escalated 631/631 real images.
            nvAssessed = ~ismember('neovascularization', F.unassessed);
            if nvAssessed && (F.nv_at_disc || F.nv_elsewhere)
                decision = 'refer'; urgency = 'urgent';
                return
            end

            % Urgent tier on a fitted cut-point of P(grade >= 3), not on the
            % predicted grade (which caught 56 of 110 sight-threatening eyes on
            % Messidor-2, against 86 for the fitted threshold).
            if ~confidentlyNegative
                if isempty(cfg.urgent_threshold)
                    urgent = res.grade >= 3;
                else
                    urgent = res.sight_threatening_probability >= cfg.urgent_threshold;
                end
                if urgent || res.dme_risk >= 2
                    decision = 'refer'; urgency = 'urgent';
                    return
                end
            end

            if (p >= lo && p <= hi) || epiVar > cfg.uncertainty_defer
                decision = 'defer_to_human';
                if p >= cfg.referral_threshold
                    urgency = 'soon';
                else
                    urgency = 'routine';
                end
                return
            end
            if p >= cfg.referral_threshold
                decision = 'refer'; urgency = 'soon';
                return
            end
            decision = 'auto_report'; urgency = 'routine';
        end
    end

    methods (Static)
        function res = emptyResult(imageId, modelVersion)
            % Field order and defaults of the Python ScreeningResult dataclass.
            res = struct( ...
                'image_id', imageId, ...
                'gradeable', true, ...
                'quality', struct(), ...
                'enhancement_applied', {{}}, ...
                'landmarks', struct(), ...
                'grade', -1, ...
                'grade_label', '', ...
                'class_probabilities', [], ...
                'referable', false, ...
                'referable_probability', 0.0, ...
                'confidence', 0.0, ...
                'uncertainty', struct(), ...
                'decision', '', ...
                'urgency', 'routine', ...
                'dme_risk', 0, ...
                'clinical_features', struct(), ...
                'evidence', {{}}, ...
                'rule_based_grade', -1, ...
                'sight_threatening_probability', 0.0, ...
                'rule_based_reasons', {{}}, ...
                'agreement', '', ...
                'recapture_advice', {{}}, ...
                'timing_ms', struct(), ...
                'model_version', modelVersion);
        end

        function v = vesselProxy(img, mask)
            %VESSELPROXY Morphological vessel map (bottom-hat + Otsu), used for
            %calibre statistics when no vessel network is loaded.
            g = single(img(:, :, 2));
            bg = drscreen.cv.morph(g, 'close', 15);
            resp = max(bg - g, 0);
            resp(mask == 0) = 0;
            mx = max(resp(:));
            if mx <= 0
                v = zeros(size(g), 'uint8');
                return
            end
            nrm = uint8(floor(resp / mx * single(255)));
            t = drscreen.cv.otsu(nrm);
            hi = uint8(nrm > t) * 255;
            v = drscreen.cv.morph(hi, 'open', 3);
        end

        function a = agreement(neural, rule)
            d = abs(neural - rule);
            if d == 0
                a = 'exact';
            elseif d == 1
                a = 'within_one_grade';
            else
                a = 'disagree';
            end
        end

        function ev = buildEvidence(F, reasons, dmeReason, res)
            C = drscreen.constants();
            ev = {};
            for i = 1:numel(F.unassessed)
                ev{end + 1} = struct( ...
                    'finding', strrep(F.unassessed{i}, '_', ' '), ...
                    'status', 'not assessed', ...
                    'detail', ['no pixel supervision for this class in the training ' ...
                               'corpus; absence of a detection is not evidence of ' ...
                               'absence of the lesion.']); %#ok<AGROW>
            end
            for i = 1:numel(C.LESION_CLASSES)
                name = C.LESION_CLASSES{i};
                if ismember(name, F.unassessed)
                    continue
                end
                n = F.counts.(name);
                if n == 0
                    continue
                end
                pq = struct();
                for j = 1:numel(C.QUADRANTS)
                    q = C.QUADRANTS{j};
                    if F.per_quadrant.(name).(q)
                        pq.(q) = F.per_quadrant.(name).(q);
                    end
                end
                ev{end + 1} = struct( ...
                    'finding', strrep(name, '_', ' '), ...
                    'count', n, ...
                    'per_quadrant', pq, ...
                    'area_percent', round(F.area_fraction.(name) * 100 * 1e3) / 1e3); %#ok<AGROW>
            end
            for i = 1:numel(reasons)
                ev{end + 1} = struct('criterion', reasons{i}); %#ok<AGROW>
            end
            if ~isempty(dmeReason)
                ev{end + 1} = struct('macular_assessment', dmeReason);
            end
            if strcmp(res.agreement, 'disagree')
                ev{end + 1} = struct('caution', sprintf(['The deep model graded %d (%s) while ' ...
                    'the rule-based criteria give %d (%s). Flagged for human adjudication.'], ...
                    res.grade, C.ICDR_GRADES{res.grade + 1}, res.rule_based_grade, ...
                    C.ICDR_GRADES{res.rule_based_grade + 1}));
            end
        end
    end
end


function v = ms(t)
v = toc(t) * 1000;
end


function T = roundTiming(T)
f = fieldnames(T);
for i = 1:numel(f)
    T.(f{i}) = round(T.(f{i}) * 100) / 100;
end
end
