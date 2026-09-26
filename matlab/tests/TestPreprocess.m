classdef TestPreprocess < matlab.unittest.TestCase
    %TESTPREPROCESS Each pipeline stage on real photographs, against Python.
    %
    %   Every stage is fed the *Python* output of the stage before it, so a
    %   mismatch is attributed to the stage that caused it rather than
    %   inherited from upstream. Three cases with full intermediate arrays:
    %   an auto-report, a sight-threatening referral, a moderate referral.

    properties (TestParameter)
        caseName = {'grade0_case1', 'grade3_case1', 'grade2_case1'}
    end

    properties
        Rows
        Pipe
    end

    methods (TestClassSetup)
        function setup(tc)
            f = fullfile(drscreen.paths().tests, 'fixtures', 'cases.json');
            tc.assumeTrue(isfile(f), 'cases.json missing: run tools/make_fixtures.py');
            j = jsondecode(drscreen.io.readText(f));
            tc.Rows = containers.Map();
            for i = 1:numel(j.cases)
                if iscell(j.cases), r = j.cases{i}; else, r = j.cases(i); end
                tc.Rows(r.case) = r;
            end
            tc.Pipe = drscreen.Pipeline.load('prepool', 'McSamples', 0);
        end
    end

    methods
        function [A, row, raw] = fixture(tc, name)
            A = load(fullfile(drscreen.paths().tests, 'fixtures', 'arrays', [name '.mat']));
            row = tc.Rows(name);
            raw = drscreen.io.readImage(fullfile(drscreen.paths().root, strrep(row.file, '/', filesep)));
        end
    end

    methods (Test)
        function geometry(tc, caseName)
            [A, ~, raw] = tc.fixture(caseName);
            [img, mask] = drscreen.preprocess.standardize(raw, 512);
            tc.verifyEqual(mask, A.fov_mask, 'FOV mask');
            d = abs(double(img) - double(A.standardized));
            tc.verifyLessThanOrEqual(mean(d(:) > 0), 0.001, ...
                'standardised image: JPEG decoders may differ on a handful of pixels, no more');
        end

        function landmarksAndQuality(tc, caseName)
            [A, row] = tc.fixture(caseName);
            py = row.prepool;
            lm = drscreen.preprocess.locateLandmarks(A.standardized, A.fov_mask);
            tc.verifyLessThanOrEqual(norm(lm.disc_xy - py.landmarks.disc_xy(:)'), 2, 'disc');
            tc.verifyLessThanOrEqual(norm(lm.fovea_xy - py.landmarks.fovea_xy(:)'), 3, 'fovea');
            fov = drscreen.preprocess.detectFov(drscreen.io.readImage( ...
                fullfile(drscreen.paths().root, strrep(row.file, '/', filesep))));
            q = drscreen.preprocess.assessQuality(A.standardized, A.fov_mask, fov, lm);
            ref = py.quality.first_pass.scores;
            for k = fieldnames(ref)'
                tc.verifyLessThan(abs(q.scores.(k{1}) - ref.(k{1})), 0.02, ['quality ' k{1}]);
            end
            tc.verifyEqual(q.overall, py.quality.first_pass.overall);
        end

        function enhancementAndModelInput(tc, caseName)
            [A, row] = tc.fixture(caseName);
            issues = row.prepool.quality.first_pass.issues;
            if isempty(issues), issues = {}; end
            [enh, applied] = drscreen.preprocess.adaptiveEnhance(A.standardized, A.fov_mask, cellstr(issues));
            tc.verifyEqual(applied(:), cellstr(row.prepool.enhancement_applied(:)));
            % bit-exact, including grey world's float32 sums and clahe_lab's
            % integer Lab round trip
            tc.verifyEqual(enh, A.enhanced, 'enhanced image');

            mi = drscreen.preprocess.toModelInput(A.enhanced, A.fov_mask);
            tc.verifyEqual(mi, A.model_input, 'model input from the Python-enhanced image');
        end

        function segmentationAndActivation(tc, caseName)
            A = tc.fixture(caseName);
            probs = tc.Pipe.Segmenter.segment(A.model_input, 512);
            % Measured against float32 PyTorch (fixtures recorded with TF32 off):
            % lesion maps ~1e-5, activation ~2e-6 relative, CAM <= 8e-4.
            tc.verifyLessThan(max(abs(double(probs(:)) - double(A.lesion_probs(:)))), 1e-4, 'lesion maps');
            act = tc.Pipe.Grader.trunk(tc.Pipe.Grader.normalize(A.model_input));
            tc.verifyLessThan(max(abs(double(act(:)) - double(A.activation(:)))) / ...
                max(abs(double(A.activation(:)))), 2e-5, 'CAM-layer activation');
            if isfield(A, 'cam')
                cam = tc.Pipe.Grader.gradCamPP(A.model_input, A.fov_mask, act);
                tc.verifyLessThan(max(abs(double(cam(:)) - double(A.cam(:)))), 3e-3, 'Grad-CAM++');
            end
        end

        function vesselProxy(tc, caseName)
            A = tc.fixture(caseName);
            v = drscreen.Pipeline.vesselProxy(A.enhanced, A.fov_mask);
            tc.verifyGreaterThanOrEqual(mean(v(:) == A.vessel_mask(:)), 0.999);
        end
    end
end
