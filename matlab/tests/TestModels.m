classdef TestModels < matlab.unittest.TestCase
    %TESTMODELS Both networks against PyTorch on fixed probe tensors.
    %
    %   The probes are already-normalised tensors, so these tests isolate the
    %   network code from preprocessing. The EfficientNet is checked after the
    %   stem and after every timm stage, so a wrong layer is located, not just
    %   detected. The network tests run twice: on the plain-MATLAB convolution
    %   and on the Deep Learning Toolbox's dlconv (skipped if not installed).

    properties
        M
        Stages = [1 3 5 8 11 15 16]     % last block index of each timm stage
    end

    properties (TestParameter)
        bundle = {'prepool', 'pooled'}
        backend = {'toolbox', 'plain'}
    end

    methods (TestClassSetup)
        function loadFixtures(tc)
            f = fullfile(drscreen.paths().tests, 'fixtures', 'models.mat');
            tc.assumeTrue(isfile(f), 'models.mat missing: run tools/make_fixtures.py');
            tc.M = load(f);
        end
    end

    methods
        function useBackend(tc, backend)
            before = drscreen.nn.useToolbox();
            tc.addTeardown(@() drscreen.nn.useToolbox(before));
            want = strcmp(backend, 'toolbox');
            got = drscreen.nn.useToolbox(want);
            tc.assumeEqual(got, want, 'Deep Learning Toolbox not installed');
        end
    end

    methods (Test)
        function efficientNetStages(tc, bundle, backend)
            tc.useBackend(backend);
            g = drscreen.models.Grader(fullfile(drscreen.paths().models, bundle));
            [A, taps] = g.trunk(single(tc.M.probe128_hwc), [0 tc.Stages]);
            ref = tc.M.([bundle '_stem128']);
            tc.verifyLessThan(relErr(taps{1}, ref), 1e-4, 'stem');
            for s = 1:numel(tc.Stages)
                ref = tc.M.(sprintf('%s_stage%d_128', bundle, s - 1));
                tc.verifyLessThan(relErr(taps{s + 1}, ref), 1e-4, sprintf('after stage %d', s - 1));
            end
            tc.verifyLessThan(relErr(A, tc.M.([bundle '_act128'])), 1e-4, 'CAM layer');
        end

        function cornLogits(tc, bundle, backend)
            tc.useBackend(backend);
            g = drscreen.models.Grader(fullfile(drscreen.paths().models, bundle));
            for tag = {'128', '512'}
                A = g.trunk(single(tc.M.(['probe' tag{1} '_hwc'])));
                tc.verifyLessThan(relErr(A, tc.M.([bundle '_act' tag{1}])), 1e-4, ['activation ' tag{1}]);
                z = g.head(A);
                ref = tc.M.([bundle '_logits' tag{1}]);
                tc.verifyLessThan(max(abs(double(z(:)) - double(ref(:)))), 1e-3, ['logits ' tag{1}]);
            end
        end

        function unetLogits(tc, backend)
            tc.useBackend(backend);
            s = drscreen.models.Segmenter(drscreen.paths().models);
            got = s.forward(single(tc.M.probe128_hwc));
            tc.verifyLessThan(relErr(got, tc.M.unet_logits128), 1e-4);
        end

        function cornIsMonotoneAndNormalised(tc)
            z = randn(4, 200) * 3;
            P = drscreen.models.cornClassProbs(z);
            tc.verifyLessThan(max(abs(sum(P, 1) - 1)), 1e-12);
            tc.verifyGreaterThanOrEqual(min(P(:)), 0);
            cum = cumprod(1 ./ (1 + exp(-z)), 1);
            tc.verifyTrue(all(all(diff(cum, 1, 1) <= 1e-12)), 'P(y>k) must never increase with k');
        end

        function gradeStopsAtFirstFailure(tc)
            % a low later cut-point must not pass after an earlier one failed
            tc.verifyEqual(drscreen.models.gradeFromCumulative([0.9 0.2 0.15 0.1], [0.5 0.5 0.1 0.05]), 1);
            tc.verifyEqual(drscreen.models.gradeFromCumulative([0.9 0.8 0.7 0.6], 0.5), 4);
            tc.verifyEqual(drscreen.models.gradeFromCumulative([0.4 0.3 0.2 0.1], 0.5), 0);
        end

        function calibratorIsMonotoneAndStrict(tc)
            cfg = jsondecode(fileread(fullfile(drscreen.paths().models, 'prepool', 'pipeline.json')));
            p = linspace(0, 1, 2001);
            q = drscreen.models.calibrate(p, cfg.calibrator);
            tc.verifyTrue(all(diff(q) > 0), 'tie-break blend must keep a strict order');
            tc.verifyGreaterThanOrEqual(min(q), 0);
            tc.verifyLessThanOrEqual(max(q), 1);
        end
    end
end


function e = relErr(a, b)
a = double(a); b = double(b);
e = max(abs(a(:) - b(:))) / max(1, max(abs(b(:))));
end
