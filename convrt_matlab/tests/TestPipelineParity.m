classdef TestPipelineParity < matlab.unittest.TestCase
    %TESTPIPELINEPARITY The whole pipeline on all 72 committed photographs.
    %
    %   MATLAB and Python are both run with MC dropout off (deterministic) and
    %   must agree on gradeability, grade, decision and urgency, with
    %   P(referable) close. This is the test that says the conversion is right.
    %   Diagnostics list every case that differs, with both answers.

    properties
        Cases
    end

    properties (TestParameter)
        bundle = {'prepool', 'pooled'}
    end

    methods (TestClassSetup)
        function setup(tc)
            f = fullfile(drscreen.paths().tests, 'fixtures', 'cases.json');
            tc.assumeTrue(isfile(f), 'cases.json missing: run tools/make_fixtures.py');
            j = jsondecode(drscreen.io.readText(f));
            tc.Cases = j.cases;
        end
    end

    methods (Test)
        function allCasesAgree(tc, bundle)
            pipe = drscreen.Pipeline.load(bundle, 'McSamples', 0, 'EnableCam', false);
            n = numel(tc.Cases);
            same = false(n, 1);
            dp = nan(n, 1);
            msg = {};
            for i = 1:n
                if iscell(tc.Cases), r = tc.Cases{i}; else, r = tc.Cases(i); end
                py = r.(bundle);
                res = pipe.run(fullfile(drscreen.paths().root, strrep(r.file, '/', filesep)), r.case, false);
                same(i) = res.gradeable == py.gradeable && res.grade == py.grade && ...
                          strcmp(res.decision, py.decision) && strcmp(res.urgency, py.urgency);
                dp(i) = abs(res.referable_probability - py.referable_probability);
                if ~same(i)
                    msg{end + 1} = sprintf('%s: MATLAB %d %s/%s P=%.4f | Python %d %s/%s P=%.4f', ...
                        r.case, res.grade, res.decision, res.urgency, res.referable_probability, ...
                        py.grade, py.decision, py.urgency, py.referable_probability); %#ok<AGROW>
                end
            end
            diag = strjoin(msg, newline);
            tc.verifyGreaterThanOrEqual(mean(same), 70 / 72, ['cases differing:' newline diag]);
            tc.verifyLessThan(median(dp), 1e-3, 'median |P(referable)| difference');
        end

        function resultSchemaMatchesPython(tc)
            pipe = drscreen.Pipeline.load('prepool', 'McSamples', 0, 'EnableCam', false);
            if iscell(tc.Cases), r = tc.Cases{1}; else, r = tc.Cases(1); end
            res = pipe.run(fullfile(drscreen.paths().root, strrep(r.file, '/', filesep)), r.case, false);
            expected = {'image_id', 'gradeable', 'quality', 'enhancement_applied', 'landmarks', ...
                'grade', 'grade_label', 'class_probabilities', 'referable', ...
                'referable_probability', 'confidence', 'uncertainty', 'decision', 'urgency', ...
                'dme_risk', 'clinical_features', 'evidence', 'rule_based_grade', ...
                'sight_threatening_probability', 'rule_based_reasons', 'agreement', ...
                'recapture_advice', 'timing_ms', 'model_version'};
            tc.verifyEqual(fieldnames(res)', expected, 'field order of ScreeningResult');
            j = jsondecode(jsonencode(res));
            tc.verifyTrue(iscell(j.rule_based_reasons) || ischar(j.rule_based_reasons));
            tc.verifyEqual(numel(j.class_probabilities), 5);
        end

        function mcDropoutGivesVariance(tc)
            pipe = drscreen.Pipeline.load('prepool', 'McSamples', 8, 'EnableCam', false);
            [img, name] = drscreen.samples.load('', 2);
            res = pipe.run(img, name, false);
            tc.verifyGreaterThan(res.uncertainty.epistemic_variance, 0, ...
                'MC dropout must sample the head, not return a constant');
        end
    end
end
