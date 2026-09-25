function C = constants()
%CONSTANTS Clinical constants shared across the pipeline.
%
%   Everything a clinician or an evaluator might want to audit is stated once
%   here -- the grading scale, the lesion taxonomy, the target operating point
%   -- and never re-hardcoded inside model code. Mirrors
%   src/drscreen/constants.py in the Python repository.

persistent cache
if ~isempty(cache)
    C = cache;
    return
end

% International Clinical Diabetic Retinopathy (ICDR) severity scale, grades 0..4
C.ICDR_GRADES = {'No apparent DR', 'Mild NPDR', 'Moderate NPDR', ...
                 'Severe NPDR', 'Proliferative DR'};
C.NUM_GRADES = 5;
C.REFERABLE_THRESHOLD = 2;            % grade >= 2 is referred
C.SIGHT_THREATENING_THRESHOLD = 3;    % grade >= 3 is sight-threatening

% problem-statement targets for referable DR
C.TARGET_SENSITIVITY = 0.90;
C.TARGET_SPECIFICITY = 0.85;

% lesion taxonomy -- channel order is fixed across the whole codebase
C.LESION_CLASSES = {'microaneurysm', 'hemorrhage', 'hard_exudate', ...
                    'soft_exudate', 'neovascularization'};
C.NUM_LESION_CLASSES = 5;

% Classes the public corpora annotate at pixel level. IDRiD ships no
% neovascularisation masks, so on real data that channel is never supervised:
% a zero count for it means "not assessed", not "absent".
C.PIXEL_ANNOTATED_LESION_CLASSES = {'microaneurysm', 'hemorrhage', ...
                                    'hard_exudate', 'soft_exudate'};

% overlay colours, RGB (the Python constants are BGR)
C.LESION_COLORS = struct( ...
    'microaneurysm',      [255   0   0], ...
    'hemorrhage',         [255  96   0], ...
    'hard_exudate',       [255 255   0], ...
    'soft_exudate',       [255 255 255], ...
    'neovascularization', [255   0 255]);

% minimum blob area in px at 512x512, per lesion class
C.MIN_AREA_512 = struct('microaneurysm', 3, 'hemorrhage', 12, ...
                        'hard_exudate', 4, 'soft_exudate', 25, ...
                        'neovascularization', 20);

C.QUADRANTS = {'superior', 'inferior', 'nasal', 'temporal'};

% Optic-disc score: brightness * ((1 - w) + w * vessel_density). Fitted on
% IDRiD's hand-marked disc centres; the phantom-tuned 0.70 was worse on real
% retinas (96.0% vs 98.5% within 1 DD).
C.DISC_VESSEL_WEIGHT = 0.10;

C.QUALITY_LABELS = {'ungradeable', 'borderline', 'good'};

% human-readable recapture instructions keyed by the failing quality metric
C.RECAPTURE_ADVICE = struct( ...
    'focus',          'Image is out of focus. Re-focus on the optic disc and hold steady before capture.', ...
    'illumination',   'Uneven or insufficient illumination. Re-centre the flash and check the working distance.', ...
    'over_exposure',  'Image is over-exposed / washed out. Reduce flash intensity and recapture.', ...
    'under_exposure', 'Image is too dark. Increase flash intensity or dilate the pupil before recapture.', ...
    'fov',            'Field of view is clipped. Centre the retina in the frame and recapture the full 45-degree field.', ...
    'artifact',       'Large artifact / reflection detected. Clean the lens, ask the patient to blink, then recapture.', ...
    'macula',         'Macula is not visible or is obscured. Direct the patient''s gaze so the macula is centred.', ...
    'contrast',       'Vessel contrast is too low to grade. Consider pupil dilation before recapture.');

% ImageNet normalisation the networks were trained with
C.IMAGENET_MEAN = single([0.485 0.456 0.406]);
C.IMAGENET_STD  = single([0.229 0.224 0.225]);

% Model bundle used when a launcher is not told which one (models/<bundle>/).
%   'pooled'  -- grader trained on all four corpora; the bundle the showcase
%                set (data/showcase) was selected with. The default.
%   'prepool' -- grader trained on APTOS-2019 + IDRiD, Messidor-2 held out;
%                the bundle behind the external-validation figures.
% Every launcher, the console and the web server read this one value.
C.DEFAULT_BUNDLE = 'pooled';

cache = C;
end
