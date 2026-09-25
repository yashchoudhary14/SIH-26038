function [p, PatientBus] = district_model_params()
%DISTRICT_MODEL_PARAMS  Every variable district_model.slx uses, in one place.
%
%   Edit a value here and run the model again, or change it in the workspace
%   (p.n_technicians = 3;) before pressing Run. The model loads this file
%   itself when it opens if `p` is not already in the workspace.
%
%       [p, PatientBus] = district_model_params();
%       sim('district_model');
%
%   Two variables change the model's structure, not just its numbers:
%   include_gate and add_scopes. After changing either, rebuild:
%
%       build_district_model(p);
%
%   Time base: MINUTES.

% -- session -------------------------------------------------------------------
p.sim_stop = 360;                    % one screening session (the model's stop time)

% -- arrivals ------------------------------------------------------------------
% Group-arrival events happen at rate
%
%     lambda(t) = lambda_peak * exp(-(t - t_peak)^2 / (2 sigma^2)) * (N - A(t)) / N
%
% where A(t) is the number of patients arrived by time t, and each event brings
% G = 1 + Poisson(mu) patients. lambda_peak is not set by hand: the Arrivals
% block derives it (district_peak_rate.m) so that the session brings
% sim_stop / mean_interarrival patients on average, one every
% mean_interarrival minutes. N must exceed that number.
p.mean_interarrival = 3;             % minutes between patients, on average over the session
p.t_peak = 150;                      % arrivals peak this many minutes after opening
p.sigma  = 90;                       % minutes: how widely arrivals spread around the peak
p.N      = 200;                      % catchment: patients who may come this session
p.mu     = 0.5;                      % mean companions per arrival event

% -- patients ------------------------------------------------------------------
p.prevalence_referable = 0.08;       % share of patients with referable DR

% -- image quality: the AI sorts every captured image into three classes --------
% (shares of captured images; normalised if they do not add up to 1)
p.p_pass   = 0.70;                   % good: graded as it is
p.p_repair = 0.15;                   % borderline: restored first
p.p_reject = 0.15;                   % bad: cannot be graded
p.p_restore = 0.873;                 % borderline images the restoration makes gradeable
p.max_attempts = 3;                  % captures allowed per patient (used when include_gate = 1)

% -- service times, minutes ------------------------------------------------------
p.t_capture = 5;                     % photograph both eyes
p.t_quality = 0.01;                  % AI quality check (runs in milliseconds on CPU)
p.t_restore = 0.02;                  % AI restoration of a borderline image
p.t_grade   = 0.25;                  % AI grading on the edge device
p.t_upload  = 0.67;                  % send a referred image to the district hub
p.t_review  = 0.5;                   % the doctor's read of an AI-graded image

% -- the grader's operating point ---------------------------------------------------
p.sensitivity = 0.986;
p.specificity = 0.873;

% -- resources ---------------------------------------------------------------------
p.n_technicians      = 5;
p.n_cameras          = 5;
p.n_edge_devices     = 5;
p.n_ophthalmologists = 5;

% -- queue capacities ------------------------------------------------------------
p.cap_waiting_room  = 500;
p.cap_upload_buffer = 1000;
p.cap_review_queue  = 1000;

% -- structure (rebuild with build_district_model after changing) -------------------
p.include_gate = 0;                  % 1: bad images go back for another capture, up to max_attempts
                                     % 0: bad images go straight to the doctor as ungradable
p.add_scopes   = 1;                  % 1: utilisation and queue scopes, and count displays

% -- replication -------------------------------------------------------------------
% every random stream in the model is seeded from this; change it for another,
% independent run
p.seed = 1;

% -- entity type ---------------------------------------------------------------------
%   true_status    1 if the patient has referable DR
%   quality_class  the AI's call on the image: 1 good, 2 borderline, 3 bad
%   restored       restoration outcome: 0 not needed, 1 restored, 2 failed
%   ungradable     1 if the image went to the doctor ungraded
%   attempts       captures so far
%   flagged        1 if the patient is referred
%   route          where RouteDecision sends the patient: 1 cleared, 2 doctor, 3 recapture
names = {'true_status', 'quality_class', 'restored', 'ungradable', ...
         'attempts', 'flagged', 'route'};
PatientBus = Simulink.Bus;
PatientBus.Description = 'One patient moving through the district screening model';
els = Simulink.BusElement.empty;
for k = 1:numel(names)
    e = Simulink.BusElement;
    e.Name = names{k};
    e.DataType = 'double';
    els(end + 1) = e; %#ok<AGROW>
end
PatientBus.Elements = els;
end
