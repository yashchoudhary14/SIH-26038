function mdl = build_district_model(p, file)
%BUILD_DISTRICT_MODEL  Build district_model.slx from its parameters.
%
%   build_district_model()              % from district_model_params.m
%   build_district_model(p)             % from a struct you have edited
%   build_district_model(p, file)       % somewhere else, e.g. a variant
%
%   One screening session at a health centre, and the doctor review it
%   feeds:
%
%   1 CAPTURE AT THE HEALTH CENTRE
%     Patients arrive in groups (see district_model_params.m), wait in the
%     waiting hall, take a technician and a camera, and are photographed.
%   2 AI ON THE EDGE DEVICE
%     The AI checks the image and sorts it: good, borderline or bad.
%     Borderline images are restored; p_restore of them become gradeable.
%     Gradeable images are graded, and the patient is cleared on the spot
%     or referred. A bad image, or one the restoration could not save, goes
%     back for another capture when include_gate = 1 (up to max_attempts),
%     and otherwise to the doctor as ungradable.
%   3 DOCTOR REVIEW AT THE DISTRICT HUB
%     Referred images are uploaded, queue for an ophthalmologist, and are
%     read.
%
%   Every number the blocks use is read from `p` when the model runs, so
%   editing p (or district_model_params.m) is enough. include_gate and
%   add_scopes change which blocks exist, so they need a rebuild.

here = fileparts(mfilename('fullpath'));
[p0, PatientBus] = district_model_params();
if nargin < 1 || isempty(p), p = p0; end
if nargin < 2 || isempty(file), file = fullfile(here, 'district_model.slx'); end
[~, mdl] = fileparts(file);
assignin('base', 'p', p); assignin('base', 'PatientBus', PatientBus);
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
m = [mdl '/'];
gate = logical(p.include_gate);
scopes = logical(p.add_scopes);

% ---- layout (Simulink pixels) ---------------------------------------------------
BW = 90;                                   % block width
X0 = 40;
XR = 1440;                                 % left edge of the last column
W = XR + BW;                               % right edge of the diagram
Y1 = 140;                                  % 1: capture row
YR = 200;  Y12 = 218;                      % recapture loop, and the carriage return to row 2
YG = 340;  YM = 440;  YB = 540;            % 2: good / main / bad lanes
YL = 568;  Y23 = 595;                      % recapture loop, carriage return to row 3
Y3 = 720;                                  % 3: review row
YP = 872;                                  % resource pools
spread = @(k, n) X0 + round(k * (XR - X0) / (n - 1));
XL = X0 - 26;                              % the carriage returns run down here, left of the panels

    function blk(type, name, x, yc, hgt, varargin)
        add_block(['built-in/' type], [m name], 'Position', [x, yc - hgt / 2, x + BW, yc + hgt / 2], varargin{:});
    end

% ---- 1 capture at the health centre ------------------------------------------------
row1 = {'Arrivals', 'WaitingRoom', 'AcqTechnician', 'AcqCamera', 'CaptureStation', 'RelTechnician', 'RelCamera'};
if gate, row1 = [row1(1), {'MergeRetakes'}, row1(2:end)]; end
x1 = containers.Map(row1, arrayfun(@(k) spread(k, numel(row1)), 0:numel(row1) - 1, 'UniformOutput', false));

% with the recapture loop, Arrivals feeds MergeRetakes' upper port in a straight line
blk('EntityGenerator', 'Arrivals', x1('Arrivals'), Y1 - 15 * gate, 60, ...
    'TimeSource', 'MATLAB action', 'IntergenerationTimeAction', ARRIVALS, ...
    'GenerateEntityAtSimulationStart', 'off', 'GenerateAction', GENERATE, ...
    'EntityType', 'Bus object', 'EntityTypeName', 'PatientBus');
if gate
    blk('EntityInputSwitch', 'MergeRetakes', x1('MergeRetakes'), Y1, 60, ...
        'NumberInputPorts', '2', 'ActivePortSelection', 'All');
end
blk('Queue', 'WaitingRoom', x1('WaitingRoom'), Y1, 60, 'Capacity', 'p.cap_waiting_room', 'QueueType', 'FIFO');
blk('EntityResourceAcquirer', 'AcqTechnician', x1('AcqTechnician'), Y1, 60, 'ResourceName', 'Technician', 'ResourceAmount', '1');
% a patient holding a technician waits here for a camera
blk('EntityResourceAcquirer', 'AcqCamera', x1('AcqCamera'), Y1, 60, 'ResourceName', 'Camera', 'ResourceAmount', '1', ...
    'NumberWaitingEntities', 'p.n_technicians');
% servers take anyone who reaches them: how many work at once is set by the
% resources a patient must hold to get there (a technician and a camera here)
blk('EntityServer', 'CaptureStation', x1('CaptureStation'), Y1, 60, ...
    'Capacity', 'Inf', 'ServiceTimeSource', 'Dialog', 'ServiceTimeValue', 'p.t_capture', ...
    'ServiceCompleteAction', 'entity.attempts = entity.attempts + 1;');
blk('EntityResourceReleaser', 'RelTechnician', x1('RelTechnician'), Y1, 60, 'ResourceName', 'Technician');
blk('EntityResourceReleaser', 'RelCamera', x1('RelCamera'), Y1, 60, 'ResourceName', 'Camera');

% ---- 2 AI on the edge device ----------------------------------------------------------
X2 = [40 170 300 470 610 760 890 1030 1160 1290 1440];
dev = 'Inf';                               % limited by the edge devices held
blk('EntityResourceAcquirer', 'AcqDevice', X2(1), YM, 60, 'ResourceName', 'EdgeDevice', 'ResourceAmount', '1', ...
    'NumberWaitingEntities', 'p.cap_waiting_room');
blk('EntityServer', 'QualityCheck', X2(2), YM, 60, 'Capacity', dev, ...
    'ServiceTimeSource', 'Dialog', 'ServiceTimeValue', 'p.t_quality', 'ServiceCompleteAction', QUALITY);
blk('EntityOutputSwitch', 'QualitySwitch', X2(3), YM, 90, ...
    'NumberOutputPorts', '3', 'SwitchingCriterion', 'From attribute', 'SwitchAttributeName', 'quality_class');
blk('EntityServer', 'Restoration', X2(4), YM, 60, 'Capacity', dev, ...
    'ServiceTimeSource', 'Dialog', 'ServiceTimeValue', 'p.t_restore', 'ServiceCompleteAction', RESTORE);
blk('EntityOutputSwitch', 'RestoreSwitch', X2(5), YM, 60, ...
    'NumberOutputPorts', '2', 'SwitchingCriterion', 'From attribute', 'SwitchAttributeName', 'restored');
blk('EntityInputSwitch', 'GradeMerge', X2(6), YG, 60, 'NumberInputPorts', '2', 'ActivePortSelection', 'All');
blk('EntityServer', 'GradeOnDevice', X2(7), YG, 60, 'Capacity', dev, ...
    'ServiceTimeSource', 'Dialog', 'ServiceTimeValue', 'p.t_grade', 'ServiceCompleteAction', GRADE);
blk('EntityInputSwitch', 'DeviceMerge', X2(8), YM, 90, 'NumberInputPorts', '3', 'ActivePortSelection', 'All');
blk('EntityResourceReleaser', 'RelDevice', X2(9), YM, 60, 'ResourceName', 'EdgeDevice');
nr = 2 + gate;
blk('EntityOutputSwitch', 'RouteDecision', X2(10), YM, 30 * nr, ...
    'NumberOutputPorts', num2str(nr), 'SwitchingCriterion', 'From attribute', 'SwitchAttributeName', 'route');
yClear = YM - 15 * (nr - 1);                % RouteDecision's first port
blk('EntityTerminator', 'ClearedOnSpot', X2(11), yClear, 60);

% ---- 3 doctor review at the district hub ---------------------------------------------
row3 = {'UploadBuffer', 'Upload', 'ReviewQueue', 'AcqOphthalmologist', 'SpecialistReview', 'RelOphthalmologist', 'Reviewed'};
x3 = containers.Map(row3, arrayfun(@(k) spread(k, numel(row3)), 0:numel(row3) - 1, 'UniformOutput', false));
blk('Queue', 'UploadBuffer', x3('UploadBuffer'), Y3, 60, 'Capacity', 'p.cap_upload_buffer', 'QueueType', 'FIFO');
blk('EntityServer', 'Upload', x3('Upload'), Y3, 60, 'Capacity', '1', ...          % one upload at a time
    'ServiceTimeSource', 'Dialog', 'ServiceTimeValue', 'p.t_upload');
blk('Queue', 'ReviewQueue', x3('ReviewQueue'), Y3, 60, 'Capacity', 'p.cap_review_queue', 'QueueType', 'FIFO');
blk('EntityResourceAcquirer', 'AcqOphthalmologist', x3('AcqOphthalmologist'), Y3, 60, ...
    'ResourceName', 'Ophthalmologist', 'ResourceAmount', '1');
blk('EntityServer', 'SpecialistReview', x3('SpecialistReview'), Y3, 60, 'Capacity', 'Inf', ...
    'ServiceTimeSource', 'Dialog', 'ServiceTimeValue', 'p.t_review');
blk('EntityResourceReleaser', 'RelOphthalmologist', x3('RelOphthalmologist'), Y3, 60, 'ResourceName', 'Ophthalmologist');
blk('EntityTerminator', 'Reviewed', x3('Reviewed'), Y3, 60);

% ---- resources ---------------------------------------------------------------------------
pools = {'PoolTechnician', 'Technician', 'p.n_technicians'; 'PoolCamera', 'Camera', 'p.n_cameras'; ...
         'PoolEdgeDevice', 'EdgeDevice', 'p.n_edge_devices'; 'PoolOphthalmologist', 'Ophthalmologist', 'p.n_ophthalmologists'};
xp = @(i) 300 + (i - 1) * 330;
for i = 1:size(pools, 1)
    blk('EntityResourcePool', pools{i, 1}, xp(i), YP, 60, ...
        'ResourceName', pools{i, 2}, 'ResourceAmount', pools{i, 3});
end

% ---- statistics shown on scopes and displays ------------------------------------------
if scopes
    sinks = {'Arrivals', 'NumberEntitiesDeparted', 'Display', 'Patients_Arrived'; ...
             'WaitingRoom', 'NumberEntitiesInBlock', 'Scope', 'WaitingRoomLen'; ...
             'ClearedOnSpot', 'NumberEntitiesArrived', 'Display', 'Patients_Cleared'; ...
             'ReviewQueue', 'NumberEntitiesInBlock', 'Scope', 'ReviewQueueLen'; ...
             'Reviewed', 'NumberEntitiesArrived', 'Display', 'Patients_Reviewed'};
    utils = {'PoolTechnician', 'Util_Technician'; 'PoolCamera', 'Util_Camera'; ...
             'PoolEdgeDevice', 'Util_EdgeDevice'; 'PoolOphthalmologist', 'Util_Ophthalmologist'};
    for i = 1:size(sinks, 1)
        set_param([m sinks{i, 1}], sinks{i, 2}, 'on');
    end
    for i = 1:size(utils, 1)
        set_param([m utils{i, 1}], 'AverageUtilization', 'on');
    end
end

% ---- connections ---------------------------------------------------------------------------
    function line(pts)
        add_line(mdl, pts);
    end
    function xy = port(name, side, k)
        % a block's k-th entity port on its left ('in') or right ('out') side,
        % or its statistic port on the top edge ('stat')
        b = [m name]; ph = get_param(b, 'PortHandles'); pos = get_param(b, 'Position');
        hs = ph.Outport; if strcmp(side, 'in'), hs = ph.Inport; end
        P = cell2mat(arrayfun(@(h) get_param(h, 'Position'), hs(:), 'UniformOutput', false));
        switch side                             % (ports sit just outside the block's edge)
            case 'in',   P = P(P(:, 1) <= pos(1) + 1, :);
            case 'out',  P = P(P(:, 1) >= pos(3) - 1, :);
            case 'stat', P = P(P(:, 2) <= pos(2) + 1, :);
        end
        P = sortrows(P, 2);
        if nargin < 3, k = 1; end
        xy = P(k, :);
    end
    function straight(a, b, ka, kb)
        if nargin < 3, ka = 1; end
        if nargin < 4, kb = 1; end
        s = port(a, 'out', ka); d = port(b, 'in', kb);
        if abs(s(2) - d(2)) < 1
            line([s; d]);
        else                                    % a step halfway
            xm = round((s(1) + d(1)) / 2);
            line([s; xm s(2); xm d(2); d]);
        end
    end

% row 1
for i = 1:numel(row1) - 1
    straight(row1{i}, row1{i + 1});
end
% carriage return to row 2
s = port('RelCamera', 'out'); d = port('AcqDevice', 'in');
line([s; W + 20, s(2); W + 20, Y12; XL, Y12; XL, d(2); d]);

% row 2: the AI sorts, restores, grades
straight('AcqDevice', 'QualityCheck');
straight('QualityCheck', 'QualitySwitch');
xa = X2(3) + BW + 15;                       % the fan out of QualitySwitch
xc = X2(8) - 15;                            % the fan into DeviceMerge
s = port('QualitySwitch', 'out', 1); d = port('GradeMerge', 'in', 1);          % good
line([s; xa s(2); xa d(2); d]);
straight('QualitySwitch', 'Restoration', 2);                                   % borderline
s = port('QualitySwitch', 'out', 3); d = port('DeviceMerge', 'in', 3);         % bad
line([s; xa s(2); xa YB; xc YB; xc d(2); d]);
straight('Restoration', 'RestoreSwitch');
s = port('RestoreSwitch', 'out', 1); d = port('GradeMerge', 'in', 2);          % restored
line([s; X2(6) - 15, s(2); X2(6) - 15, d(2); d]);
s = port('RestoreSwitch', 'out', 2); d = port('DeviceMerge', 'in', 2);         % not restored
line([s; xc - 15, s(2); xc - 15, d(2); d]);
straight('GradeMerge', 'GradeOnDevice');
s = port('GradeOnDevice', 'out'); d = port('DeviceMerge', 'in', 1);            % graded
line([s; xc s(2); xc d(2); d]);
straight('DeviceMerge', 'RelDevice');
straight('RelDevice', 'RouteDecision');
straight('RouteDecision', 'ClearedOnSpot', 1);                                  % cleared
xg = X2(10) + BW + 30;
s = port('RouteDecision', 'out', 2); d = port('UploadBuffer', 'in');           % to the doctor
line([s; xg s(2); xg Y23; XL, Y23; XL, d(2); d]);
if gate                                                                          % another capture
    s = port('RouteDecision', 'out', 3); d = port('MergeRetakes', 'in', 2);
    xm = x1('MergeRetakes') - 15;
    line([s; xg - 15, s(2); xg - 15, YL; XL - 8, YL; XL - 8, YR; xm YR; xm d(2); d]);
end

% row 3
for i = 1:numel(row3) - 1
    straight(row3{i}, row3{i + 1});
end

% scopes and displays, above the block they read
if scopes
    for i = 1:size(sinks, 1)
        s = port(sinks{i, 1}, 'stat');
        if strcmp(sinks{i, 3}, 'Scope'), w = 40; h = 40; else, w = 90; h = 30; end
        yc = s(2) - 45; xl = s(1) + 25;
        add_block(['built-in/' sinks{i, 3}], [m sinks{i, 4}], 'Position', [xl, yc - h / 2, xl + w, yc + h / 2]);
        line([s; s(1) yc; port(sinks{i, 4}, 'in')]);
    end
    for i = 1:size(utils, 1)                    % each pool's utilisation, to its right
        ph = get_param([m utils{i, 1}], 'PortHandles');
        s = get_param(ph.Outport(1), 'Position');     % a pool's only port, on its top edge
        pos = get_param([m utils{i, 1}], 'Position');
        xs = pos(3) + 45; yc = (pos(2) + pos(4)) / 2;
        add_block('built-in/Scope', [m utils{i, 2}], 'Position', [xs, yc - 20, xs + 40, yc + 20]);
        d = port(utils{i, 2}, 'in');
        line([s; s(1), pos(2) - 14; xs - 20, pos(2) - 14; xs - 20, d(2); d]);
    end
end

% ---- what the picture says ---------------------------------------------------------------
    function area(txt, pos, rgb)
        a = add_block('built-in/Area', [m txt]);
        set_param(a, 'Position', pos, 'ForegroundColor', sprintf('[%g %g %g]', rgb), ...
            'FontSize', 13, 'FontWeight', 'bold');
    end
    function note(txt, x, y)
        a = add_block('built-in/Note', [m txt]);
        set_param(a, 'Position', [x, y - 8, x + 7 * numel(txt) + 6, y + 6], 'HorizontalAlignment', 'left', ...
            'ForegroundColor', '[0.30 0.33 0.40]', 'BackgroundColor', 'automatic', 'FontSize', 10, 'FontAngle', 'italic');
    end
XA = X0 - 12; XB = W + 90;                 % (the returns at XL run outside the panels)
area('1  CAPTURE AT THE HEALTH CENTRE', [XA 8 XB 192], [0.925 0.950 0.990]);
area('2  AI ON THE EDGE DEVICE: QUALITY CHECK, RESTORATION, GRADING', [XA 232 XB 580], [0.915 0.975 0.955]);
area('3  DOCTOR REVIEW AT THE DISTRICT HUB', [XA 608 XB 785], [0.995 0.960 0.910]);
area('RESOURCE POOLS (utilisation on the scopes)', [XA 800 XB 925], [0.950 0.950 0.955]);
% the arrival process, where row 1 has room for it
a = add_block('built-in/Note', [m 'arrivals formula']);
set_param(a, 'Interpreter', 'tex', 'Text', ['Arrivals: group events at  \lambda(t) = \lambda_{peak} ' ...
    'e^{-(t - t_{peak})^2 / 2\sigma^2} (N - A(t)) / N,  each bringing  G = 1 + Poisson(\mu)  patients' newline ...
    '\lambda_{peak} is set so that, on average, a patient arrives every p.mean\_interarrival minutes'], ...
    'Position', [x1('AcqTechnician') + 40, 36, x1('AcqTechnician') + 760, 78], 'HorizontalAlignment', 'left', ...
    'ForegroundColor', '[0.22 0.26 0.34]', 'BackgroundColor', 'automatic', 'FontSize', 11);
note('good', xa + 22, YG - 25);
note('borderline', X2(3) + BW + 4, YM - 10);
note('bad', xa + 22, YB - 10);
note('restored', X2(5) + BW - 8, YM - 50);
note('not restored', X2(6) + 40, YM + 5);
note('cleared', X2(10) + BW + 2, yClear - 10);
note('refer', xg + 6, YM + 80);
if gate, note('recapture: back to the waiting room', 560, YL - 10); end

% ---- model settings ------------------------------------------------------------------------
set_param(mdl, 'StopTime', 'p.sim_stop');
set_param(mdl, 'PostLoadFcn', strjoin({
    'dm_dir = fileparts(get_param(bdroot, ''FileName''));'
    'if ~contains(path, dm_dir), addpath(dm_dir); end, clear dm_dir'
    'if ~exist(''p'', ''var'') || ~isstruct(p) || ~isfield(p, ''n_edge_devices'')'
    '    [p, PatientBus] = district_model_params();'
    'elseif ~exist(''PatientBus'', ''var'')'
    '    [~, PatientBus] = district_model_params();'
    'end'}, newline));
set_param(mdl, 'InitFcn', 'district_model_check(bdroot);');
set_param(mdl, 'Description', ['District DR screening: group arrivals, capture, AI quality check ' ...
    '(good / borderline / bad), restoration, AI grading, and doctor review. Variables: ' ...
    'district_model_params.m. Built by build_district_model.m.']);
save_system(mdl, file);
fprintf('built %s (include_gate = %d, add_scopes = %d)\n', file, gate, scopes);
end

% ---- the blocks' MATLAB actions -----------------------------------------------------------
% Each action that draws random numbers seeds its own stream once per run,
% from p.seed plus its own offset: SimEvents restarts every action's stream
% from MATLAB's default seed at the start of each simulation, so without
% this every run would repeat the same draws.

function s = ARRIVALS()
s = strjoin({
'% Group arrivals: events at rate'
'%   lambda(t) = lambda_peak * exp(-(t - t_peak)^2 / (2 sigma^2)) * (N - A(t)) / N'
'% (A(t): patients so far), each bringing G = 1 + Poisson(mu) patients.'
'% Drawn by thinning: candidates at lambda_peak, each kept with'
'% probability lambda(t) / lambda_peak.'
'persistent seeded tclock remaining arrived lam'
'if isempty(seeded)'
'    rng(p.seed * 100 + 1, ''twister''); seeded = 1;'
'    tclock = 0; remaining = 0; arrived = 0;'
'    lam = district_peak_rate(p);          % one patient every p.mean_interarrival min, on average'
'end'
'if remaining > 0'
'    % the rest of a group arrives at the same moment'
'    remaining = remaining - 1;'
'    arrived = arrived + 1;'
'    dt = 0;'
'else'
'    tc = tclock;'
'    dt = 1e9;                             % no more arrivals this session'
'    while arrived < p.N && tc < p.sim_stop'
'        tc = tc - log(rand) / lam;        % a candidate at the peak rate'
'        rate = lam * exp(-(tc - p.t_peak)^2 / (2 * p.sigma^2)) * (p.N - arrived) / p.N;'
'        if rand * lam <= rate'
'            k = 0; L = exp(-p.mu); u = rand;   % group size - 1 ~ Poisson(mu)'
'            while u > L'
'                k = k + 1; u = u * rand;'
'            end'
'            remaining = min(k, p.N - arrived - 1);'
'            arrived = arrived + 1;'
'            dt = tc - tclock;'
'            break'
'        end'
'    end'
'end'
'tclock = tclock + dt;'}, newline);
end

function s = GENERATE()
s = strjoin({
'persistent seeded'
'if isempty(seeded), rng(p.seed * 100 + 2, ''twister''); seeded = 1; end'
'entity.true_status   = double(rand < p.prevalence_referable);'
'entity.quality_class = 0;'
'entity.restored      = 0;'
'entity.ungradable    = 0;'
'entity.attempts      = 0;'
'entity.flagged       = 0;'
'entity.route         = 1;'}, newline);
end

function s = BAD_IMAGE()
s = strjoin({
'    % another capture if the gate is in and captures remain,'
'    % otherwise to the doctor ungraded'
'    if p.include_gate && entity.attempts < p.max_attempts'
'        entity.route = 3;'
'    else'
'        entity.ungradable = 1; entity.flagged = 1; entity.route = 2;'
'    end'}, newline);
end

function s = QUALITY()
s = strjoin({
'% the AI sorts the image: 1 good, 2 borderline, 3 bad'
'persistent seeded'
'if isempty(seeded), rng(p.seed * 100 + 3, ''twister''); seeded = 1; end'
'r = rand * (p.p_pass + p.p_repair + p.p_reject);'
'if r < p.p_pass'
'    entity.quality_class = 1;'
'elseif r < p.p_pass + p.p_repair'
'    entity.quality_class = 2;'
'else'
'    entity.quality_class = 3;'
BAD_IMAGE()
'end'}, newline);
end

function s = RESTORE()
s = strjoin({
'% restoration saves p_restore of borderline images'
'persistent seeded'
'if isempty(seeded), rng(p.seed * 100 + 4, ''twister''); seeded = 1; end'
'if rand < p.p_restore'
'    entity.restored = 1;'
'else'
'    entity.restored = 2;'
BAD_IMAGE()
'end'}, newline);
end

function s = GRADE()
s = strjoin({
'% the AI grades a gradeable image'
'persistent seeded'
'if isempty(seeded), rng(p.seed * 100 + 5, ''twister''); seeded = 1; end'
'if entity.true_status == 1'
'    entity.flagged = double(rand < p.sensitivity);'
'else'
'    entity.flagged = double(rand > p.specificity);'
'end'
'entity.route = entity.flagged + 1;       % 1 cleared on the spot, 2 to the doctor'}, newline);
end
