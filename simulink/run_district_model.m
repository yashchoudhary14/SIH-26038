function R = run_district_model(p, nReps, outJson)
%RUN_DISTRICT_MODEL  Replicate district_model.slx and summarise what it reports.
%
%   R = run_district_model()                  % the parameters as set, 100 runs
%   R = run_district_model(p, 50)             % your own p
%   R = run_district_model(p, 100, 'out.json')
%
%   Run r uses p.seed + r - 1, so the runs are independent and any one can be
%   reproduced by setting p.seed. If p.include_gate differs from how
%   district_model.slx was built, a matching copy is built in tempdir and
%   run instead; the saved model is never changed.
%
%   Logged per run, from each block's statistics port: patients arrived,
%   borderline images restored (attempts), patients cleared on the spot,
%   referred images uploaded, patients reviewed, and the utilisation of the
%   technicians, the edge devices and the ophthalmologists, as the resource
%   pools report it (averaged up to each pool's last event).

here = fileparts(mfilename('fullpath'));
[p0, PatientBus] = district_model_params();
if nargin < 1 || isempty(p), p = p0; end
if nargin < 2 || isempty(nReps), nReps = 100; end
if nargin < 3, outJson = ''; end

mdl = 'district_model';
file = fullfile(here, [mdl '.slx']);
if bdIsLoaded(mdl), close_system(mdl, 0); end
load_system(file);
if (numel(get_param([mdl '/RouteDecision'], 'PortHandles').Outport) >= 3) ~= logical(p.include_gate)
    close_system(mdl, 0);
    mdl = 'district_model_variant';
    file = fullfile(tempdir, [mdl '.slx']);
    q = p; q.add_scopes = 0;
    build_district_model(q, file);
    load_system(file);
    p.add_scopes = 0;
end
assignin('base', 'p', p); assignin('base', 'PatientBus', PatientBus);
cleanup = onCleanup(@() close_system(mdl, 0));
m = [mdl '/'];

logs = {'Arrivals',         'NumberEntitiesDeparted', 'arrived'; ...
        'Restoration',      'NumberEntitiesDeparted', 'restorations'; ...
        'ClearedOnSpot',    'NumberEntitiesArrived',  'cleared'; ...
        'Upload',           'NumberEntitiesDeparted', 'uploaded'; ...
        'Reviewed',         'NumberEntitiesArrived',  'reviewed'; ...
        'PoolTechnician',   'AverageUtilization',     'util_technician'; ...
        'PoolEdgeDevice',   'AverageUtilization',     'util_device'; ...
        'PoolOphthalmologist', 'AverageUtilization',  'util_ophthalmologist'};
for i = 1:size(logs, 1)
    b = [m logs{i, 1}];
    set_param(b, logs{i, 2}, 'on');
    % the statistics port: a pool's only port, or the one on a block's top edge
    ph = get_param(b, 'PortHandles'); pos = get_param(b, 'Position');
    top = ph.Outport(arrayfun(@(h) subsref(get_param(h, 'Position'), substruct('()', {2})) <= pos(2) + 1, ph.Outport));
    if strcmp(get_param(b, 'BlockType'), 'EntityResourcePool'), top = ph.Outport; end
    set_param(top(1), 'DataLogging', 'on', 'DataLoggingNameMode', 'Custom', 'DataLoggingName', logs{i, 3});
end
set_param(mdl, 'SignalLogging', 'on', 'SignalLoggingName', 'logsout', 'ReturnWorkspaceOutputs', 'on');

names = logs(:, 3)';
V = nan(nReps, numel(names));
seed0 = p.seed;
for r = 1:nReps
    p.seed = seed0 + r - 1;
    assignin('base', 'p', p);
    out = sim(mdl);
    L = out.logsout;
    for k = 1:numel(names)
        el = L.getElement(names{k});
        if isempty(el), V(r, k) = 0; continue; end    % the block never reported
        d = el.Values.Data;
        V(r, k) = double(d(end));
    end
end
p.seed = seed0;

R = struct('model', 'district_model', 'stop_min', p.sim_stop, 'reps', nReps, 'p', p);
for k = 1:numel(names)
    col = V(:, k);
    R.(names{k}) = struct('mean', mean(col, 'omitnan'), 'sd', std(col, 'omitnan'), 'runs', col');
end
if ~isempty(outJson)
    fid = fopen(outJson, 'w'); fwrite(fid, jsonencode(R, 'PrettyPrint', true)); fclose(fid);
end
fprintf('district_model (include_gate = %d), %d runs of %g min\n', p.include_gate, nReps, p.sim_stop);
for k = 1:numel(names)
    fprintf('  %-13s %8.3f  (sd %.3f)\n', names{k}, R.(names{k}).mean, R.(names{k}).sd);
end
end
