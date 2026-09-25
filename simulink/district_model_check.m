function district_model_check(mdl)
%DISTRICT_MODEL_CHECK  Run by district_model.slx before every simulation
%(InitFcn). Makes sure the workspace variables exist and agree with the
%model's structure, and stops with a plain message if they do not.

if nargin < 1, mdl = bdroot; end
if ~evalin('base', 'exist(''p'', ''var'') && isstruct(p) && isfield(p, ''n_edge_devices'')')
    [p, PatientBus] = district_model_params(); %#ok<ASGLU>
    assignin('base', 'p', p); assignin('base', 'PatientBus', PatientBus);
end
if ~evalin('base', 'exist(''PatientBus'', ''var'')')
    [~, PatientBus] = district_model_params();
    assignin('base', 'PatientBus', PatientBus);
end
p = evalin('base', 'p');

% the structural switches must match what was built
built = numel(get_param([mdl '/RouteDecision'], 'PortHandles').Outport) >= 3;
if logical(p.include_gate) ~= built
    error('district_model:rebuild', ['p.include_gate is %d but the model was built with include_gate = %d. ' ...
        'Run build_district_model(p) to rebuild it.'], p.include_gate, built);
end
hasScopes = ~isempty(find_system(mdl, 'SearchDepth', 1, 'BlockType', 'Scope'));
if logical(p.add_scopes) ~= hasScopes
    warning('district_model:scopes', ['p.add_scopes is %d but the model was built with add_scopes = %d; ' ...
        'run build_district_model(p) to add or remove them.'], p.add_scopes, hasScopes);
end

% the numbers must make sense
target = p.sim_stop / p.mean_interarrival;
if target >= p.N
    error('district_model:catchment', ['A patient every %g min over %g min is %.0f patients, but the catchment ' ...
        'p.N is only %g. Raise p.N above %.0f.'], p.mean_interarrival, p.sim_stop, target, p.N, target);
end
shares = [p.p_pass, p.p_repair, p.p_reject];
if any(shares < 0) || sum(shares) <= 0
    error('district_model:quality', 'p.p_pass, p.p_repair and p.p_reject must be non-negative shares.');
end
if abs(sum(shares) - 1) > 1e-6
    warning('district_model:quality', 'p.p_pass + p.p_repair + p.p_reject = %.3f; the model uses them in proportion.', sum(shares));
end
end
