% Compare the SimEvents model against the SimPy reference results.
%
% Two implementations of one model are only useful if they agree. This script
% runs the Simulink model and checks its headline outputs against the JSON
% written by drscreen (scripts/run_simulation.py --export-matlab).
%
%   >> validate_against_simpy('../outputs/simulation/results.json')

function validate_against_simpy(jsonPath)
if nargin < 1, jsonPath = '../outputs/simulation/results.json'; end
ref = jsondecode(fileread(jsonPath));

build_dr_screening_model('dr_screening');
out = sim('dr_screening');

sim_throughput = numel(out.throughput.Data) / ref.config.sim_days * ...
                 ref.config.working_days_per_year;

fprintf('\n%-28s %14s %14s %10s\n', 'metric', 'SimPy', 'SimEvents', 'rel.diff');
fprintf('%s\n', repmat('-', 1, 70));
compare('throughput / year', ref.throughput_per_year, sim_throughput);

    function compare(name, a, b)
        rel = abs(a - b) / max(abs(a), eps);
        flag = '';
        if rel > 0.10, flag = '  <-- CHECK'; end
        fprintf('%-28s %14.1f %14.1f %9.1f%%%s\n', name, a, b, rel*100, flag);
    end
end
