% Verify dr_screening_params.m parses and its derived quantities are correct.
%
% Runs under GNU Octave as well as MATLAB, so the parameter half of the
% Simulink bridge can be checked without a MathWorks licence. It does NOT
% exercise SimEvents -- nothing can, outside MATLAB.
%
%   octave-cli --quiet --eval "run('verify_params.m')"
%   >> verify_params            % in MATLAB

p = dr_screening_params();
fail = 0;

printf_or_disp = @(s) disp(s);
disp('=== dr_screening_params.m ===');
disp(['fields                : ' num2str(numel(fieldnames(p)))]);
disp(['mean_interarrival_min : ' num2str(p.mean_interarrival_min, '%.4f')]);
disp(['payload_mb            : ' num2str(p.payload_mb, '%.2f')]);
disp(['capture mu/sigma      : ' num2str(p.capture_mu, '%.4f') ' / ' num2str(p.capture_sigma, '%.4f')]);
disp(['review  mu/sigma      : ' num2str(p.review_mu, '%.4f') ' / ' num2str(p.review_sigma, '%.4f')]);

% --- the lognormal conversion --------------------------------------------
% SimConfig stores mean and coefficient of variation; MATLAB's Lognormal
% block wants mu and sigma OF THE LOG. Getting this backwards is silent and
% would make every service time wrong, so it is checked by round-tripping.
m  = exp(p.capture_mu + p.capture_sigma^2/2);
cv = sqrt(exp(p.capture_sigma^2) - 1);
disp(['capture round-trip    : mean ' num2str(m, '%.4f') ' (want ' num2str(p.capture_time_min, '%.4f') ...
      '), cv ' num2str(cv, '%.4f') ' (want ' num2str(p.capture_time_cv, '%.4f') ')']);
if abs(m - p.capture_time_min) > 1e-6 || abs(cv - p.capture_time_cv) > 1e-6
    disp('  FAIL: capture lognormal conversion'); fail = fail + 1;
end

m  = exp(p.review_mu + p.review_sigma^2/2);
cv = sqrt(exp(p.review_sigma^2) - 1);
if abs(m - p.review_time_min) > 1e-6 || abs(cv - p.review_time_cv) > 1e-6
    disp('  FAIL: review lognormal conversion'); fail = fail + 1;
end

% --- the link-state Markov chain ------------------------------------------
% Uptime is derived so the chain's stationary up-fraction equals the declared
% availability. If that identity breaks, the Simulink link model and the
% SimPy one disagree about how often the network is down.
stat = p.net_uptime_mean_min / (p.net_uptime_mean_min + p.net_outage_mean_min);
disp(['markov stationary up  : ' num2str(stat, '%.4f') ' (want ' num2str(p.net_availability, '%.4f') ')']);
if abs(stat - p.net_availability) > 1e-9
    disp('  FAIL: Markov uptime does not reproduce availability'); fail = fail + 1;
end

% --- arrival rate ----------------------------------------------------------
expected = (p.hours_per_day * 60) / (p.annual_patients / (p.n_phc * p.working_days_per_year));
if abs(p.mean_interarrival_min - expected) > 1e-9
    disp('  FAIL: mean_interarrival_min inconsistent with the demand figures');
    fail = fail + 1;
end

disp('');
if fail == 0
    disp('ALL PARAMETER CHECKS PASSED');
else
    disp(['FAILURES: ' num2str(fail)]);
end
