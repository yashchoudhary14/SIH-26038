function lam = district_peak_rate(p)
%DISTRICT_PEAK_RATE  The peak arrival-event rate lambda_peak (events per minute)
%that makes a session of p.sim_stop minutes bring, on average, one patient
%every p.mean_interarrival minutes.
%
%   With lambda(t) = lambda_peak * g(t) * (N - A)/N, g the Gaussian bump, and
%   1 + mu patients per event, the expected arrivals obey
%   dE[A]/dt = (1 + mu) * lambda_peak * g(t) * (N - E[A]) / N, so
%
%       E[A(T)] = N * (1 - exp(-(1 + mu) * lambda_peak * Gint(T) / N))
%       Gint(T) = sigma * sqrt(2 pi) * (Phi((T - t_peak)/sigma) - Phi(-t_peak/sigma))
%
%   (exact apart from the cap on the last group). Setting E[A(T)] to
%   T / mean_interarrival and solving gives lambda_peak. The Arrivals block
%   calls this once per run, so editing p takes effect immediately.
%#codegen
target = p.sim_stop / p.mean_interarrival;
Phi = @(x) 0.5 * erfc(-x / sqrt(2));
Gint = p.sigma * sqrt(2 * pi) * (Phi((p.sim_stop - p.t_peak) / p.sigma) - Phi(-p.t_peak / p.sigma));
lam = -p.N * log(1 - min(target / p.N, 0.999)) / ((1 + p.mu) * Gint);
end
