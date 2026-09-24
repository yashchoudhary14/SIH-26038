classdef Grader < handle
    %GRADER EfficientNet-B0 + CORN ordinal head, rebuilt from exported weights.
    %
    %   g = drscreen.models.Grader(bundleDir)
    %   out = g.predict(modelInput, mcSamples, temperature, gradeThresholds)
    %   cam = g.gradCamPP(modelInput, fovMask)
    %
    %   The backbone is tf_efficientnet_b0 as timm builds it: a stride-2 stem,
    %   sixteen MBConv blocks with squeeze-excitation, a 1x1 head conv. It is
    %   evaluated with plain matrix operations (drscreen.nn.*), so no Deep
    %   Learning Toolbox is needed. tools/export_to_matlab.py re-ran this exact
    %   computation from the exported arrays and matched PyTorch to 4e-6 in
    %   the logits.
    %
    %   Why CORN and not softmax: DR grades are ordered and the screening
    %   decision is cumulative ("is this grade >= 2?"). CORN predicts the
    %   conditionals P(y > k | y > k-1), so P(y > k) = prod_{j<=k} sigmoid(z_j)
    %   is monotone by construction, and the referable probability is a number
    %   the network optimised rather than a sum of unordered class scores.
    %
    %   The backbone has no dropout, so MC-dropout sampling runs the backbone
    %   once and resamples only the head -- the same distribution the Python
    %   pipeline draws from by re-running the whole network eight times.

    properties
        W          % exported weights (struct of single arrays)
        Spec       % architecture description (grader_spec.json)
        Mean       % ImageNet mean, 1x1x3
        Std        % ImageNet std, 1x1x3
    end

    methods
        function obj = Grader(bundleDir)
            obj.W = load(fullfile(bundleDir, 'grader.mat'));
            obj.Spec = jsondecode(fileread(fullfile(bundleDir, 'grader_spec.json')));
            obj.Mean = reshape(single(obj.Spec.normalization.mean), 1, 1, 3);
            obj.Std = reshape(single(obj.Spec.normalization.std), 1, 1, 3);
        end

        function X = normalize(obj, modelInput)
            % to_tensor: /255 then ImageNet normalisation, plane by plane.
            X = (single(modelInput) / 255 - obj.Mean) ./ obj.Std;
        end

        function [A, taps] = trunk(obj, X, tapAfter)
            %TRUNK Backbone up to the 1x1 head conv (the Grad-CAM layer).
            %   [A, taps] = trunk(X, tapAfter) also returns the activation after
            %   each listed block (0 = after the stem), for stage-by-stage tests.
            if nargin < 3, tapAfter = []; end
            taps = {};
            W = obj.W;
            sp = obj.Spec;
            X = drscreen.nn.conv2d(X, W.stem_w, [], sp.stem.stride);
            X = act(X .* W.stem_s + W.stem_t, sp.stem.act);
            if ismember(0, tapAfter), taps{end + 1} = X; end
            blocks = sp.blocks;
            for i = 1:numel(blocks)
                if iscell(blocks), b = blocks{i}; else, b = blocks(i); end
                p = b.name;
                acts = cellstr(b.acts);
                inp = X;
                if strcmp(b.type, 'ds')
                    X = drscreen.nn.conv2d(X, W.([p '_dw_w']), [], b.stride);
                    X = act(X .* W.([p '_bn1_s']) + W.([p '_bn1_t']), acts{1});
                    X = se(X, W, p);
                    X = drscreen.nn.conv2d(X, W.([p '_pw_w']));
                    X = act(X .* W.([p '_bn2_s']) + W.([p '_bn2_t']), acts{2});
                else
                    X = drscreen.nn.conv2d(X, W.([p '_pw_w']));
                    X = act(X .* W.([p '_bn1_s']) + W.([p '_bn1_t']), acts{1});
                    X = drscreen.nn.conv2d(X, W.([p '_dw_w']), [], b.stride);
                    X = act(X .* W.([p '_bn2_s']) + W.([p '_bn2_t']), acts{2});
                    X = se(X, W, p);
                    X = drscreen.nn.conv2d(X, W.([p '_pwl_w']));
                    X = act(X .* W.([p '_bn3_s']) + W.([p '_bn3_t']), acts{3});
                end
                if b.skip
                    X = X + inp;
                end
                if ismember(i, tapAfter), taps{end + 1} = X; end
            end
            A = drscreen.nn.conv2d(X, W.headconv_w);
        end

        function [z, c] = head(obj, A, nSamples, stochastic)
            %HEAD bn2 + SiLU + global pool + LayerNorm MLP -> CORN logits (4 x S).
            if nargin < 3, nSamples = 1; end
            if nargin < 4, stochastic = false; end
            W = obj.W;
            m = obj.Spec.mlp;
            a1 = A .* W.headconv_s + W.headconv_t;
            a2 = drscreen.nn.silu(a1);
            f = reshape(mean(a2, [1 2]), [], 1);
            F = repmat(f, 1, nSamples);
            if stochastic && m.feature_dropout > 0
                keep = rand(size(F), 'like', F) >= m.feature_dropout;
                F = F .* keep / (1 - m.feature_dropout);
            end
            N = size(F, 1);
            mu = sum(F, 1) / N;
            D = F - mu;
            v = sum(D .^ 2, 1) / N;
            rstd = 1 ./ sqrt(v + m.layernorm_eps);
            xhat = D .* rstd;
            n = xhat .* W.ln_g(:) + W.ln_b(:);
            u = W.fc1_w * n + W.fc1_b(:);
            h = drscreen.nn.silu(u);
            if stochastic && m.hidden_dropout > 0
                keep = rand(size(h), 'like', h) >= m.hidden_dropout;
                h = h .* keep / (1 - m.hidden_dropout);
            end
            z = W.fc2_w * h + W.fc2_b(:);
            if nargout > 1
                c = struct('a1', a1, 'xhat', xhat, 'rstd', rstd, 'u', u, 'N', N);
            end
        end

        function z = logits(obj, modelInput)
            z = obj.head(obj.trunk(obj.normalize(modelInput)));
        end

        function out = predict(obj, modelInput, mcSamples, temperature, gradeThresholds)
            %PREDICT Grade, class probabilities, referable probability, uncertainty.
            %
            %   The grade comes from the ordinal first-failure rule over the
            %   cumulative probabilities -- the rule every reported metric was
            %   computed with -- not from argmax over class probabilities.
            if nargin < 3 || isempty(mcSamples), mcSamples = 0; end
            if nargin < 4 || isempty(temperature), temperature = 1; end
            if nargin < 5 || isempty(gradeThresholds), gradeThresholds = 0.5; end

            A = obj.trunk(obj.normalize(modelInput));
            if mcSamples > 0
                z = obj.head(A, mcSamples, true) / temperature;
                P = drscreen.models.cornClassProbs(double(z));   % 5 x S
                p = mean(P, 2);
                epistemic = sum(var(P, 0, 2));
            else
                z = obj.head(A) / temperature;
                p = drscreen.models.cornClassProbs(double(z));
                epistemic = 0;
            end
            cum = drscreen.models.cumulativeFromClassProbs(p);
            K = numel(p);
            out = struct( ...
                'class_probs', p, ...
                'grade', drscreen.models.gradeFromCumulative(cum, gradeThresholds), ...
                'expected_grade', sum(p(:) .* (0:K-1)'), ...
                'referable_prob', sum(p(3:end)), ...
                'entropy', -sum(log(max(p, 1e-9)) .* p), ...
                'epistemic', epistemic, ...
                'activation', A);
        end

        function cam = gradCamPP(obj, modelInput, fovMask, A)
            %GRADCAMPP Grad-CAM++ over the log-odds of P(grade >= 2).
            %
            %   Differentiates the quantity the referral threshold is applied to,
            %   log p - log(1 - p) with p = sigmoid(z1) * sigmoid(z2), with
            %   respect to the 1x1 head-conv output -- the same tensor and target
            %   as the Python explain.cam module. The gradient flows back only
            %   through the small head, so it is written out analytically rather
            %   than taken from an autodiff framework.
            if nargin < 4 || isempty(A)
                A = obj.trunk(obj.normalize(modelInput));
            end
            [z, c] = obj.head(A);
            z = double(z);

            logp = -log1p(exp(-z(1))) - log1p(exp(-z(2)));
            p = exp(logp);
            dScore = 1;
            if p < 1 - 1e-6
                dScore = 1 + p / (1 - p);
            end
            dz = zeros(size(z));
            dz(1) = dScore * drscreen.nn.sigmoid(-z(1));
            dz(2) = dScore * drscreen.nn.sigmoid(-z(2));

            W = obj.W;
            dh = double(W.fc2_w)' * dz;
            su = drscreen.nn.sigmoid(double(c.u));
            du = dh .* su .* (1 + double(c.u) .* (1 - su));
            dn = double(W.fc1_w)' * du;
            dxhat = dn .* double(W.ln_g(:));
            xhat = double(c.xhat);
            N = c.N;
            df = double(c.rstd) / N * (N * dxhat - sum(dxhat) - xhat * sum(dxhat .* xhat));
            [h, w, C] = size(A);
            da2 = reshape(df, 1, 1, C) / (h * w);
            sa = drscreen.nn.sigmoid(double(c.a1));
            dA = da2 .* sa .* (1 + double(c.a1) .* (1 - sa)) .* double(W.headconv_s);

            acts = double(A);
            g2 = dA .^ 2;
            g3 = dA .^ 3;
            sumA = sum(acts, [1 2]);
            denom = 2 * g2 + sumA .* g3;
            denom(denom == 0) = 1;
            alpha = g2 ./ denom;
            weights = sum(alpha .* max(dA, 0), [1 2]);
            m = max(sum(weights .* acts, 3), 0);

            sz = size(modelInput);
            m = drscreen.cv.resize(m, sz(1:2), 'linear');
            if nargin >= 3 && ~isempty(fovMask)
                m(fovMask == 0) = 0;
            end
            lo = min(m(:)); hi = max(m(:));
            if hi > lo
                cam = single((m - lo) / (hi - lo));
            else
                cam = zeros(size(m), 'single');
            end
        end
    end
end


function X = act(X, name)
if strcmp(name, 'silu')
    X = drscreen.nn.silu(X);
end
end


function X = se(X, W, p)
X = drscreen.nn.squeezeExcite(X, W.([p '_se_rw']), W.([p '_se_rb']), ...
                              W.([p '_se_ew']), W.([p '_se_eb']));
end
