classdef Segmenter < handle
    %SEGMENTER Attention U-Net for the five lesion classes, from exported weights.
    %
    %   s = drscreen.models.Segmenter(modelsDir)
    %   probs = s.segment(modelInput512)        % 512 x 512 x 5, in [0, 1]
    %
    %   Runs at the resolution it was trained at (1024): a microaneurysm is a
    %   few pixels wide, and at 512 the lesion head scored Dice 0.00 on them
    %   against 0.485 at 1024. The grader still works at 512; only this stage
    %   pays for the larger frame. Attention gates on the skip connections
    %   suppress the large avascular background so it stops injecting texture
    %   into the finest decoder stage.
    %
    %   Plain matrix operations throughout (drscreen.nn.*); transposed
    %   convolutions and max-pooling are written out directly.

    properties
        W
        Spec
        Mean
        Std
    end

    properties (Dependent)
        SupervisedClasses
        Size
    end

    methods
        function obj = Segmenter(modelsDir)
            obj.W = load(fullfile(modelsDir, 'segmentation.mat'));
            obj.Spec = jsondecode(fileread(fullfile(modelsDir, 'segmentation_spec.json')));
            C = drscreen.constants();
            obj.Mean = reshape(C.IMAGENET_MEAN, 1, 1, 3);
            obj.Std = reshape(C.IMAGENET_STD, 1, 1, 3);
        end

        function v = get.SupervisedClasses(obj)
            v = cellstr(obj.Spec.supervised_lesion_classes)';
        end

        function v = get.Size(obj)
            v = obj.Spec.size;
        end

        function logits = forward(obj, X)
            %FORWARD Normalised H x W x 3 single -> H x W x 5 logits.
            W = obj.W;
            sp = obj.Spec;
            gnEps = sp.gn_eps;
            n = numel(sp.widths);
            skips = cell(1, n - 1);
            x = X;
            for i = 0:n-1
                x = convBlock(x, W, sprintf('e%d', i), sp.encoder_groups(i + 1), gnEps);
                if i < n - 1
                    skips{i + 1} = x;
                    x = maxPool2(x);
                end
            end
            for i = 0:n-2
                x = transConv2x2(x, W.(sprintf('u%d_w', i)), W.(sprintf('u%d_b', i)));
                skip = skips{n - 1 - i};
                g = drscreen.nn.conv2d(x, W.(sprintf('a%d_gw', i)), W.(sprintf('a%d_gb', i)));
                s = drscreen.nn.conv2d(skip, W.(sprintf('a%d_xw', i)), W.(sprintf('a%d_xb', i)));
                psi = drscreen.nn.sigmoid(drscreen.nn.conv2d(drscreen.nn.silu(g + s), ...
                        W.(sprintf('a%d_pw', i)), W.(sprintf('a%d_pb', i))));
                skip = skip .* psi;
                x = convBlock(cat(3, x, skip), W, sprintf('d%d', i), sp.decoder_groups(i + 1), gnEps);
                x = drscreen.nn.squeezeExcite(x, W.(sprintf('s%d_rw', i)), W.(sprintf('s%d_rb', i)), ...
                                              W.(sprintf('s%d_ew', i)), W.(sprintf('s%d_eb', i)));
            end
            logits = drscreen.nn.conv2d(x, W.head_w, W.head_b);
        end

        function probs = segment(obj, modelInput, outSize)
            %SEGMENT Lesion probability maps at the grader's resolution.
            if nargin < 3, outSize = size(modelInput, 1); end
            segSize = obj.Size;
            img = modelInput;
            if size(img, 1) ~= segSize
                if size(img, 1) < segSize
                    img = drscreen.cv.resize(img, [segSize segSize], 'cubic');
                else
                    img = drscreen.cv.resize(img, [segSize segSize], 'area');
                end
            end
            X = (single(img) / 255 - obj.Mean) ./ obj.Std;
            probs = drscreen.nn.sigmoid(obj.forward(X));
            if size(probs, 1) ~= outSize
                probs = drscreen.cv.resize(probs, [outSize outSize], 'area');
            end
            probs = gather(probs);
        end
    end
end


function x = convBlock(x, W, p, groups, gnEps)
% conv3x3 -> GroupNorm -> SiLU, twice
x = drscreen.nn.conv2d(x, W.([p '_c1']));
x = drscreen.nn.silu(drscreen.nn.groupNorm(x, groups, W.([p '_g1g']), W.([p '_g1b']), gnEps));
x = drscreen.nn.conv2d(x, W.([p '_c2']));
x = drscreen.nn.silu(drscreen.nn.groupNorm(x, groups, W.([p '_g2g']), W.([p '_g2b']), gnEps));
end


function y = maxPool2(x)
% 2x2 max pooling, stride 2
[H, Wd, C] = size(x);
y = reshape(x, 2, H / 2, 2, Wd / 2, C);
y = max(max(y, [], 1), [], 3);
y = reshape(y, H / 2, Wd / 2, C);
end


function y = transConv2x2(x, Wt, b)
% ConvTranspose2d(kernel 2, stride 2): each input pixel writes a 2x2 block.
% Wt is [2 2 O C]; out(2i+di, 2j+dj, o) = sum_c x(i, j, c) * Wt(di, dj, o, c).
[H, Wd, C] = size(x);
O = size(Wt, 3);
Xm = reshape(x, H * Wd, C);
y = zeros(2 * H, 2 * Wd, O, 'like', x);
for di = 0:1
    for dj = 0:1
        Wm = reshape(Wt(di + 1, dj + 1, :, :), O, C);
        y(1 + di:2:end, 1 + dj:2:end, :) = reshape(Xm * Wm', H, Wd, O);
    end
end
y = y + reshape(b, 1, 1, []);
end
