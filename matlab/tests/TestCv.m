classdef TestCv < matlab.unittest.TestCase
    %TESTCV Each +drscreen/+cv primitive against OpenCV's own output.
    %
    %   The fixtures were produced by cv2 itself (tools/make_fixtures.py). The
    %   tolerances are what tools/verify_cv_algorithms.py measured for the same
    %   algorithms in Python: most are bit-exact, so a failure here is a
    %   MATLAB translation error, not an algorithm difference.

    properties
        P
    end

    methods (TestClassSetup)
        function loadFixtures(tc)
            f = fullfile(drscreen.paths().tests, 'fixtures', 'primitives.mat');
            tc.assumeTrue(isfile(f), 'primitives.mat missing: run tools/make_fixtures.py');
            tc.P = load(f);
        end
    end

    methods (Test)
        function greyIsBitExact(tc)
            tc.verifyEqual(drscreen.cv.rgb2gray(tc.P.small_rgb), tc.P.gray);
        end

        function rgbToLabIsBitExact(tc)
            tc.verifyEqual(drscreen.cv.rgb2lab(tc.P.small_rgb), tc.P.lab, 'photograph');
            tc.verifyEqual(drscreen.cv.rgb2lab(tc.P.rgb_rand), tc.P.rgb_rand_lab, '262k random colours');
        end

        function labToRgbIsBitExact(tc)
            tc.verifyEqual(drscreen.cv.lab2rgb(tc.P.lab), tc.P.lab2rgb, 'photograph');
            tc.verifyEqual(drscreen.cv.lab2rgb(tc.P.lab_rand), tc.P.lab_rand_rgb, '262k random Lab values');
        end

        function blur8BitIsBitExact(tc)
            s = tc.P.blur_u8_sigmas;
            for i = 1:numel(s)
                got = drscreen.cv.gaussianBlur(tc.P.gray, s(i));
                tc.verifyEqual(got, tc.P.(sprintf('blur_u8_%d', i - 1)), ...
                    sprintf('uint8 blur, sigma %.1f', s(i)));
            end
            got = drscreen.cv.gaussianBlur(tc.P.small_rgb, 0.033 * size(tc.P.small_rgb, 2));
            tc.verifyEqual(got, tc.P.blur_rgb, '3-channel uint8 blur');
        end

        function blurFloatMatches(tc)
            s = tc.P.blur_f_sigmas;
            for i = 1:numel(s)
                got = drscreen.cv.gaussianBlur(single(tc.P.gray), s(i));
                ref = tc.P.(sprintf('blur_f_%d', i - 1));
                tc.verifyLessThan(max(abs(double(got(:)) - double(ref(:)))), 1e-3, ...
                    sprintf('float blur, sigma %.1f', s(i)));
            end
        end

        function resizeArea(tc)
            tc.verifyEqual(drscreen.cv.resize(tc.P.gray, [137 151], 'area'), tc.P.resize_area_down);
            tc.verifyEqual(drscreen.cv.resize(tc.P.src_rgb, [512 512], 'area'), tc.P.resize_area_std);
            got = drscreen.cv.resize(single(tc.P.gray), [151 142], 'area');
            tc.verifyLessThan(max(abs(double(got(:)) - double(tc.P.resize_area_half(:)))), 1e-3);
            got = drscreen.cv.resize(tc.P.float_probe, [32 32], 'area');
            tc.verifyLessThan(max(abs(double(got(:)) - double(tc.P.resize_area_float2x(:)))), 1e-5);
        end

        function resizeCubicAndNearest(tc)
            tc.verifyEqual(drscreen.cv.resize(tc.P.std_rgb, [1024 1024], 'cubic'), tc.P.resize_cubic_2x, ...
                'the U-Net input is this exact upsampling');
            got = drscreen.cv.resize(tc.P.gray, [600 640], 'cubic');
            tc.verifyLessThanOrEqual(max(abs(double(got(:)) - double(tc.P.resize_cubic_up(:)))), 1);
            tc.verifyEqual(drscreen.cv.resize(tc.P.gray, [97 400], 'nearest'), tc.P.resize_nearest);
        end

        function claheIsBitExact(tc)
            tc.verifyEqual(drscreen.cv.clahe(tc.P.std_rgb(:, :, 2), 3.0, 8), tc.P.clahe_green_3);
            tc.verifyEqual(drscreen.cv.clahe(tc.P.gray, 2.5, 8), tc.P.clahe_odd_25, ...
                'non-divisible size exercises the reflect-101 padding quirk');
        end

        function otsuThreshold(tc)
            tc.verifyEqual(drscreen.cv.otsu(tc.P.otsu_in), double(tc.P.otsu_t));
        end

        function ellipseElements(tc)
            for k = tc.P.ellipse_ks(:)'
                tc.verifyEqual(double(drscreen.cv.ellipse(k)), double(tc.P.(sprintf('ellipse_%d', k))), ...
                    sprintf('ellipse %d', k));
            end
        end

        function morphology(tc)
            g = single(tc.P.std_rgb(:, :, 2));
            tc.verifyEqual(drscreen.cv.morph(g, 'close', 13), tc.P.close_f13);
            tc.verifyEqual(drscreen.cv.morph(g, 'close', 15), tc.P.close_f15);
            tc.verifyEqual(drscreen.cv.morph(tc.P.bin, 'open', 5), tc.P.open_bin5);
            tc.verifyEqual(drscreen.cv.morph(tc.P.std_mask, 'erode', 31), tc.P.erode_mask31);
        end

        function filterAndDistance(tc)
            got = drscreen.cv.filter2d(single(tc.P.std_rgb(:, :, 2)), [1 -2 1; -2 4 -2; 1 -2 1]);
            tc.verifyLessThan(max(abs(got(:) - double(tc.P.filter_noise(:)))), 1e-3);
            got = drscreen.cv.distanceTransform(tc.P.dt_in);
            tc.verifyLessThan(max(abs(double(got(:)) - double(tc.P.dt_out(:)))), 1e-4);
        end

        function numpyPercentile(tc)
            got = drscreen.cv.percentile(tc.P.pct_in, tc.P.pct_p);
            tc.verifyLessThan(max(abs(got(:) - tc.P.pct_out(:))), 1e-12);
        end

        function connectedComponents(tc)
            S = drscreen.cv.conncomp(tc.P.bin > 0);
            c = reshape([S.Centroid], 2, [])';
            [~, order] = sortrows([c(:, 2), c(:, 1)]);
            tc.verifyEqual(double([S(order).Area]'), tc.P.cc_area(:));
            tc.verifyLessThan(max(max(abs(c(order, :) - tc.P.cc_cent))), 1e-9);
        end

        function benGrahamPlane(tc)
            g = tc.P.small_rgb(:, :, 2);
            b = drscreen.cv.gaussianBlur(g, double(tc.P.benG_sigma));
            got = drscreen.cv.saturateU8(4 * double(g) - 4 * double(b) + 128);
            tc.verifyEqual(got, tc.P.benG, 'the second grader plane amplifies blur error x4');
        end
    end
end
