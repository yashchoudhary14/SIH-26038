classdef TestServer < matlab.unittest.TestCase
    %TESTSERVER The HTTP API: routes in-process, plus one real socket round trip.
    %
    %   Endpoint behaviour is tested through Server.request (no socket), because
    %   MATLAB is single-threaded: a client call from this process would block
    %   the thread that has to answer it. The socket layer is then exercised
    %   once with an external curl process while this process serves.

    properties
        S
    end

    properties (TestParameter)
        transport = {'dotnet', 'java'}
    end

    methods (TestClassSetup)
        function makeServer(tc)
            tc.S = drscreen.server.Server('Port', 8765, 'Verbose', false, ...
                'AuditLog', [tempname '_reviews.jsonl']);
        end
    end

    methods (TestClassTeardown)
        function stopServer(tc)
            tc.S.stop();
        end
    end

    methods (Test)
        function health(tc)
            j = jsonOf(tc.S.request('GET', '/health'));
            tc.verifyEqual(j.status, 'ok');
            tc.verifyTrue(j.grader_loaded && j.segmentation_loaded);
            tc.verifyEqual(j.image_size, 512);
            tc.verifyTrue(isfield(j.quality_thresholds, 'focus'));
        end

        function casesAndImages(tc)
            j = jsonOf(tc.S.request('GET', '/cases'));
            tc.verifyEqual(j.n, 12);
            r = tc.S.request('GET', '/cases/grade0_case1/image');
            tc.verifyEqual(r.status, 200);
            tc.verifyEqual(r.type, 'image/jpeg');
            r = tc.S.request('GET', '/cases/nope');
            tc.verifyEqual(r.status, 404);
        end

        function screenUploadedImage(tc)
            f = drscreen.samples.list();
            fid = fopen(f{1}, 'r'); data = fread(fid, Inf, '*uint8')'; fclose(fid);
            b = 'XyZBoundary123';
            body = [uint8(sprintf(['--%s\r\nContent-Disposition: form-data; name="file"; ' ...
                'filename="probe.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'], b)), data, ...
                uint8(sprintf('\r\n--%s--\r\n', b))];
            r = tc.S.request('POST', '/screen', body, ['multipart/form-data; boundary=' b]);
            tc.verifyEqual(r.status, 200);
            j = jsonOf(r);
            tc.verifyEqual(j.image_id, 'probe');
            tc.verifyTrue(isfield(j, 'panel_jpeg_b64') && numel(j.panel_jpeg_b64) > 1000);
            tc.verifyTrue(ismember(j.decision, {'auto_report', 'refer', 'defer_to_human', 'recapture'}));
        end

        function demoAndReview(tc)
            j = jsonOf(tc.S.request('GET', '/demo/3'));
            tc.verifyEqual(j.ground_truth.grade, 3);
            tc.verifyFalse(j.synthetic);
            r = tc.S.request('POST', '/review', ...
                uint8('image_id=test&model_grade=3&reviewer_grade=2&reviewer=unit&seconds=12'), ...
                'application/x-www-form-urlencoded');
            j = jsonOf(r);
            tc.verifyTrue(j.recorded);
            tc.verifyEqual(j.agreement, 'within_one');
            a = jsonOf(tc.S.request('GET', '/audit'));
            tc.verifyGreaterThanOrEqual(a.n, 1);
        end

        function staticAndRedirect(tc)
            r = tc.S.request('GET', '/');
            tc.verifyEqual(r.status, 302);
            r = tc.S.request('GET', '/web/index.html');
            tc.verifyEqual(r.status, 200);
            r = tc.S.request('GET', '/web/../../startup.m');
            tc.verifyEqual(r.status, 404, 'path traversal must be refused');
        end

        function realSocketRoundTrip(tc, transport)
            % The server polls on a MATLAB timer; pause() yields to it while
            % external curl processes make genuine HTTP requests: a health check
            % and a multipart photograph upload through /screen.
            tc.assumeTrue(ispc || strcmp(transport, 'java'), '.NET transport is Windows-only');
            curl = 'curl';
            if ispc, curl = 'curl.exe'; end
            [st, ~] = system([curl ' --version']);
            tc.assumeEqual(st, 0, 'curl not available for the socket test');
            port = 8766 + strcmp(transport, 'java');
            s = drscreen.server.Server('Port', port, 'Verbose', false, ...
                'Pipeline', tc.S.Pipeline, 'Transport', transport);
            cleaner = onCleanup(@() s.stop());
            s.start();
            base = sprintf('http://127.0.0.1:%d', port);

            j = curlJson({curl, '-s', '-m', '20', [base '/health']});
            tc.verifyEqual(j.status, 'ok');
            tc.verifyEqual(j.runtime, 'matlab');

            f = drscreen.samples.list();
            j = curlJson({curl, '-s', '-m', '120', '-F', ['file=@' f{1}], [base '/screen']});
            tc.verifyTrue(isfield(j, 'grade') && isfield(j, 'decision'), 'screening result over HTTP');
            tc.verifyTrue(ismember(j.decision, {'auto_report', 'refer', 'defer_to_human', 'recapture'}));
        end
    end
end


function j = jsonOf(resp)
j = jsondecode(native2unicode(resp.body, 'UTF-8'));
end


function j = curlJson(args)
% Run curl as a separate process (so this MATLAB thread stays free to serve)
% and decode what it wrote.
out = [tempname '.json'];
args = [args(1), {'-o', out}, args(2:end)];
cmd = javaArray('java.lang.String', numel(args));
for k = 1:numel(args)
    cmd(k) = java.lang.String(args{k});
end
proc = java.lang.Runtime.getRuntime().exec(cmd);
t0 = tic;
while proc.isAlive() && toc(t0) < 150
    pause(0.05);
end
assert(isfile(out), 'no response written by curl');
j = jsondecode(drscreen.io.readText(out));
delete(out);
end
