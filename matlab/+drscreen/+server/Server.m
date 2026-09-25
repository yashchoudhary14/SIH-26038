classdef Server < handle
    %SERVER The screening backend over HTTP, in MATLAB -- a drop-in for the FastAPI app.
    %
    %   s = drscreen.server.Server('Port', 8000);   % loads the default bundle
    %   s.start();          % serve in the background (MATLAB stays usable)
    %   s.run();            % or serve in the foreground until stop() / Ctrl+C
    %   s.stop();
    %
    %   Endpoints (identical paths, parameters and JSON to src/drscreen/api.py,
    %   so web/index.html runs against it unchanged):
    %
    %     GET  /                       -> redirect to /web/index.html
    %     GET  /web/...                   the review console (static)
    %     GET  /outputs/verification_set/...   committed cases (static)
    %     GET  /health                    liveness, loaded models, operating point
    %     POST /screen                    multipart image -> full JSON result
    %     POST /screen/report             multipart image -> HTML report
    %     GET  /cases                     the real held-out verification photographs
    %     GET  /cases/{name}[?report=1]   screen one of them live
    %     GET  /cases/{name}/image        its pixels
    %     GET  /demo/{grade}[?case=..]    screen a real photograph of that grade
    %     POST /review                    record an ophthalmologist's decision
    %     GET  /audit[?limit=500]         the review log, summarised
    %
    %   Transport (no toolbox either way):
    %     Windows  .NET System.Net.HttpListener (http.sys). The only option inside
    %              a MATLAB Compiler app: since R2025 the MATLAB Runtime starts
    %              without a JVM, so Java sockets do not exist there. Serving on
    %              all interfaces ('Host', '0.0.0.0') needs a one-time URL
    %              reservation by an administrator:
    %                netsh http add urlacl url=http://+:8080/ user=Everyone
    %     other    java.net.ServerSocket from the JVM in desktop MATLAB.
    %   'Transport', 'java' forces the Java socket on Windows desktop MATLAB.
    %
    %   One request is handled at a time; others queue (in http.sys or the socket
    %   backlog). That suits a screening station, a clinic LAN or a demo, not a
    %   public service -- for that, deploy the same pipeline behind MATLAB
    %   Production Server, using request() below as the entry point.

    properties
        Port = 8000
        Host = '127.0.0.1'
        Pipeline
        Verbose = true
        MaxBodyBytes = 25e6
        AuditLog         % JSONL review log; defaults to outputs/audit/reviews.jsonl
        Transport = 'auto'   % 'auto' | 'dotnet' | 'java'
    end

    properties (SetAccess = private)
        Paths
        Cases            % containers.Map: case name -> verification_summary row
        Running = false
    end

    properties (Access = private)
        Socket           % java.net.ServerSocket
        Listener         % System.Net.HttpListener
        Pending          % the listener's outstanding GetContextAsync task
        Timer
    end

    methods
        function obj = Server(varargin)
            ip = inputParser;
            ip.addParameter('Port', 8000);
            ip.addParameter('Host', '127.0.0.1');
            ip.addParameter('Bundle', drscreen.constants().DEFAULT_BUNDLE);
            ip.addParameter('Pipeline', []);
            ip.addParameter('Verbose', true);
            ip.addParameter('AuditLog', '');
            ip.addParameter('Transport', 'auto');
            ip.parse(varargin{:});
            obj.Transport = lower(char(ip.Results.Transport));
            obj.Port = ip.Results.Port;
            obj.Host = ip.Results.Host;
            obj.Verbose = ip.Results.Verbose;
            obj.Paths = drscreen.paths();
            obj.AuditLog = ip.Results.AuditLog;
            if isempty(obj.AuditLog), obj.AuditLog = obj.Paths.audit; end
            if isempty(ip.Results.Pipeline)
                obj.Pipeline = drscreen.Pipeline.load(ip.Results.Bundle);
            else
                obj.Pipeline = ip.Results.Pipeline;
            end
            obj.loadCases();
        end

        function u = url(obj)
            u = sprintf('http://%s:%d/', obj.Host, obj.Port);
        end

        function start(obj)
            %START Serve in the background via a MATLAB timer.
            obj.open();
            obj.Timer = timer('ExecutionMode', 'fixedSpacing', 'Period', 0.05, ...
                'BusyMode', 'drop', 'Name', 'drscreen-http', ...
                'TimerFcn', @(~, ~) obj.pollOnce(2));
            start(obj.Timer);
            obj.log('serving %s (background)', obj.url());
        end

        function run(obj, stopFcn)
            %RUN Serve in the foreground until stop(), Ctrl+C, or stopFcn() is true.
            if nargin < 2, stopFcn = @() false; end
            obj.open();
            obj.log('serving %s -- Ctrl+C to stop', obj.url());
            cleaner = onCleanup(@() obj.stop());
            while obj.Running && ~stopFcn()
                obj.pollOnce(250);
                drawnow limitrate;
            end
        end

        function stop(obj)
            if ~isempty(obj.Timer) && isvalid(obj.Timer)
                stop(obj.Timer);
                delete(obj.Timer);
            end
            obj.Timer = [];
            if ~isempty(obj.Socket)
                try, obj.Socket.close(); catch, end
            end
            obj.Socket = [];
            if ~isempty(obj.Listener)
                try, obj.Listener.Stop(); obj.Listener.Close(); catch, end
            end
            obj.Listener = [];
            obj.Pending = [];
            if obj.Running
                obj.log('stopped');
            end
            obj.Running = false;
        end

        function delete(obj)
            obj.stop();
        end

        function resp = request(obj, method, target, body, contentType)
            %REQUEST Dispatch one request without a socket.
            %   resp = s.request('GET', '/health')
            %   resp = s.request('POST', '/screen', multipartBytes, 'multipart/form-data; boundary=..')
            %   Returns struct(status, type, body); decode JSON with
            %   jsondecode(native2unicode(resp.body, 'UTF-8')). Used by the tests,
            %   and the natural entry point behind MATLAB Production Server.
            if nargin < 4, body = uint8([]); end
            if nargin < 5, contentType = ''; end
            req.method = upper(method);
            req.headers = containers.Map();
            if ~isempty(contentType), req.headers('content-type') = contentType; end
            q = strfind(target, '?');
            if isempty(q)
                req.path = decodeUrl(target);
                req.query = containers.Map();
            else
                req.path = decodeUrl(target(1:q(1) - 1));
                req.query = parseQuery(target(q(1) + 1:end));
            end
            req.body = uint8(body(:))';
            try
                resp = obj.route(req);
            catch err
                resp = obj.json(struct('detail', err.message), 500);
            end
        end

        function served = pollOnce(obj, timeoutMs)
            %POLLONCE Accept and fully handle at most one request.
            served = false;
            if ~isempty(obj.Listener)
                if isempty(obj.Pending), return; end
                try
                    ready = obj.Pending.Wait(int32(timeoutMs));
                catch
                    return                          % listener stopped under us
                end
                if ~ready, return; end
                ctx = obj.Pending.Result;
                obj.Pending = obj.Listener.GetContextAsync();
                served = true;
                obj.handleDotnet(ctx);
                return
            end
            if isempty(obj.Socket), return; end
            obj.Socket.setSoTimeout(timeoutMs);
            try
                sock = obj.Socket.accept();
            catch err
                if contains(err.message, 'SocketTimeout') || contains(err.message, 'timed out') ...
                        || contains(err.message, 'Socket closed')
                    return
                end
                rethrow(err);
            end
            served = true;
            obj.handle(sock);
        end
    end

    methods
        function open(obj)
            %OPEN Bind the listening socket (start/run call this themselves).
            %   Fails if the port is taken. SO_REUSEADDR is set only off Windows:
            %   there it merely allows a quick restart, but on Windows it would
            %   let this server bind a port another program is listening on.
            if obj.Running, return; end
            if strcmp(obj.transportKind(), 'dotnet')
                NET.addAssembly('System');
                l = System.Net.HttpListener();
                if any(strcmp(obj.Host, {'0.0.0.0', '+', '*', ''}))
                    hosts = {'+'};                          % all interfaces (needs urlacl)
                elseif any(strcmp(obj.Host, {'127.0.0.1', 'localhost'}))
                    hosts = {'127.0.0.1', 'localhost'};     % http.sys matches the Host header
                else
                    hosts = {obj.Host};
                end
                for h = hosts
                    l.Prefixes.Add(sprintf('http://%s:%d/', h{1}, obj.Port));
                end
                l.IgnoreWriteExceptions = true;
                try
                    l.Start();
                catch err
                    if contains(err.message, 'Access is denied')
                        error('drscreen:http', ['Windows refused to serve %s on port %d without ' ...
                            'a URL reservation. As administrator, once:\n' ...
                            '    netsh http add urlacl url=http://+:%d/ user=Everyone'], ...
                            obj.Host, obj.Port, obj.Port);
                    end
                    rethrow(err);
                end
                obj.Listener = l;
                obj.Pending = l.GetContextAsync();
            else
                if ~usejava('jvm')
                    error('drscreen:http', ['No Java in this MATLAB session, so no socket ' ...
                        'transport: the web server needs Windows (.NET) or desktop MATLAB.']);
                end
                ss = java.net.ServerSocket();
                if ~ispc
                    ss.setReuseAddress(true);
                end
                ss.bind(java.net.InetSocketAddress(obj.Host, obj.Port), 50);
                obj.Socket = ss;
            end
            obj.Running = true;
        end

        function k = transportKind(obj)
            k = obj.Transport;
            if strcmp(k, 'auto')
                if ispc, k = 'dotnet'; else, k = 'java'; end
            end
        end
    end

    methods (Access = private)

        function log(obj, fmt, varargin)
            if obj.Verbose
                fprintf(['[drscreen-http] ' fmt '\n'], varargin{:});
            end
        end

        function loadCases(obj)
            obj.Cases = containers.Map();
            f = fullfile(obj.Paths.vset, 'verification_summary.json');
            if ~isfile(f), return; end
            s = jsondecode(drscreen.io.readText(f));
            rows = s.cases;
            for i = 1:numel(rows)
                if iscell(rows), r = rows{i}; else, r = rows(i); end
                obj.Cases(r.case) = r;
            end
        end

        % ---------------------------------------------------------------
        % HTTP plumbing
        % ---------------------------------------------------------------
        function handle(obj, sock)
            t0 = tic;
            resp = [];
            req = struct('method', '?', 'path', '?');
            try
                sock.setSoTimeout(30000);
                req = obj.readRequest(sock);
                resp = obj.route(req);
            catch err
                resp = obj.json(struct('detail', err.message), 500);
            end
            try
                obj.writeResponse(sock, resp);
            catch
            end
            try, sock.close(); catch, end
            obj.log('%s %s -> %d (%.0f ms)', req.method, req.path, resp.status, toc(t0) * 1000);
        end

        function handleDotnet(obj, ctx)
            % One HttpListenerContext: http.sys has already parsed the request,
            % so only the body is read here; routing is shared with request().
            t0 = tic;
            rq = ctx.Request;
            method = char(rq.HttpMethod);
            target = char(rq.RawUrl);
            ctype = '';
            if ~isempty(rq.ContentType), ctype = char(rq.ContentType); end
            try
                if rq.ContentLength64 > obj.MaxBodyBytes
                    resp = obj.json(struct('detail', sprintf('request body too large (%d bytes)', ...
                        rq.ContentLength64)), 413);
                else
                    body = uint8([]);
                    if rq.HasEntityBody
                        ms = System.IO.MemoryStream();
                        rq.InputStream.CopyTo(ms);
                        body = uint8(ms.ToArray());
                    end
                    resp = obj.request(method, target, body, ctype);
                end
            catch err
                resp = obj.json(struct('detail', err.message), 500);
            end
            try
                r = ctx.Response;
                r.StatusCode = int32(resp.status);
                r.ContentType = resp.type;
                r.KeepAlive = false;
                r.AddHeader('Cache-Control', 'no-store');
                r.AddHeader('Access-Control-Allow-Origin', '*');
                r.AddHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
                r.AddHeader('Access-Control-Allow-Headers', '*');
                if isfield(resp, 'location') && ~isempty(resp.location)
                    r.RedirectLocation = resp.location;
                end
                bytes = uint8(resp.body(:));
                r.ContentLength64 = numel(bytes);
                if ~isempty(bytes)
                    r.OutputStream.Write(NET.convertArray(bytes, 'System.Byte'), 0, numel(bytes));
                end
                r.Close();
            catch
                try, ctx.Response.Abort(); catch, end
            end
            q = strfind(target, '?');
            if ~isempty(q), target = target(1:q(1) - 1); end
            obj.log('%s %s -> %d (%.0f ms)', method, target, resp.status, toc(t0) * 1000);
        end

        function req = readRequest(obj, sock)
            in = sock.getInputStream();
            buf = zeros(1, 16384, 'uint8');
            n = 0;
            while true
                b = in.read();
                if b < 0, break; end
                n = n + 1;
                if n > numel(buf)
                    error('drscreen:http', 'request header too large');
                end
                buf(n) = b;
                if n >= 4 && buf(n) == 10 && buf(n - 1) == 13 && buf(n - 2) == 10 && buf(n - 3) == 13
                    break
                end
            end
            head = char(buf(1:n));
            lines = regexp(head, '\r\n', 'split');
            first = strsplit(strtrim(lines{1}), ' ');
            if numel(first) < 2
                error('drscreen:http', 'malformed request line');
            end
            req.method = upper(first{1});
            target = first{2};
            req.headers = containers.Map();
            for i = 2:numel(lines)
                k = strfind(lines{i}, ':');
                if isempty(k), continue; end
                req.headers(lower(strtrim(lines{i}(1:k(1) - 1)))) = strtrim(lines{i}(k(1) + 1:end));
            end
            q = strfind(target, '?');
            if isempty(q)
                req.path = decodeUrl(target);
                req.query = containers.Map();
            else
                req.path = decodeUrl(target(1:q(1) - 1));
                req.query = parseQuery(target(q(1) + 1:end));
            end
            req.body = zeros(1, 0, 'uint8');
            if isKey(req.headers, 'content-length')
                len = str2double(req.headers('content-length'));
                if len > obj.MaxBodyBytes
                    error('drscreen:http', 'request body too large (%d bytes)', len);
                end
                if len > 0
                    ch = java.nio.channels.Channels.newChannel(in);
                    bb = java.nio.ByteBuffer.allocate(len);
                    while bb.hasRemaining()
                        if ch.read(bb) < 0, break; end
                    end
                    req.body = typecast(bb.array(), 'uint8')';
                end
            end
        end

        function writeResponse(~, sock, resp)
            codes = containers.Map({200, 204, 302, 400, 404, 405, 413, 500, 503}, ...
                {'OK', 'No Content', 'Found', 'Bad Request', 'Not Found', ...
                 'Method Not Allowed', 'Payload Too Large', 'Internal Server Error', ...
                 'Service Unavailable'});
            reason = 'OK';
            if isKey(codes, resp.status), reason = codes(resp.status); end
            body = resp.body;
            hdr = sprintf(['HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n' ...
                'Connection: close\r\nCache-Control: no-store\r\n' ...
                'Access-Control-Allow-Origin: *\r\n' ...
                'Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n' ...
                'Access-Control-Allow-Headers: *\r\n'], ...
                resp.status, reason, resp.type, numel(body));
            if isfield(resp, 'location') && ~isempty(resp.location)
                hdr = [hdr sprintf('Location: %s\r\n', resp.location)];
            end
            bytes = [uint8([hdr sprintf('\r\n')]), uint8(body(:))'];
            out = sock.getOutputStream();
            out.write(typecast(bytes, 'int8'));
            out.flush();
        end

        % ---------------------------------------------------------------
        % Routing
        % ---------------------------------------------------------------
        function resp = route(obj, req)
            p = req.path;
            if strcmp(req.method, 'OPTIONS')
                resp = struct('status', 204, 'type', 'text/plain', 'body', uint8([]));
                return
            end
            parts = strsplit(strip(p, 'both', '/'), '/');
            if isempty(parts{1}), parts = {}; end

            if isempty(parts)
                resp = struct('status', 302, 'type', 'text/plain', 'body', uint8([]), ...
                              'location', '/web/index.html');
            elseif strcmp(parts{1}, 'web')
                resp = obj.staticFile(obj.Paths.web, parts(2:end), 'index.html');
            elseif numel(parts) >= 2 && strcmp(parts{1}, 'outputs') && strcmp(parts{2}, 'verification_set')
                resp = obj.staticFile(obj.Paths.vset, parts(3:end), '');
            elseif strcmp(p, '/README.md')
                resp = obj.staticFile(obj.Paths.docs, {'README.md'}, '');
            elseif strcmp(p, '/RESULTS.md')
                resp = obj.staticFile(obj.Paths.docs, {'RESULTS.md'}, '');
            elseif strcmp(p, '/health')
                resp = obj.json(obj.health());
            elseif strcmp(p, '/screen') && strcmp(req.method, 'POST')
                resp = obj.screenUpload(req, false);
            elseif strcmp(p, '/screen/report') && strcmp(req.method, 'POST')
                resp = obj.screenUpload(req, true);
            elseif strcmp(p, '/cases')
                resp = obj.json(obj.listCases());
            elseif numel(parts) == 3 && strcmp(parts{1}, 'cases') && strcmp(parts{3}, 'image')
                resp = obj.caseImage(parts{2});
            elseif numel(parts) == 2 && strcmp(parts{1}, 'cases')
                resp = obj.screenCase(parts{2}, flag(req.query, 'report'));
            elseif numel(parts) == 2 && strcmp(parts{1}, 'demo')
                resp = obj.demo(str2double(parts{2}), getq(req.query, 'case', ''), ...
                                flag(req.query, 'report'));
            elseif strcmp(p, '/review') && strcmp(req.method, 'POST')
                resp = obj.json(obj.review(req));
            elseif strcmp(p, '/audit')
                lim = str2double(getq(req.query, 'limit', '500'));
                resp = obj.json(obj.audit(lim));
            else
                resp = obj.json(struct('detail', 'Not Found'), 404);
            end
        end

        % ---------------------------------------------------------------
        % Endpoints
        % ---------------------------------------------------------------
        function h = health(obj)
            pipe = obj.Pipeline;
            h = struct( ...
                'status', 'ok', ...
                'segmentation_loaded', ~isempty(pipe.Segmenter), ...
                'grader_loaded', ~isempty(pipe.Grader), ...
                'device', sprintf('cpu (MATLAB R%s)', version('-release')), ...
                'referral_threshold', pipe.Cfg.referral_threshold, ...
                'temperature', pipe.Cfg.temperature, ...
                'model_version', pipe.Cfg.model_version, ...
                'artifacts_dir', ['models/' char(string(pipe.Bundle))], ...
                'image_size', pipe.Cfg.size, ...
                'quality_thresholds', drscreen.preprocess.qualityThresholds(), ...
                'runtime', 'matlab');
        end

        function resp = screenUpload(obj, req, asReport)
            [img, name] = obj.uploadedImage(req);
            if isempty(img)
                resp = obj.json(struct('detail', 'Could not decode the uploaded file as an image.'), 400);
                return
            end
            [res, art] = obj.Pipeline.run(img, name);
            if asReport
                prov = sprintf('Real image attached %s uploaded file ''%s'', %d%s%d px.', ...
                    char(8212), name, size(img, 2), char(215), size(img, 1));
                resp = obj.html(drscreen.report.renderHtml(res, art, prov));
            else
                resp = obj.json(obj.withPanel(res, art));
            end
        end

        function out = listCases(obj)
            names = keys(obj.Cases);
            rows = cell(1, numel(names));
            g = zeros(1, numel(names));
            for i = 1:numel(names)
                r = obj.Cases(names{i});
                rows{i} = struct('case', r.case, 'true_grade', r.true_grade, ...
                    'true_label', r.true_label, 'source', r.source, ...
                    'subject', getf(r, 'subject', ''), 'provenance', getf(r, 'provenance', ''), ...
                    'thumbnail', ['/cases/' r.case '/image']);
                g(i) = r.true_grade;
            end
            [~, order] = sortrows([g(:), (1:numel(g))']);
            if isempty(rows)
                out = struct('n', 0, 'cases', {{}}, 'note', ['No verification set on disk. ' ...
                    'Use /demo/{grade} to screen one of the committed held-out photographs.']);
            else
                out = struct('n', numel(rows), 'real_images', true, 'cases', {rows(order)});
            end
        end

        function f = casePath(obj, name)
            f = '';
            if ~isKey(obj.Cases, name), return; end
            r = obj.Cases(name);
            rel = getf(r, 'image', ['images/' name '.jpg']);
            f = fullfile(obj.Paths.vset, strrep(rel, '/', filesep));
            if ~isfile(f), f = ''; end
        end

        function resp = caseImage(obj, name)
            f = obj.casePath(name);
            if isempty(f)
                resp = obj.json(struct('detail', ['no such case: ' name]), 404);
                return
            end
            resp = obj.fileResponse(f);
        end

        function resp = screenCase(obj, name, asReport)
            f = obj.casePath(name);
            if isempty(f)
                resp = obj.json(struct('detail', ['no such case: ' name]), 404);
                return
            end
            meta = obj.Cases(name);
            [res, art] = obj.Pipeline.run(f, name);
            prov = getf(meta, 'provenance', 'Real image attached.');
            if asReport
                resp = obj.html(drscreen.report.renderHtml(res, art, prov));
                return
            end
            payload = res;
            payload.synthetic = false;
            payload.provenance = prov;
            payload.ground_truth = struct('grade', meta.true_grade, 'label', meta.true_label, ...
                'source', meta.source, 'subject', getf(meta, 'subject', ''), ...
                'reference_standard', 'corpus label, held-out split');
            resp = obj.json(obj.withPanel(payload, art));
        end

        function resp = demo(obj, grade, caseName, asReport)
            if isnan(grade) || grade < 0 || grade > 4 || grade ~= fix(grade)
                resp = obj.json(struct('detail', 'grade must be one of [0, 1, 2, 3, 4]'), 400);
                return
            end
            try
                [img, name, trueGrade] = drscreen.samples.load(caseName, grade);
            catch err
                resp = obj.json(struct('detail', err.message), 503);
                return
            end
            [res, art] = obj.Pipeline.run(img, name);
            prov = sprintf(['Real fundus photograph (%s), held-out test split, reference ' ...
                'grade %d. Demonstration only, not clinical evidence.'], name, trueGrade);
            if asReport
                resp = obj.html(drscreen.report.renderHtml(res, art, prov));
                return
            end
            payload = res;
            payload.synthetic = false;
            payload.provenance = prov;
            payload.ground_truth = struct('grade', trueGrade, 'case', name);
            resp = obj.json(obj.withPanel(payload, art));
        end

        function out = review(obj, req)
            F = parseForm(req);
            mg = str2double(getq(F, 'model_grade', 'NaN'));
            rg = str2double(getq(F, 'reviewer_grade', 'NaN'));
            if isnan(mg) || isnan(rg)
                error('drscreen:http', 'model_grade and reviewer_grade are required');
            end
            if mg == rg
                agree = 'exact';
            elseif abs(mg - rg) == 1
                agree = 'within_one';
            else
                agree = 'disagree';
            end
            rec = struct( ...
                'timestamp', char(datetime('now', 'TimeZone', 'UTC', ...
                    'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSSxxx')), ...
                'image_id', getq(F, 'image_id', ''), ...
                'model_grade', mg, 'reviewer_grade', rg, ...
                'reviewer', getq(F, 'reviewer', 'unknown'), ...
                'review_seconds', str2double(getq(F, 'seconds', '0')), ...
                'notes', getq(F, 'notes', ''), ...
                'agreement', agree);
            drscreen.io.writeText(obj.AuditLog, [jsonencode(rec) newline], 'a');
            out = rec;
            out.recorded = true;
            out = orderfields(out, [numel(fieldnames(out)), 1:numel(fieldnames(out)) - 1]);
        end

        function out = audit(obj, limit)
            out = struct('n', 0, 'reviews', {{}}, 'summary', struct());
            if ~isfile(obj.AuditLog), return; end
            lines = strsplit(drscreen.io.readText(obj.AuditLog), newline);
            lines = lines(~cellfun(@(l) isempty(strtrim(l)), lines));
            if isnan(limit), limit = 500; end
            lines = lines(max(1, end - limit + 1):end);
            rows = cellfun(@jsondecode, lines, 'UniformOutput', false);
            n = numel(rows);
            if n == 0, return; end
            ag = cellfun(@(r) r.agreement, rows, 'UniformOutput', false);
            secs = cellfun(@(r) r.review_seconds, rows);
            secs = secs(secs > 0);
            med = NaN; u30 = NaN;
            if ~isempty(secs)
                med = median(secs);
                u30 = mean(secs <= 30);
            end
            out = struct('n', n, 'reviews', {rows(max(1, end - 49):end)}, ...
                'summary', struct( ...
                    'exact_agreement', mean(strcmp(ag, 'exact')), ...
                    'within_one_grade', mean(ismember(ag, {'exact', 'within_one'})), ...
                    'median_review_seconds', med, ...
                    'under_30s_fraction', u30));
        end

        % ---------------------------------------------------------------
        % Helpers
        % ---------------------------------------------------------------
        function payload = withPanel(~, payload, art)
            try
                panel = drscreen.report.buildReviewPanel(payload, art);
                payload.panel_jpeg_b64 = drscreen.io.base64Image(panel, 'jpg', 88);
            catch
            end
        end

        function [img, name] = uploadedImage(~, req)
            img = [];
            name = 'case';
            parts = parseMultipart(req);
            for i = 1:numel(parts)
                if strcmp(parts(i).name, 'file')
                    [~, name] = fileparts(parts(i).filename);
                    if isempty(name), name = 'case'; end
                    img = decodeImage(parts(i).data, parts(i).filename);
                    return
                end
            end
        end

        function resp = staticFile(obj, root, parts, defaultFile)
            rel = strjoin(parts, filesep);
            if isempty(rel), rel = defaultFile; end
            if any(strcmp(parts, '..')) || contains(rel, [':' filesep])
                resp = obj.json(struct('detail', 'forbidden'), 404);
                return
            end
            f = fullfile(root, rel);
            if isfolder(f) && ~isempty(defaultFile)
                f = fullfile(f, defaultFile);
            end
            if ~isfile(f)
                resp = obj.json(struct('detail', 'Not Found'), 404);
                return
            end
            resp = obj.fileResponse(f);
        end

        function resp = fileResponse(~, f)
            [~, ~, ext] = fileparts(f);
            types = struct('html', 'text/html; charset=utf-8', 'htm', 'text/html; charset=utf-8', ...
                'js', 'application/javascript; charset=utf-8', 'css', 'text/css; charset=utf-8', ...
                'json', 'application/json; charset=utf-8', 'png', 'image/png', ...
                'jpg', 'image/jpeg', 'jpeg', 'image/jpeg', 'svg', 'image/svg+xml', ...
                'ico', 'image/x-icon', 'md', 'text/plain; charset=utf-8', ...
                'txt', 'text/plain; charset=utf-8', 'woff2', 'font/woff2');
            key = lower(strrep(ext, '.', ''));
            type = 'application/octet-stream';
            if isfield(types, key), type = types.(key); end
            fid = fopen(f, 'r');
            data = fread(fid, Inf, '*uint8')';
            fclose(fid);
            resp = struct('status', 200, 'type', type, 'body', data);
        end

        function resp = json(~, payload, status)
            if nargin < 3, status = 200; end
            resp = struct('status', status, 'type', 'application/json', ...
                'body', unicode2native(jsonencode(payload), 'UTF-8'));
        end

        function resp = html(~, text)
            resp = struct('status', 200, 'type', 'text/html; charset=utf-8', ...
                'body', unicode2native(text, 'UTF-8'));
        end
    end
end


% ======================================================================
% Local functions
% ======================================================================
function s = decodeUrl(s)
% Path segments: '+' is a literal plus.
s = percentDecode(s, false);
end


function s = percentDecode(s, plusIsSpace)
% %XX escapes -> bytes -> UTF-8 text. Plain MATLAB, because a compiled app has
% no JVM (java.net.URLDecoder is not available there). A malformed escape is
% kept as literal text rather than rejected.
s = char(s);
if plusIsSpace
    s(s == '+') = ' ';
end
b = zeros(1, numel(s), 'uint8');
n = 0;
i = 1;
while i <= numel(s)
    if s(i) == '%' && i + 2 <= numel(s) && all(isstrprop(s(i + 1:i + 2), 'xdigit'))
        n = n + 1;
        b(n) = hex2dec(s(i + 1:i + 2));
        i = i + 3;
    else
        n = n + 1;
        b(n) = uint8(s(i));
        i = i + 1;
    end
end
s = native2unicode(b(1:n), 'UTF-8');
end


function m = parseQuery(q)
% Query strings: '+' in a value is a space (form encoding).
m = containers.Map();
for kv = strsplit(q, '&')
    if isempty(kv{1}), continue; end
    e = strfind(kv{1}, '=');
    if isempty(e)
        m(decodeUrl(kv{1})) = '';
    else
        m(decodeUrl(kv{1}(1:e(1) - 1))) = percentDecode(kv{1}(e(1) + 1:end), true);
    end
end
end


function v = getq(m, k, default)
if isKey(m, k), v = m(k); else, v = default; end
end


function tf = flag(m, k)
tf = isKey(m, k) && any(strcmpi(m(k), {'1', 'true', 'yes', 'on'}));
end


function v = getf(s, k, default)
if isfield(s, k) && ~isempty(s.(k)), v = s.(k); else, v = default; end
end


function parts = parseMultipart(req)
parts = struct('name', {}, 'filename', {}, 'data', {});
if ~isKey(req.headers, 'content-type'), return; end
ct = req.headers('content-type');
tok = regexp(ct, 'boundary="?([^";]+)"?', 'tokens', 'once');
if isempty(tok), return; end
delim = ['--' tok{1}];
body = req.body;
txt = char(body);
idx = strfind(txt, delim);
for k = 1:numel(idx) - 1
    a = idx(k) + numel(delim);
    b = idx(k + 1) - 1;
    seg = body(a:b);
    if numel(seg) >= 2 && seg(1) == 13 && seg(2) == 10
        seg = seg(3:end);
    end
    sep = strfind(char(seg), sprintf('\r\n\r\n'));
    if isempty(sep), continue; end
    head = char(seg(1:sep(1) - 1));
    data = seg(sep(1) + 4:end);
    if numel(data) >= 2 && data(end - 1) == 13 && data(end) == 10
        data = data(1:end - 2);
    end
    nm = regexp(head, 'name="([^"]*)"', 'tokens', 'once');
    fn = regexp(head, 'filename="([^"]*)"', 'tokens', 'once');
    p = struct('name', '', 'filename', '', 'data', data);
    if ~isempty(nm), p.name = nm{1}; end
    if ~isempty(fn), p.filename = fn{1}; end
    parts(end + 1) = p; %#ok<AGROW>
end
end


function F = parseForm(req)
F = containers.Map();
ct = '';
if isKey(req.headers, 'content-type'), ct = req.headers('content-type'); end
if contains(ct, 'multipart/form-data')
    parts = parseMultipart(req);
    for i = 1:numel(parts)
        F(parts(i).name) = native2unicode(parts(i).data, 'UTF-8');
    end
else
    F = parseQuery(native2unicode(req.body, 'UTF-8'));
end
end


function img = decodeImage(data, filename)
img = [];
if numel(data) < 8, return; end
[~, ~, ext] = fileparts(filename);
if isempty(ext)
    if data(1) == 255 && data(2) == 216
        ext = '.jpg';
    elseif data(1) == 137 && data(2) == 80
        ext = '.png';
    elseif data(1) == 66 && data(2) == 77
        ext = '.bmp';
    else
        ext = '.tif';
    end
end
tmp = [tempname ext];
fid = fopen(tmp, 'w');
fwrite(fid, data, 'uint8');
fclose(fid);
try
    img = drscreen.io.readImage(tmp);
catch
    img = [];
end
delete(tmp);
end
