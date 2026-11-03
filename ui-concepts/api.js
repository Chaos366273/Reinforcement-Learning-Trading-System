/* Mock of the PathRise gateway. Same paths the real FastAPI service will expose.
   Set window.PR_API = "https://api.example" before this file to stop intercepting. */
(function () {
  if (window.PR_API) return;
  const nativeFetch = window.fetch.bind(window);
  const NativeES = window.EventSource;

  function json(data, status) {
    return new Response(JSON.stringify(data), { status: status || 200, headers: { "Content-Type": "application/json" } });
  }
  async function mock(url, opts) {
    const path = new URL(String(url), location.origin).pathname;
    const method = ((opts && opts.method) || "GET").toUpperCase();
    const body = opts && opts.body ? JSON.parse(opts.body) : {};
    if (path === "/api/health") return json({ ok: true, mode: "mock" });
    if (path === "/api/plans/current" && method === "GET") return json(Plan.current());
    if (path === "/api/plans/current" && method === "PUT") return json(Plan.replace(body));
    if (path === "/api/plans/propose" && method === "POST") return json(Plan.propose(body.text || ""));
    if (path === "/api/plans/estimate") return json(Plan.estimate());
    if (path === "/api/runs" && method === "POST") {
      const id = "run_" + Math.floor(1000 + Math.random() * 9000);
      try { sessionStorage.setItem("pr-run", id); } catch {}
      return json({ id: id, status: "queued", plan: Plan.current() }, 201);
    }
    if (/^\/api\/runs\/[\w-]+$/.test(path)) return json({ id: path.split("/").pop(), status: "running", best: "ckpt_8" });
    if (/^\/api\/runs\/[\w-]+\/(pause|stop|branch)$/.test(path) && method === "POST") {
      return json({ ok: true, action: path.split("/").pop() });
    }
    if (path === "/api/report") {
      return json({ grade: "B+", exam: 2.1, train: 4.2, gap: 2.1, sharpe: 0.81, maxdd: -3.4, beatBuyHold: true, bestRegime: "trend", worstRegime: "high_vol" });
    }
    if (path === "/api/audit") {
      return json({ items: [
        { at: "Today 15:02", who: "Faisal A.", what: "Exam stored", subject: "run_1842" },
        { at: "Today 14:11", who: "Clawbot", what: "Plan patch applied", subject: "cap 15% → 8%" },
        { at: "Today 09:40", who: "Faisal A.", what: "Quick Train queued", subject: "Balanced" },
      ] });
    }
    return json({ error: "not_found", path: path }, 404);
  }

  window.fetch = function (url, opts) {
    if (String(url).indexOf("/api/") !== -1) return mock(url, opts);
    return nativeFetch(url, opts);
  };

  function MockES(url) {
    this.url = String(url);
    this.readyState = 0;
    const self = this;
    const events = (window.QUICK_RUN || []).map((e, i) => ({ id: String(i + 1), data: e }));
    let i = 0;
    const timer = setInterval(function () {
      if (i >= events.length) { clearInterval(timer); self.readyState = 2; return; }
      const msg = { lastEventId: events[i].id, data: JSON.stringify(events[i].data) };
      if (self.onmessage) self.onmessage(msg);
      i += 1;
    }, 450);
    this.close = function () { clearInterval(timer); };
    this.addEventListener = function (type, fn) { if (type === "message") self.onmessage = fn; };
  }
  window.EventSource = function (url, opts) {
    if (String(url).indexOf("/api/") !== -1) return new MockES(url);
    return new NativeES(url, opts);
  };

  window.PR = window.PR || {};
  PR.api = {
    health: () => fetch("/api/health").then((r) => r.json()),
    plan: () => fetch("/api/plans/current").then((r) => r.json()),
    savePlan: (p) => fetch("/api/plans/current", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) }).then((r) => r.json()),
    propose: (text) => fetch("/api/plans/propose", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text }) }).then((r) => r.json()),
    startRun: () => fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }).then((r) => r.json()),
    report: () => fetch("/api/report").then((r) => r.json()),
    stream: (id, onEvent) => {
      const es = new EventSource("/api/runs/" + id + "/events");
      es.onmessage = (e) => onEvent(JSON.parse(e.data));
      return es;
    },
  };
})();
