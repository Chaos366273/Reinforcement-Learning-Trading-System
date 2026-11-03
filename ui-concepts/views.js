/* Logged-in pages that share one renderer so Simple and Pro stay in sync. */
(function () {
  const main = document.getElementById("main");
  if (!main || !main.dataset.view) return;
  const view = main.dataset.view;
  const params = new URLSearchParams(location.search);
  const SEC = { AAPL:"Tech", MSFT:"Tech", NVDA:"Tech", GOOGL:"Tech", META:"Tech", AMD:"Tech", INTC:"Tech", QCOM:"Tech", ORCL:"Tech", CRM:"Tech", JPM:"Financials", V:"Financials", BAC:"Financials", MA:"Financials", AMZN:"Consumer", WMT:"Consumer", HD:"Consumer", NKE:"Consumer", KO:"Consumer", COST:"Consumer", DIS:"Consumer", T:"Telecom", VZ:"Telecom", JNJ:"Healthcare", UNH:"Healthcare", PFE:"Healthcare", LLY:"Healthcare", ABBV:"Healthcare", XOM:"Energy", CVX:"Energy" };
  const day = (t) => (PR.rng(PR.hash(t))() - 0.46) * 4;

  function shell(title, sub, body) {
    const state = params.get("state") || (params.has("empty") ? "empty" : "");
    if (state === "loading") return `<div class="page-head"><div><h1>${title}</h1></div></div><div class="card"><div class="card-b" style="display:grid;gap:8px"><div class="skeleton" style="height:18px;width:30%"></div><div class="skeleton" style="height:160px"></div></div></div>`;
    if (state === "error") return `<div class="page-head"><div><h1>${title}</h1></div></div><div class="banner" role="alert">This page failed to load. <a href="${location.pathname}">Retry</a></div>`;
    const offline = state === "offline" ? `<div class="banner" role="alert">Offline. This is the last saved view.</div>` : "";
    if (state === "empty") return `<section class="card state-card" style="display:flex"><div class="state-ico"><span data-i="inbox" data-s="26"></span></div><h3>Nothing here yet</h3><p>Train a bot or open the demo replay. This page fills in after the first run.</p><div class="state-actions"><a class="btn btn-gold" href="training.html">Build a bot</a><a class="btn" href="arena.html?demo=1">Watch the demo</a></div></section>`;
    return `${offline}<div class="page-head"><div><h1>${title}</h1><p>${sub}</p></div></div>${body}`;
  }

  const views = {
    markets() {
      const sec = params.get("sec") || "All";
      const list = PR.data.tickers.filter((t) => sec === "All" || SEC[t] === sec);
      const sectors = ["All", ...new Set(Object.values(SEC))];
      const cards = list.map((t) => {
        const d = day(t);
        const series = PR.series(16, PR.hash(t), 100, 100 + d, 0.01);
        return `<button class="card" data-sym="${t}" style="text-align:left;color:inherit;cursor:pointer"><div class="card-b"><div style="display:flex;justify-content:space-between"><b class="num">${t}</b><span class="${PR.cls(d)} num">${d >= 0 ? "+" : ""}${d.toFixed(1)}%</span></div><div class="muted" style="font-size:12px">${SEC[t] || "Other"}</div>${PR.spark(series, 120, 32, d >= 0 ? "var(--up)" : "var(--down)")}</div></button>`;
      }).join("");
      return shell("Markets", "The 30 names the bot is allowed to touch.", `
        <div class="seg" id="secs">${sectors.map((s) => `<button data-sec="${s}" class="${s === sec ? "on" : ""}">${s}</button>`).join("")}</div>
        <div class="grid-3">${cards}</div>
        <section class="card" data-mode-only="pro"><div class="card-h"><h2>Regime</h2></div><div class="card-b"><p>Bull trend 0.62 · low vol 0.21 · mean revert 0.11 · high vol 0.06. The controller allows normal size while trend leads.</p><p class="muted">Signal history is stored with each order so the explainer can open it later.</p></div></section>`);
    },
    portfolio() {
      const rows = PR.data.positions.map((p) => `<tr><td><b>${p.s}</b><div class="muted">${p.n}</div></td><td class="r num">${p.q}</td><td class="r num">${PR.usd(p.last)}</td><td class="r num ${PR.cls(p.pnl)}">${PR.usd(p.pnl)}</td><td class="r num ${PR.cls(p.pnlPct)}">${PR.pct(p.pnlPct)}</td></tr>`).join("");
      const orders = PR.data.trades.map((t) => `<tr><td class="num">${t.t}</td><td>${t.a}</td><td class="num">${t.s}</td><td class="r">${t.tier}</td></tr>`).join("");
      return shell("Portfolio", "Paper book · " + PR.usd(PR.data.equity, 0), `
        <div data-mode-only="simple" class="card"><div class="card-b"><p>Today the book is up about ${PR.usd(PR.data.dayPnl, 0)}. Most of that is NVDA and Costco. JPMorgan and Exxon are the drags.</p></div></div>
        <section class="card"><div class="card-h"><h2>Positions</h2></div><div class="card-b tbl-wrap"><table class="tbl"><thead><tr><th>Name</th><th class="r">Qty</th><th class="r">Last</th><th class="r">P&L</th><th class="r">%</th></tr></thead><tbody>${rows}</tbody></table></div></section>
        <section class="card"><div class="card-h"><h2>Orders today</h2></div><div class="card-b tbl-wrap"><table class="tbl"><thead><tr><th>Time</th><th>Side</th><th>Name</th><th class="r">Gate</th></tr></thead><tbody>${orders}</tbody></table></div></section>
        <section class="card" data-mode-only="pro"><div class="card-h"><h2>Exposure, turnover, slippage</h2></div><div class="card-b"><p>Tech 22% · Healthcare 9% · Consumer 6% · Financials 3% · Energy 1%.</p><p class="muted">Turnover today 0.18 of the book. Average simulated slippage 1.2 bps. Neither number is a forecast.</p></div></section>`);
    },
    risk() {
      const p = window.Plan ? Plan.current() : { cap: 15, stop: 8, halt: 25 };
      return shell("Risk", "How close the paper book is to a limit.", `
        <div class="grid-3">
          <div class="card"><div class="card-b"><div class="eyebrow">Gross</div><b class="num" style="font-size:26px">41%</b><div class="meter"><i style="width:41%;background:var(--gold)"></i></div><span class="muted">Limit 100% · far away</span></div></div>
          <div class="card"><div class="card-b"><div class="eyebrow">Largest name</div><b class="num" style="font-size:26px">7.2%</b><div class="meter"><i style="width:48%;background:var(--up)"></i></div><span class="muted">Cap ${p.cap}% · NVDA</span></div></div>
          <div class="card"><div class="card-b"><div class="eyebrow">Drawdown</div><b class="num" style="font-size:26px">−1.8%</b><div class="meter"><i style="width:7%;background:var(--up)"></i></div><span class="muted">Halt ${p.halt}%</span></div></div>
        </div>
        <section class="card" data-mode-only="simple"><div class="card-b lock-note">${PR.icon("risk", 16)} These limits are protected by PathRise in Simple. Switch to Pro to edit them. They still cannot exceed the plan schema.</div></section>
        <section class="card" data-mode-only="pro"><div class="card-h"><h2>Edit limits</h2></div><div class="card-b" style="display:grid;gap:10px">
          <div class="slider"><label>Cap ${p.cap}%</label><input id="cap" type="range" min="5" max="15" value="${p.cap}"></div>
          <div class="slider"><label>Trailing stop ${p.stop}%</label><input id="stop" type="range" min="4" max="12" value="${p.stop}"></div>
          <div class="slider"><label>Session halt ${p.halt}%</label><input id="halt" type="range" min="8" max="25" value="${p.halt}"></div>
          <p class="muted">A 2× fee stress on the last exam still finishes positive (+0.6%). A kill-switch test cancels open paper orders and blocks new buys until you clear it.</p>
          <button class="btn" id="kill">Run kill-switch test</button>
        </div></section>`);
    },
    agent() {
      return shell("Clawbot", "Same tools as Build a bot, plus the notes you save.", `
        <div class="grid-2">
          <section class="card"><div class="card-h"><h2>Chat</h2></div><div class="card-b">
            <div class="coach" id="alog"></div>
            <form id="aform" style="display:flex;gap:8px;margin-top:10px"><input id="ain" aria-label="Message Clawbot" placeholder="make it more cautious" style="flex:1;padding:10px 12px;border-radius:11px;border:1px solid var(--line-2);background:var(--bg-2);color:inherit"><button class="btn btn-gold" type="submit">Send</button></form>
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px" id="prompts"></div>
          </div></section>
          <section class="card"><div class="card-h"><h2>Model card</h2><span class="pill gold"><i></i>ckpt_8</span></div><div class="card-b">
            <p>Rules mentor, then PPO Quick Train, then checkpoint ckpt_8.</p>
            <p data-mode-only="simple">It is strongest on calm trend days and weakest when the market is violent. It should stand aside then.</p>
            <p data-mode-only="pro" class="muted">Exam Sharpe 0.81 deterministic, 0.66 stochastic. Entropy coef ended at 0.04 after the boost at step 12,288.</p>
            <a class="btn btn-sm" href="report.html">Report card</a>
          </div></section>
        </div>`);
    },
    studio() {
      if (!PR.isPro()) {
        return shell("Studio", "Reward, features, and curriculum.", `
          <section class="card"><div class="card-b" style="display:grid;gap:10px;justify-items:start"><h2 style="margin:0">Pro edits the reward</h2><p class="muted">Simple keeps the research defaults and shows them as protected. The sentence the bot feels is still visible on Build a bot.</p><button class="btn btn-gold" id="unlock">Switch to Pro</button></div></section>`);
      }
      const p = Plan.current();
      return shell("Studio", "Writes into the same plan as Build a bot.", `
        <div class="grid-2">
          <section class="card"><div class="card-h"><h2>Reward</h2></div><div class="card-b" style="display:grid;gap:12px">
            <div class="slider"><label>Trading incentive · <b class="num">${p.tradeIncentive}</b></label><input id="ti" type="range" min="0" max="2" step="0.05" value="${p.tradeIncentive}"></div>
            <div class="slider"><label>Hold penalty · <b class="num">${p.holdPenalty}</b></label><input id="hp" type="range" min="-0.5" max="0" step="0.01" value="${p.holdPenalty}"></div>
            <p>The bot is paid ${p.tradeIncentive} for acting and ${p.holdPenalty} for sitting still.</p>
          </div></section>
          <section class="card"><div class="card-h"><h2>Features and curriculum</h2></div><div class="card-b">
            <p>RSI, MACD, Bollinger width, ATR, SPY context, volume ratio. The curriculum starts on trend days, then adds mean-revert, then high vol.</p>
            <p class="muted">Changing this does not start a run. Review it on Build a bot.</p>
            <a class="btn btn-sm" href="training.html">Open the plan</a>
          </div></section>
        </div>`);
    },
    oversight() {
      return shell("Oversight", "Approve a paper promotion. Live routing is not in this build.", `
        <a class="status-strip" href="report.html"><span class="pill warn"><i></i>2 of 5</span><span>Exam is in. Paper days are not.</span><b>Report card</b></a>
        <section class="card"><div class="card-h"><h2>Waiting on you</h2></div><div class="card-b" id="queue" style="display:grid;gap:12px"></div></section>`);
    },
    broker() {
      return shell("Broker", "Paper is connected. Live stays locked.", `
        <div class="grid-2">
          <section class="card"><div class="card-h"><h2>Alpaca paper</h2><span class="pill up"><i></i>Connected</span></div><div class="card-b" style="display:grid;gap:8px">
            <p>Key scope: paper trading only. The secret is not in the browser.</p>
            <button class="btn" id="health">Check connection</button>
            <p id="health-out" class="muted"></p>
          </div></section>
          <section class="card"><div class="card-h"><h2>Live</h2><span class="pill down"><i></i>Locked</span></div><div class="card-b"><p>Live routing opens only after Release passes and Oversight approves it. Clawbot cannot flip this.</p><a class="btn btn-sm" href="release.html">Open Release</a></div></section>
        </div>`);
    },
    audit() {
      return shell("Audit", "Who changed a plan, a limit, or a key.", `
        <div style="display:flex;gap:8px;flex-wrap:wrap"><input id="q" placeholder="Filter by actor or action" aria-label="Filter audit" style="padding:8px 10px;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:inherit"><button class="btn btn-sm" id="export">Export CSV</button></div>
        <section class="card"><div class="card-b tbl-wrap"><table class="tbl"><thead><tr><th>When</th><th>Who</th><th>What</th><th>Subject</th></tr></thead><tbody id="arows"></tbody></table></div></section>
        <section class="card" id="diff" hidden><div class="card-h"><h2>Diff</h2></div><div class="card-b" id="diff-body"></div></section>`);
    },
    release() {
      const gates = [
        ["Exam finished", "Deterministic card is stored.", true, "Sharpe 0.81"],
        ["Train / exam gap", "Gap is 2.1 points, under the caution line of 3.", true, "2.1"],
        ["Paper days", "2 of 5 quiet paper sessions are done.", false, "2 / 5"],
        ["Drawdown inside halt", "Session drawdown is −1.8% against a 25% halt.", true, "−1.8%"],
        ["Oversight sign-off", "A person still has to approve paper promotion.", false, "Waiting"],
      ];
      const simple = `<ol style="display:grid;gap:10px;padding-left:18px">${gates.map((g, i) => `<li><b>${g[0]}</b> — ${g[1]} <span class="pill ${g[2] ? "up" : "warn"}">${g[2] ? "Done" : "Open"}</span></li>`).join("")}</ol>`;
      const table = `<table class="tbl"><thead><tr><th>Gate</th><th>Detail</th><th class="r">Value</th><th>Status</th></tr></thead><tbody>${gates.map((g) => `<tr><td>${g[0]}</td><td>${g[1]}</td><td class="r num">${g[3]}</td><td>${g[2] ? "Pass" : "Fail"}</td></tr>`).join("")}</tbody></table>`;
      return shell("Release", "Five gates. Live routing is not available.", `
        <a class="status-strip" href="report.html"><span class="pill gold"><i></i>B+</span><span>Exam +2.1% vs buy-and-hold. Gap 2.1.</span><b>Report</b></a>
        <section class="card" data-mode-only="simple"><div class="card-h"><h2>Guided checklist</h2></div><div class="card-b">${simple}<p>Finish the open rows, then ask Oversight. There is no go-live button.</p></div></section>
        <section class="card" data-mode-only="pro"><div class="card-h"><h2>Gate table</h2></div><div class="card-b tbl-wrap">${table}</div></section>`);
    },
  };

  const auditRows = [
    { at: "Today 15:02", who: "Faisal A.", what: "Exam stored", subject: "run_1842", diff: "No plan change." },
    { at: "Today 14:11", who: "Clawbot", what: "Plan patch applied", subject: "cap", diff: "Position cap 15% → 8% after you pressed Apply." },
    { at: "Today 09:40", who: "Faisal A.", what: "Quick Train queued", subject: "Balanced", diff: "Budget quick · 20,000 steps · seed 42." },
  ];

  function render() {
    const fn = views[view];
    if (!fn) { main.innerHTML = "<p>Unknown page.</p>"; return; }
    main.innerHTML = fn();
    PR.hydrate(main);
    PR.stampMode();
    bind();
  }

  function bind() {
    const secs = document.getElementById("secs");
    if (secs) secs.onclick = (e) => {
      const b = e.target.closest("[data-sec]");
      if (!b) return;
      const u = new URL(location.href);
      if (b.dataset.sec === "All") u.searchParams.delete("sec"); else u.searchParams.set("sec", b.dataset.sec);
      history.replaceState(null, "", u);
      params.set("sec", b.dataset.sec === "All" ? "" : b.dataset.sec);
      if (b.dataset.sec === "All") params.delete("sec");
      render();
    };
    main.onclick = (e) => {
      const sym = e.target.closest("[data-sym]");
      if (sym && view === "markets") {
        const t = sym.dataset.sym;
        const d = day(t);
        PR.openSheet(`<h2 style="margin:0">${t}</h2><p class="${PR.cls(d)} num">${d >= 0 ? "+" : ""}${d.toFixed(1)}% · ${SEC[t] || ""}</p><p data-mode-only="pro">Regime tilt follows the book: trend 0.62. No order is suggested from this tile.</p><a class="btn btn-sm" href="explainer.html">Decision explainer</a>`, t);
      }
    };
    ["cap", "stop", "halt"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.oninput = () => {
        Plan.set({ [id]: +el.value });
        const label = el.parentElement.querySelector("label");
        if (label) label.textContent = (id === "cap" ? "Cap " : id === "stop" ? "Trailing stop " : "Session halt ") + el.value + "%";
      };
    });
    const kill = document.getElementById("kill");
    if (kill) kill.onclick = () => PR.toast("Kill-switch test passed. Open paper orders would be cancelled. Live routing is still off.");
    const unlock = document.getElementById("unlock");
    if (unlock) unlock.onclick = () => PR.setMode("pro");
    const ti = document.getElementById("ti");
    const hp = document.getElementById("hp");
    if (ti) ti.oninput = () => { Plan.set({ tradeIncentive: +ti.value }); ti.parentElement.querySelector("b").textContent = ti.value; };
    if (hp) hp.oninput = () => { Plan.set({ holdPenalty: +hp.value }); hp.parentElement.querySelector("b").textContent = hp.value; };
    const health = document.getElementById("health");
    if (health) health.onclick = async () => {
      const res = await PR.api.health();
      document.getElementById("health-out").textContent = res.ok ? "Paper endpoint answered. Scope is paper trading only." : "No answer.";
    };
    if (view === "oversight") {
      const items = [
        { t: "Promote ckpt_8 to the paper desk", d: "The exam is acceptable. Five quiet paper days are not done.", plain: "Let this bot keep paper-trading the book it already trades." },
        { t: "Keep the entropy boost", d: "Coach raised entropy at step 12,288 after a HOLD collapse.", plain: "Let the bot keep exploring instead of freezing." },
      ];
      const qEl = document.getElementById("queue");
      const drawQ = () => {
        qEl.innerHTML = items.map((item, i) => `<article class="health info"><div style="flex:1"><b>${PR.isPro() ? item.t : item.plain}</b><div class="muted">${PR.isPro() ? item.d : ""}</div>
          <textarea data-reason="${i}" rows="2" placeholder="Reason" style="width:100%;margin-top:8px;border-radius:10px;border:1px solid var(--line);background:var(--bg-2);color:inherit;padding:8px"></textarea>
          <div style="display:flex;gap:8px;margin-top:8px"><button class="btn btn-sm btn-gold" data-yes="${i}">Approve</button><button class="btn btn-sm" data-no="${i}">Reject</button></div></div></article>`).join("") || `<p class="muted">Queue clear.</p>`;
      };
      drawQ();
      qEl.onclick = (e) => {
        const yes = e.target.closest("[data-yes]");
        const no = e.target.closest("[data-no]");
        const idx = yes ? +yes.dataset.yes : no ? +no.dataset.no : -1;
        if (idx < 0) return;
        const reason = qEl.querySelector(`[data-reason="${idx}"]`).value.trim();
        if (!reason) { PR.toast("Add a reason first."); return; }
        items.splice(idx, 1);
        PR.toast((yes ? "Approved. " : "Rejected. ") + "Recorded in the audit trail.");
        drawQ();
      };
    }
    if (view === "audit") {
      const qEl = document.getElementById("q");
      const body = document.getElementById("arows");
      const drawA = () => {
        const f = (qEl.value || "").toLowerCase();
        body.innerHTML = auditRows.map((r, i) => ({ r, i })).filter((x) => (x.r.who + x.r.what + x.r.subject).toLowerCase().includes(f)).map((x) => `<tr data-row="${x.i}" style="cursor:pointer"><td>${x.r.at}</td><td>${x.r.who}</td><td>${x.r.what}</td><td>${x.r.subject}</td></tr>`).join("");
      };
      drawA();
      qEl.oninput = drawA;
      body.onclick = (e) => {
        const tr = e.target.closest("[data-row]");
        if (!tr) return;
        const row = auditRows[+tr.dataset.row];
        const box = document.getElementById("diff");
        box.hidden = false;
        document.getElementById("diff-body").textContent = row.diff;
      };
      document.getElementById("export").onclick = () => PR.download("audit.csv", "when,who,what,subject\n" + auditRows.map((r) => [r.at, r.who, r.what, r.subject].join(",")).join("\n"), "text/csv");
    }
    if (view === "agent") {
      const log = document.getElementById("alog");
      const prompts = ["make it more cautious", "cap each name at 8%", "drop energy stocks", "/estimate"];
      document.getElementById("prompts").innerHTML = prompts.map((p) => `<button class="pill" data-prompt="${p}">${p}</button>`).join("") + `<button class="pill" id="save-prompt">Save last question</button>`;
      const saved = JSON.parse(localStorage.getItem("pr-prompts") || "[]");
      saved.forEach((p) => { document.getElementById("prompts").insertAdjacentHTML("beforeend", `<button class="pill" data-prompt="${p}">${p}</button>`); });
      const push = (who, html) => { log.insertAdjacentHTML("beforeend", `<div class="ev"><small>${who}</small>${html}</div>`); log.scrollTop = log.scrollHeight; };
      push("Clawbot", "Ask for a plan change. I show a diff and wait. I cannot go live or place an order.");
      async function ask(text) {
        push("You", text);
        const proposal = await PR.api.propose(text);
        push("Clawbot", Plan.proposalHtml(proposal));
        log.dataset.last = text;
      }
      document.getElementById("aform").onsubmit = (e) => { e.preventDefault(); const v = document.getElementById("ain").value.trim(); if (!v) return; document.getElementById("ain").value = ""; ask(v); };
      document.getElementById("prompts").onclick = (e) => {
        const b = e.target.closest("[data-prompt]");
        if (b) ask(b.dataset.prompt);
        if (e.target.id === "save-prompt" && log.dataset.last) {
          const all = JSON.parse(localStorage.getItem("pr-prompts") || "[]");
          all.push(log.dataset.last);
          localStorage.setItem("pr-prompts", JSON.stringify(all.slice(-8)));
          PR.toast("Saved that prompt on this desk.");
        }
      };
      log.onclick = (e) => {
        const apply = e.target.closest("[data-apply]");
        if (!apply) return;
        Plan.applyProposal(apply.dataset.apply);
        push("Clawbot", "Applied. Undo it from Build a bot.");
      };
    }
  }

  document.addEventListener("pr:mode", () => render());
  render();
})();
