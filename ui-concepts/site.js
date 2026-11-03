/* PathRise product shell on top of shared.js (Concept B cockpit). */
(function () {
  const PR = window.PR;
  const PRIMARY = [
    ["desk", "Desk", "desk.html"],
    ["training", "Training", "training.html"],
    ["markets", "Markets", "markets.html"],
    ["portfolio", "Portfolio", "portfolio.html"],
    ["risk", "Risk", "risk.html"],
    ["agent", "Agent", "agent.html"],
    ["studio", "Studio", "studio.html"],
    ["oversight", "Oversight", "oversight.html"],
  ];
  const MORE = [
    ["broker", "Broker", "broker.html"],
    ["audit", "Audit", "audit.html"],
    ["release", "Release", "release.html"],
  ];
  const BOTS = [
    ["balanced", "Balanced", "Paper · Quick Train"],
    ["cautious", "Cautious", "Paper · Deep Train"],
    ["demo", "Demo bot", "Read only"],
  ];

  PR.bot = () => { try { return localStorage.getItem("pr-bot") || "balanced"; } catch { return "balanced"; } };
  PR.setBot = (id) => { try { localStorage.setItem("pr-bot", id); } catch {} document.dispatchEvent(new CustomEvent("pr:bot")); };

  PR.shell = (active) => {
    const bot = BOTS.find((b) => b[0] === PR.bot()) || BOTS[0];
    const tabs = PRIMARY.map(([ic, label, href]) =>
      `<a class="tab${ic === active ? " is-active" : ""}" href="${href}" ${ic === active ? 'aria-current="page"' : ""}>${PR.icon(ic === "training" ? "training" : ic, 16)}<span class="lbl">${label}</span></a>`).join("");
    const moreOn = MORE.some((m) => m[0] === active);
    return `<header class="topnav">
      <div class="has-pop">
        <button class="bot-btn" data-pop="bot-pop" aria-haspopup="true" aria-label="Switch bot">
          <span class="logo">${PR.icon("bolt", 16)}</span>
          <span><b>PathRise</b><small id="bot-label">${bot[1]} · ${bot[2]}</small></span>
          <span data-i="chevd" data-s="14"></span>
        </button>
        <div class="pop menu-pop" id="bot-pop" role="menu">
          <div class="eyebrow">Your bots</div>
          ${BOTS.map((b) => `<button class="rowlink" data-bot="${b[0]}"><span data-i="agent" data-s="16"></span><span><b>${b[1]}</b><br><span class="muted">${b[2]}</span></span></button>`).join("")}
          <a href="training.html"><span data-i="plus" data-s="16"></span>Build a new bot</a>
        </div>
      </div>
      <nav class="tabs" aria-label="Primary">${tabs}</nav>
      <div class="nav-right">
        <div class="has-pop">
          <button class="btn btn-sm ${moreOn ? "btn-gold" : ""}" data-pop="more-pop" aria-haspopup="true">More</button>
          <div class="pop menu-pop" id="more-pop">
            ${MORE.map((m) => `<a href="${m[2]}">${PR.icon(m[0], 16)} ${m[1]}</a>`).join("")}
          </div>
        </div>
        <div class="seg" role="group" aria-label="Detail level"><button data-mode-set="beginner">Simple</button><button data-mode-set="pro">Pro</button></div>
        <button class="icon-btn" data-theme-toggle></button>
        <div class="has-pop">
          <button class="avatar" data-pop="me-pop" aria-label="Account menu" title="Faisal A.">FA</button>
          <div class="pop menu-pop" id="me-pop">
            <div class="eyebrow">Faisal A. · paper</div>
            <a href="settings.html">${PR.icon("settings", 16)} Settings</a>
            <a href="oversight.html">${PR.icon("oversight", 16)} Go-live status</a>
            <button class="rowlink" data-soon="Sign out">${PR.icon("back", 16)} Sign out</button>
          </div>
        </div>
      </div>
    </header>`;
  };

  PR.publicNav = (active) => `<header class="pub-nav">
    <a class="rail-brand" style="padding:0;min-height:0" href="index.html"><span class="logo">${PR.icon("bolt", 16)}</span><span class="brand-name">PathRise<small>Train your own bot</small></span></a>
    <nav><a href="how.html">How it works</a><a href="pricing.html">Pricing</a><a href="gallery.html">Samples</a></nav>
    <span style="flex:1"></span>
    <a class="btn btn-sm" href="login.html">Log in</a>
    <a class="btn btn-sm btn-gold" href="signup.html">Start training</a>
    <button class="icon-btn" data-theme-toggle></button>
  </header>`;

  PR.authFrame = (opts) => `<div class="split">
    <section class="auth">
      <a class="rail-brand" style="padding:0;min-height:0" href="index.html"><span class="logo">${PR.icon("bolt", 16)}</span><span class="brand-name">PathRise<small>Paper only</small></span></a>
      <div><h1>${opts.title}</h1><p class="lead">${opts.lead}</p></div>
      ${opts.banner || ""}
      ${opts.body}
      <div class="muted" style="font-size:12.5px">${opts.foot || ""}</div>
    </section>
    <aside class="ambient" aria-hidden="true"><div class="card float-card"><div class="card-h"><h2>A bot learning</h2><span class="pill gold"><i></i>Replay</span></div><div class="card-b"><div class="chart" id="ambient-chart"></div><p class="muted" style="margin:8px 0 0;font-size:12.5px">Recorded Quick Train. Reward climbs as HOLD share falls. Simulated, not a promise.</p></div></div></aside>
  </div>`;

  PR.bootSite = (opts = {}) => {
    document.body.classList.add("site");
    if (opts.shell) {
      const app = document.querySelector(".app");
      app.insertAdjacentHTML("afterbegin", PR.shell(opts.active || "desk"));
    }
    if (opts.dock === false) PR.mount({ rail: false, dock: false });
    else PR.mount({ rail: false, dock: opts.agent ? false : true });
    document.querySelectorAll("[data-bot]").forEach((b) => b.addEventListener("click", () => {
      PR.setBot(b.dataset.bot);
      const meta = BOTS.find((x) => x[0] === b.dataset.bot);
      const lab = document.getElementById("bot-label");
      if (lab && meta) lab.textContent = meta[1] + " · " + meta[2];
      document.querySelectorAll(".pop.open").forEach((p) => p.classList.remove("open"));
      PR.toast("Switched to " + meta[1] + ". Every page now shows this bot.");
    }));
    document.querySelectorAll("[data-toggle-pw]").forEach((b) => b.addEventListener("click", () => {
      const input = document.getElementById(b.dataset.togglePw);
      input.type = input.type === "password" ? "text" : "password";
      b.textContent = input.type === "password" ? "Show" : "Hide";
    }));
    const amb = document.getElementById("ambient-chart");
    if (amb && PR.lineChart) {
      const a = PR.series(48, 3, -0.8, 1.4, 0.08), b = PR.series(48, 9, -0.2, 0.35, 0.05);
      PR.lineChart(amb, { id: "amb", height: 180, series: [{ data: a, color: "--gold", w: 2 }, { data: b, color: "--hold", w: 1.4, dash: true }], xlabels: ["Start", "Mid", "Now"], fmt: (v) => v.toFixed(1), tip: (i) => `<div class="num">Reward ${a[i].toFixed(2)}</div>` });
    }
  };

  PR.gate = (rows) => rows.map((r) => `<div class="chk"><span class="ico" style="width:26px;height:26px;border-radius:50%;display:grid;place-items:center;background:${r.ok ? "var(--up-soft)" : "var(--panel-3)"};color:${r.ok ? "var(--up)" : "var(--muted)"}">${PR.icon(r.ok ? "check" : "clock", 14)}</span><div><b>${r.t}</b><small style="display:block;color:var(--muted)">${r.d}</small></div><span class="num pro-only">${r.v}</span></div>`).join("");
})();
