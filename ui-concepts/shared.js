/* PathRise design samples — shared shell, dock, charts and dummy data.
   Plain JS on purpose: open any concept file directly, no build step. */
(function () {
  const PR = (window.PR = {});
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  PR.$ = $; PR.$$ = $$;

  /* ---------- icons (24px, stroke) ---------- */
  const ICONS = {
    desk: '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    markets: '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    portfolio: '<path d="M21 12A9 9 0 1 1 12 3v9z"/><path d="M15.5 3.6A9 9 0 0 1 20.4 8.5H15.5z"/>',
    risk: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    training: '<rect x="5" y="5" width="14" height="14" rx="2.5"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 1.5V5M15 1.5V5M9 19v3.5M15 19v3.5M1.5 9H5M1.5 15H5M19 9h3.5M19 15h3.5"/>',
    oversight: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    broker: '<path d="M3 21h18M5 21V10M9.5 21V10M14.5 21V10M19 21V10M2.5 10L12 4l9.5 6z"/>',
    agent: '<rect x="4" y="8" width="16" height="12" rx="3.5"/><path d="M12 8V5M9 14h.01M15 14h.01M9.5 17.2c1.5 1 3.5 1 5 0"/><circle cx="12" cy="3.6" r="1.2"/>',
    studio: '<path d="M9 3h6M10 3v6.5L4.3 19a2 2 0 0 0 1.7 3h12a2 2 0 0 0 1.7-3L14 9.5V3"/><path d="M7.5 15h9"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    audit: '<path d="M14 3H6.5A2 2 0 0 0 4.5 5v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V8.5z"/><path d="M14 3v5.5h5.5M9 15l2 2 4-4"/>',
    release: '<path d="M14 4c3-1 6-1 7-1 0 1 0 4-1 7l-7 7-6-6z"/><path d="M8 14c-2 .5-3.5 2-4 5 3-.5 4.5-2 5-4"/><circle cx="15" cy="9" r="1.4"/>',
    search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
    moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    chevl: '<path d="M15 6l-6 6 6 6"/>', chevr: '<path d="M9 6l6 6-6 6"/>', chevd: '<path d="M6 9l6 6 6-6"/>',
    menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
    send: '<path d="M5 12l14-7-5 14-2.5-5.5z"/>',
    x: '<path d="M6 6l12 12M18 6L6 18"/>',
    back: '<path d="M10 6l-6 6 6 6M4 12h16"/>',
    next: '<path d="M14 6l6 6-6 6M20 12H4"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    alert: '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.5M12 17.6h.01"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    bolt: '<path d="M13 2.5L4.5 13.5H11l-1 8 8.5-11H12z"/>',
    chat: '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8a2.5 2.5 0 0 1-2.5 2.5H10l-5 4v-4A2.5 2.5 0 0 1 4 13.5z"/>',
    wallet: '<path d="M3.5 7.5A2.5 2.5 0 0 1 6 5h12v3"/><path d="M3.5 7.5v10A2.5 2.5 0 0 0 6 20h13.5a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1H6a2.5 2.5 0 0 1-2.5-1.5z"/><circle cx="16.5" cy="14.5" r="1"/>',
    inbox: '<path d="M3 13l2.5-8h13L21 13v6a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/><path d="M3 13h5l1 2.5h6l1-2.5h5"/>',
    refresh: '<path d="M20 11a8 8 0 0 0-14.5-4M4 4v4h4M4 13a8 8 0 0 0 14.5 4M20 20v-4h-4"/>',
    layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 17l.7 1.8L21.5 19.5l-1.8.7L19 22l-.7-1.8-1.8-.7 1.8-.7z"/>',
    target: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
  };
  PR.icon = (n, s = 18) => `<svg class="ic" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[n] || ""}</svg>`;
  PR.hydrate = (root = document) => $$("[data-i]", root).forEach((el) => { el.innerHTML = PR.icon(el.dataset.i, +el.dataset.s || 18); });

  /* ---------- formatting ---------- */
  PR.usd = (n, d = 2) => (n < 0 ? "-" : "") + "$" + Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  PR.sgn = (n, d = 2) => (n >= 0 ? "+" : "-") + Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  PR.pct = (n, d = 2) => (n >= 0 ? "+" : "-") + Math.abs(n).toFixed(d) + "%";
  PR.cls = (n) => (n >= 0 ? "up" : "down");

  /* ---------- dummy data (values echo the README: 30-ticker universe, gating tiers, regimes) ---------- */
  PR.data = {
    equity: 104812.36, startEquity: 100000, dayPnl: 1284.2, cash: 61240.55, buyingPower: 122481.1,
    positions: [
      { s: "NVDA", n: "NVIDIA", sec: "Tech", q: 60, avg: 118.4, last: 124.9, held: "5h 47m", stop: 115.28 },
      { s: "MSFT", n: "Microsoft", sec: "Tech", q: 25, avg: 412.1, last: 419.35, held: "1d 2h", stop: 394.7 },
      { s: "LLY", n: "Eli Lilly", sec: "Healthcare", q: 8, avg: 782.3, last: 801.1, held: "2d 4h", stop: 761.2 },
      { s: "COST", n: "Costco", sec: "Consumer", q: 6, avg: 905.2, last: 931.4, held: "3d 1h", stop: 880.9 },
      { s: "JPM", n: "JPMorgan Chase", sec: "Financials", q: 40, avg: 201.55, last: 198.2, held: "3h 12m", stop: 183.6 },
      { s: "XOM", n: "Exxon Mobil", sec: "Energy", q: 55, avg: 112.8, last: 110.95, held: "6h 05m", stop: 102.1 },
    ],
    trades: [
      { id: 1042, t: "09:48", a: "BUY", s: "NVDA", q: 60, p: 118.4, conf: 0.31, tier: "High", why: "Momentum + volume spike" },
      { id: 1041, t: "11:20", a: "SELL", s: "AAPL", q: 30, p: 231.15, conf: 0.27, tier: "High", why: "Profit-taking near upper band", pnl: 212.4 },
      { id: 1040, t: "12:05", a: "BUY", s: "JPM", q: 40, p: 201.55, conf: 0.14, tier: "Medium", why: "Mean-reversion setup" },
      { id: 1039, t: "13:30", a: "SELL", s: "AMD", q: 45, p: 158.9, conf: 0.22, tier: "Medium", why: "Trailing stop tightened", pnl: -96.3 },
      { id: 1038, t: "14:10", a: "BUY", s: "XOM", q: 55, p: 112.8, conf: 0.12, tier: "Medium", why: "Oversold RSI, SPY steady" },
      { id: 1037, t: "15:02", a: "HOLD", s: "META", q: 0, p: 0, conf: 0.04, tier: "Low", why: "Gate blocked: cooldown active" },
    ],
    tickers: "AAPL MSFT NVDA GOOGL META AMD JPM V BAC MA AMZN WMT HD NKE KO COST JNJ UNH PFE LLY ABBV XOM CVX VZ T DIS INTC QCOM ORCL CRM".split(" "),
  };
  PR.data.positions.forEach((p) => { p.pnl = (p.last - p.avg) * p.q; p.pnlPct = (p.last / p.avg - 1) * 100; p.val = p.last * p.q; });

  /* ---------- deterministic RNG so every reload looks identical ---------- */
  PR.rng = (seed) => () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  PR.hash = (str) => { let h = 2166136261; for (const c of str) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; };

  PR.series = (n, seed, start, end, vol) => {
    const r = PR.rng(seed); const v = [start];
    for (let i = 1; i < n; i++) v.push(v[i - 1] * (1 + (r() - 0.48) * vol));
    const diff = end - v[n - 1];
    return v.map((x, i) => x + (diff * i) / (n - 1));
  };

  /* ---------- theme + mode + shell behaviours ---------- */
  const store = { get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
  const root = document.documentElement;
  const qs = new URLSearchParams(location.search);
  const startTheme = qs.get("theme") || store.get("pr-theme", "dark");
  if (startTheme === "light") root.dataset.theme = "light";
  document.body.dataset.mode = qs.get("mode") || store.get("pr-mode", "beginner");
  document.body.dataset.state = qs.get("state") || "live";
  if (qs.has("still")) { const st = document.createElement("style"); st.textContent = "*,*::before,*::after{animation:none!important;transition:none!important}"; document.head.append(st); }
  if (qs.get("dock") === "open") document.body.classList.add("dock-open");
  if (qs.get("rail") === "collapsed" || store.get("pr-rail", "") === "collapsed") document.body.classList.add("rail-collapsed");

  PR.setTheme = (t) => { if (t === "light") root.dataset.theme = "light"; else delete root.dataset.theme; store.set("pr-theme", t); document.dispatchEvent(new CustomEvent("pr:theme")); syncThemeButtons(); };
  function syncThemeButtons() {
    const light = root.dataset.theme === "light";
    $$("[data-theme-toggle]").forEach((b) => { b.innerHTML = PR.icon(light ? "moon" : "sun", 17); b.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme"); b.title = b.getAttribute("aria-label"); });
  }
  PR.setMode = (m) => { document.body.dataset.mode = m; store.set("pr-mode", m); $$("[data-mode-set]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.modeSet === m)); document.dispatchEvent(new CustomEvent("pr:mode")); };
  PR.setState = (s) => { document.body.dataset.state = s; $$("[data-state-set]").forEach((b) => b.classList.toggle("on", b.dataset.stateSet === s)); document.dispatchEvent(new CustomEvent("pr:state")); };

  let toastT;
  PR.toast = (msg) => { let t = $(".toast"); if (!t) { t = document.createElement("div"); t.className = "toast"; t.setAttribute("role", "status"); document.body.append(t); } t.textContent = msg; t.classList.add("show"); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 2600); };

  /* ---------- shell: left rail ---------- */
  const NAV = [
    ["Workspace", [["desk", "Desk", "concept-a-desk.html"], ["markets", "Markets"], ["portfolio", "Portfolio"], ["risk", "Risk"]]],
    ["Intelligence", [["training", "Training"], ["oversight", "Oversight"], ["agent", "Agent"], ["studio", "Studio"]]],
    ["Operations", [["broker", "Broker"], ["audit", "Audit"], ["release", "Release"]]],
  ];
  PR.rail = (active = "desk") => {
    const groups = NAV.map(([g, items]) => `<div class="rail-group">${g}</div>` + items.map(([ic, label, href]) =>
      `<a class="rail-link${ic === active ? " is-active" : ""}" href="${href || "#"}" ${href ? "" : `data-soon="${label}"`} ${ic === active ? 'aria-current="page"' : ""} title="${label}">${PR.icon(ic, 19)}<span class="rail-label">${label}</span>${ic === "agent" ? '<i class="dot" title="Agent online"></i>' : ""}</a>`).join("")).join("");
    return `<aside class="rail" id="rail" aria-label="Primary">
      <div class="rail-brand"><span class="logo">${PR.icon("bolt", 18).replace('stroke-width="1.7"', 'stroke-width="2.2"')}</span><span class="rail-label brand-name">PathRise<small>Desk</small></span>
        <button class="rail-toggle" data-rail-toggle aria-label="Collapse sidebar">${PR.icon("chevl", 16)}</button></div>
      <nav class="rail-nav">${groups}</nav>
      <div class="rail-foot"><span class="avatar" title="Session: FA">FA</span><div class="session"><b>Faisal A.</b><span><i></i>Paper session</span></div>
        <button class="icon-btn" data-soon="Settings" aria-label="Open Settings" title="Settings">${PR.icon("settings", 18)}</button></div>
    </aside>`;
  };

  /* ---------- shell: Clawbot dock ---------- */
  const TOOLS = [["/explain", "Explain a trade or decision"], ["/backtest", "Replay the agent on a date range"], ["/compare", "Compare two agent versions"], ["/risk", "Adjust risk limits (paper only)"], ["/report", "Open the latest unified report"]];
  const AGENTS = [["@Clawbot", "General assistant"], ["@Strategist", "Reads signals and regimes"], ["@RiskOfficer", "Stops, sizing, drawdown"], ["@Executor", "Order routing (paper)"]];
  PR.dock = () => `
    <button class="dock-handle" data-dock-open aria-expanded="false" aria-controls="dock"><span class="bot">${PR.icon("spark", 15)}</span><span class="handle-long">Message Clawbot</span><span class="handle-short">Clawbot</span><span class="badge">1</span></button>
    <section class="dock" id="dock" aria-label="Clawbot chat" role="complementary">
      <div class="dock-grip" data-dock-grip role="separator" aria-orientation="horizontal" aria-label="Resize chat" tabindex="0"></div>
      <div class="dock-h"><span class="bot">${PR.icon("spark", 15)}</span><div><b>Clawbot</b><br><small>Knows your paper book, the agent and today's regime</small></div>
        <span class="top-spacer"></span><button class="icon-btn" data-dock-close aria-label="Collapse chat" title="Collapse (Esc)">${PR.icon("chevd", 18)}</button></div>
      <div class="dock-msgs" id="dock-msgs" aria-live="polite">
        <div class="msg"><div class="who">Clawbot</div>NVDA was bought at 09:48 with <b>0.31</b> signal confidence — above the 0.25 high-confidence gate. Want the full breakdown? Try <code>/explain 1042</code>.</div>
        <div class="msg me">Why is the agent holding META?</div>
        <div class="msg"><div class="who">Clawbot</div>Gate blocked it: confidence 0.04 is below the low tier and the cooldown was still active. Nothing is wrong — it is waiting for a cleaner setup.</div>
      </div>
      <form class="composer" id="composer" autocomplete="off">
        <div class="menu" id="menu" role="listbox"></div>
        <textarea id="composer-input" rows="1" placeholder="Ask anything — type / for tools, @ for agents" aria-label="Message Clawbot"></textarea>
        <span class="kbd hide-sm">/ tools</span><span class="kbd hide-sm">@ agents</span>
        <button class="btn btn-gold btn-sm" type="submit" aria-label="Send">${PR.icon("send", 15)}Send</button>
      </form>
    </section>`;

  function initDock() {
    const dock = $("#dock"); if (!dock) return;
    const open = (v) => { document.body.classList.toggle("dock-open", v); $("[data-dock-open]").setAttribute("aria-expanded", v); if (v) setTimeout(() => $("#composer-input").focus(), 350); };
    PR.openDock = (prefill) => { open(true); if (prefill) { const i = $("#composer-input"); i.value = prefill; i.focus(); } };
    $("[data-dock-open]").onclick = () => open(true);
    $("[data-dock-close]").onclick = () => open(false);
    const saved = +store.get("pr-dock-h", 340); document.documentElement.style.setProperty("--dock-h", saved + "px");
    const grip = $("[data-dock-grip]");
    const setH = (h) => { h = Math.max(220, Math.min(innerHeight * 0.75, h)); document.documentElement.style.setProperty("--dock-h", h + "px"); store.set("pr-dock-h", Math.round(h)); };
    grip.addEventListener("pointerdown", (e) => { grip.setPointerCapture(e.pointerId); dock.classList.add("dragging"); dock.style.transition = "none"; });
    grip.addEventListener("pointermove", (e) => { if (grip.hasPointerCapture(e.pointerId)) setH(innerHeight - e.clientY); });
    grip.addEventListener("pointerup", (e) => { grip.releasePointerCapture(e.pointerId); dock.classList.remove("dragging"); dock.style.transition = ""; });
    grip.addEventListener("keydown", (e) => { const h = dock.offsetHeight; if (e.key === "ArrowUp") { setH(h + 24); e.preventDefault(); } if (e.key === "ArrowDown") { setH(h - 24); e.preventDefault(); } });

    const input = $("#composer-input"), menu = $("#menu"), msgs = $("#dock-msgs");
    let items = [], sel = 0, tokenRe = /(^|\s)([\/@])([\w-]*)$/;
    const closeMenu = () => { menu.classList.remove("open"); items = []; };
    const renderMenu = () => { menu.innerHTML = items.map((it, i) => `<button type="button" role="option" class="${i === sel ? "sel" : ""}" data-idx="${i}"><b>${it[0]}</b><span>${it[1]}</span></button>`).join(""); menu.classList.toggle("open", items.length > 0); };
    const update = () => {
      input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 120) + "px";
      const m = input.value.slice(0, input.selectionStart).match(tokenRe);
      if (!m) return closeMenu();
      const list = m[2] === "/" ? TOOLS : AGENTS; const q = (m[2] + m[3]).toLowerCase();
      items = list.filter((x) => x[0].toLowerCase().startsWith(q)); sel = 0; renderMenu();
    };
    const choose = (i) => { const it = items[i]; if (!it) return; const pos = input.selectionStart; const before = input.value.slice(0, pos).replace(tokenRe, (_, s) => s + it[0] + " "); input.value = before + input.value.slice(pos); input.setSelectionRange(before.length, before.length); closeMenu(); input.focus(); };
    input.addEventListener("input", update);
    menu.addEventListener("mousedown", (e) => { const b = e.target.closest("button"); if (b) { e.preventDefault(); choose(+b.dataset.idx); } });
    input.addEventListener("keydown", (e) => {
      if (menu.classList.contains("open")) {
        if (e.key === "ArrowDown") { sel = (sel + 1) % items.length; renderMenu(); e.preventDefault(); }
        else if (e.key === "ArrowUp") { sel = (sel - 1 + items.length) % items.length; renderMenu(); e.preventDefault(); }
        else if (e.key === "Enter" || e.key === "Tab") { choose(sel); e.preventDefault(); }
        else if (e.key === "Escape") { closeMenu(); e.stopPropagation(); }
      } else if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
    });
    const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    $("#composer").addEventListener("submit", (e) => {
      e.preventDefault(); const v = input.value.trim(); if (!v) return;
      msgs.insertAdjacentHTML("beforeend", `<div class="msg me">${esc(v)}</div><div class="msg" id="typing"><span class="typing"><i></i><i></i><i></i></span></div>`);
      input.value = ""; update(); msgs.scrollTop = msgs.scrollHeight;
      setTimeout(() => { const t = $("#typing"); if (t) { t.removeAttribute("id"); t.innerHTML = `<div class="who">Clawbot</div>` + (/^\/explain/.test(v) ? "Opening the decision breakdown — confidence cleared the 0.25 gate, driven mostly by volume (1.7× average) and rising MACD." : "On it. In the live product this reaches your agents and tools; this sample only shows the interaction.") ; msgs.scrollTop = msgs.scrollHeight; } }, 900);
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && document.body.classList.contains("dock-open") && !menu.classList.contains("open")) open(false); if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "j") { e.preventDefault(); open(!document.body.classList.contains("dock-open")); } });
  }

  /* ---------- charts ---------- */
  const ns = "http://www.w3.org/2000/svg";
  PR.lineChart = (el, o) => {
    const draw = () => {
      const W = el.clientWidth || 600, H = o.height || 280, pad = { l: 56, r: 14, t: 14, b: 26 };
      const all = o.series.flatMap((s) => s.data); let lo = Math.min(...all), hi = Math.max(...all); const sp = (hi - lo) || 1; lo -= sp * 0.08; hi += sp * 0.08;
      const n = o.series[0].data.length, x = (i) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r), y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
      const css = getComputedStyle(document.documentElement); const col = (v) => css.getPropertyValue(v).trim();
      let g = "";
      for (let k = 0; k <= 4; k++) { const v = lo + ((hi - lo) * k) / 4, yy = y(v); g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${yy}" y2="${yy}" stroke="${col("--line")}" /><text x="${pad.l - 10}" y="${yy + 4}" text-anchor="end" fill="${col("--muted")}" font-size="10.5" font-family="JetBrains Mono, monospace">${o.fmt ? o.fmt(v) : v.toFixed(0)}</text>`; }
      (o.xlabels || []).forEach((l, k, a) => { g += `<text x="${pad.l + (k / (a.length - 1)) * (W - pad.l - pad.r)}" y="${H - 6}" text-anchor="${k === 0 ? "start" : k === a.length - 1 ? "end" : "middle"}" fill="${col("--muted")}" font-size="10.5">${l}</text>`; });
      const defs = `<defs><linearGradient id="${o.id}-a" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${col(o.series[0].color)}" stop-opacity=".28"/><stop offset="1" stop-color="${col(o.series[0].color)}" stop-opacity="0"/></linearGradient></defs>`;
      const paths = o.series.map((s, si) => {
        const d = s.data.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
        const area = si === 0 && o.area !== false ? `<path d="${d}L${x(n - 1)},${H - pad.b}L${x(0)},${H - pad.b}Z" fill="url(#${o.id}-a)"/>` : "";
        return area + `<path d="${d}" fill="none" stroke="${col(s.color)}" stroke-width="${s.w || 2}" ${s.dash ? 'stroke-dasharray="4 4"' : ""} stroke-linejoin="round" stroke-linecap="round"/>`;
      }).join("");
      const marks = (o.markers || []).map((m) => { const cx = x(m.i), cy = y(o.series[0].data[m.i]); const c = col(m.a === "BUY" ? "--up" : "--down"); const up = m.a === "BUY";
        return `<g class="mk" data-tip="${m.a} ${m.s || ""} ${m.p ? "@ " + m.p : ""}" transform="translate(${cx},${cy})"><circle r="9" fill="${c}" opacity=".16"/><path d="${up ? "M0,-5 L5,4 L-5,4Z" : "M0,5 L5,-4 L-5,-4Z"}" fill="${c}"/></g>`; }).join("");
      el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${o.label || "Chart"}">${defs}${g}${paths}${marks}<line class="xh" y1="${pad.t}" y2="${H - pad.b}" stroke="${col("--gold")}" stroke-dasharray="3 3" opacity="0"/><circle class="xd" r="4.5" fill="${col(o.series[0].color)}" stroke="${col("--panel")}" stroke-width="2" opacity="0"/><rect class="hit" x="${pad.l}" y="${pad.t}" width="${W - pad.l - pad.r}" height="${H - pad.t - pad.b}" fill="transparent"/></svg><div class="tooltip"></div>`;
      const svg = $("svg", el), tip = $(".tooltip", el), xh = $(".xh", el), xd = $(".xd", el);
      $(".hit", el).addEventListener("pointermove", (e) => {
        const r = svg.getBoundingClientRect(), px = e.clientX - r.left, i = Math.max(0, Math.min(n - 1, Math.round(((px - pad.l) / (W - pad.l - pad.r)) * (n - 1))));
        xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i)); xh.setAttribute("opacity", 1); xd.setAttribute("cx", x(i)); xd.setAttribute("cy", y(o.series[0].data[i])); xd.setAttribute("opacity", 1);
        tip.innerHTML = (o.tip ? o.tip(i) : "") ; tip.style.opacity = 1; const tw = tip.offsetWidth; tip.style.left = Math.min(W - tw - 6, Math.max(6, x(i) + 14)) + "px"; tip.style.top = "8px";
      });
      $(".hit", el).addEventListener("pointerleave", () => { xh.setAttribute("opacity", 0); xd.setAttribute("opacity", 0); tip.style.opacity = 0; });
    };
    draw(); new ResizeObserver(() => draw()).observe(el); document.addEventListener("pr:theme", draw);
    return draw;
  };
  PR.spark = (data, w = 84, h = 28, color = "var(--up)") => {
    const lo = Math.min(...data), hi = Math.max(...data), n = data.length;
    const d = data.map((v, i) => `${i ? "L" : "M"}${((i / (n - 1)) * w).toFixed(1)},${(h - 2 - ((v - lo) / (hi - lo || 1)) * (h - 4)).toFixed(1)}`).join("");
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><path d="${d}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
  };

  /* ---------- boot ---------- */
  PR.mount = (opts = {}) => {
    const app = $(".app");
    if (opts.rail !== false && app && !$("#rail")) app.insertAdjacentHTML("afterbegin", PR.rail(opts.active));
    if (opts.dock !== false) document.body.insertAdjacentHTML("beforeend", PR.dock());
    PR.hydrate(); syncThemeButtons(); initDock();
    $$("[data-rail-toggle]").forEach((b) => b.addEventListener("click", () => { document.body.classList.toggle("rail-collapsed"); store.set("pr-rail", document.body.classList.contains("rail-collapsed") ? "collapsed" : ""); document.dispatchEvent(new Event("pr:theme")); }));
    $$("[data-theme-toggle]").forEach((b) => b.addEventListener("click", () => PR.setTheme(root.dataset.theme === "light" ? "dark" : "light")));
    $$("[data-mode-set]").forEach((b) => { b.setAttribute("aria-pressed", b.dataset.modeSet === document.body.dataset.mode); b.addEventListener("click", () => PR.setMode(b.dataset.modeSet)); });
    $$("[data-state-set]").forEach((b) => { b.classList.toggle("on", b.dataset.stateSet === document.body.dataset.state); b.addEventListener("click", () => PR.setState(b.dataset.stateSet)); });
    $$("[data-menu-btn]").forEach((b) => b.addEventListener("click", () => document.body.classList.toggle("nav-open")));
    document.addEventListener("click", (e) => {
      if (document.body.classList.contains("nav-open") && !e.target.closest(".rail") && !e.target.closest("[data-menu-btn]")) document.body.classList.remove("nav-open");
      const soon = e.target.closest("[data-soon]"); if (soon) { e.preventDefault(); PR.toast(`${soon.dataset.soon} isn't part of these samples — Desk, Cockpit and Explainer are.`); }
      const pb = e.target.closest("[data-pop]"); $$(".pop.open").forEach((p) => { if (!pb || p !== $("#" + pb.dataset.pop)) p.classList.remove("open"); }); if (pb) $("#" + pb.dataset.pop).classList.toggle("open");
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") $$(".pop.open").forEach((p) => p.classList.remove("open")); if (e.key === "Escape") document.body.classList.remove("nav-open"); });
    $$("[data-search]").forEach((s) => s.addEventListener("click", () => PR.toast("Command palette: jump to a ticker, trade or report (⌘K).")));
    document.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); PR.toast("Command palette: jump to a ticker, trade or report (⌘K)."); } });
  };
})();
