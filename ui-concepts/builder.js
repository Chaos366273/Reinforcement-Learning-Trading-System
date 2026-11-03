/* Build a bot: wizard, editable plan, and Clawbot proposals. */
(function () {
  const q = new URLSearchParams(location.search);
  if (q.get("template")) Plan.preset(q.get("template"));
  if (q.get("from")) Plan.set({ branchedFrom: q.get("from") }, "form");
  if (q.get("retrain") === "cautious") Plan.preset("cautious");

  const proSteps = ["Universe", "Style", "Guardrails", "Teach", "Budget", "Review"];
  const simpleSteps = ["Style", "Risk", "Budget", "Review"];
  let step = 0;

  function steps() { return PR.isPro() ? proSteps : simpleSteps; }
  function kind() { return steps()[Math.min(step, steps().length - 1)]; }

  function chips(selected) {
    return `<div style="display:flex;flex-wrap:wrap;gap:6px">${Plan.tickers.map((t) => `<button class="pill ${selected.includes(t) ? "gold" : ""}" data-tick="${t}" aria-pressed="${selected.includes(t)}">${t}</button>`).join("")}</div>`;
  }
  function choice(id, title, body, pressed, attr) {
    return `<button class="choice" ${attr}="${id}" aria-pressed="${pressed}"><b>${title}</b><span>${body}</span></button>`;
  }

  function wizard(p) {
    const k = kind();
    const err = Plan.validate(p);
    const banner = err.length && k !== "Review" ? "" : "";
    if (k === "Universe") {
      return `<h2>Which names can it touch?</h2><p class="muted">At least 4, at most 30. Clawbot can add or drop names if you ask.</p>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
          <button class="btn btn-sm" data-universe="tech">Big Tech</button>
          <button class="btn btn-sm" data-universe="def">Defensive</button>
          <button class="btn btn-sm" data-universe="all">Mixed 30</button>
        </div>${chips(p.names)}`;
    }
    if (k === "Style") {
      return `<h2>How should it behave?</h2><div class="grid-3">
        ${choice("cautious", "Cautious", "Fewer trades. Cap locks at 8%.", p.style === "cautious", "data-style")}
        ${choice("balanced", "Balanced", "The research default. Cap 15%.", p.style === "balanced", "data-style")}
        ${choice("bold", "Bold", "Pays more for taking a trade.", p.style === "bold", "data-style")}
      </div><p class="pro-only muted">Hold penalty ${p.holdPenalty} · trading incentive ${p.tradeIncentive}. Edit these in Studio or the plan.</p>`;
    }
    if (k === "Risk") {
      return `<h2>How much room does it get?</h2>
        <div class="slider"><label>Risk comfort · <b class="num">${p.risk} / 5</b></label>
        <input type="range" min="1" max="5" step="1" value="${p.risk}" data-risk></div>
        <p class="muted">That sets cap ${p.cap}%, trailing stop ${p.stop}%, and a session halt at ${p.halt}%.</p>
        <div class="lock-note">${PR.icon("risk", 16)} Protected by PathRise. Simple mode does not edit the three limits separately.</div>`;
    }
    if (k === "Guardrails") {
      return `<h2>Guardrails</h2>
        ${[["cap", "Largest single name", p.cap, 5, 15], ["stop", "Trailing stop", p.stop, 4, 12], ["halt", "Session halt", p.halt, 8, 25]].map((row) => `<div class="slider"><label>${row[1]} · <b class="num">${row[2]}%</b></label><input type="range" min="${row[3]}" max="${row[4]}" value="${row[2]}" data-field="${row[0]}"></div>`).join("")}
        <p class="muted">Cautious style will not let the cap rise past 8%.</p>`;
    }
    if (k === "Teach") {
      return `<h2>Teach it first?</h2><div class="grid-3">
        ${choice("rules", "Rules mentor", "About 70% holds. A conservative teacher.", p.mentor === "rules", "data-mentor")}
        ${choice("trend", "Trend mentor", "Acts when MACD and volume agree.", p.mentor === "trend", "data-mentor")}
        ${choice("none", "Skip", "PPO starts from scratch and may freeze on HOLD.", p.mentor === "none", "data-mentor")}
      </div>`;
    }
    if (k === "Budget") {
      return `<h2>How long should it practice?</h2><div class="budget">
        ${Object.entries(Plan.budgets).map(([id, b]) => choice(id, b.label, b.steps.toLocaleString() + " steps · " + b.credits + " credits base", p.budget === id, "data-budget")).join("")}
      </div>
      <div class="drawer pro-only open" style="margin-top:12px">
        ${[["Learning rate", "lr"], ["Rollout steps", "nSteps"], ["Batch", "batch"], ["Gamma", "gamma"], ["Entropy coef", "entropy"], ["Clip range", "clip"], ["TCN depth", "tcnDepth"], ["Seed", "seed"]].map(([label, key]) => `<div class="field"><label>${label}</label><input class="num" data-h="${key}" value="${p[key]}"></div>`).join("")}
      </div>`;
    }
    const e = Plan.estimate(p);
    const delta = Plan.diffs(Plan.lastRun, p);
    const problems = Plan.validate(p);
    return `<h2>Review and start</h2>
      ${p.fromChat && p.fromChat.length ? `<div class="banner info">Clawbot edited ${p.fromChat.length} field${p.fromChat.length > 1 ? "s" : ""}. Gold marks in the plan are still unconfirmed by you — they are already in this draft.</div>` : ""}
      ${p.branchedFrom ? `<div class="banner info">Branched from <b class="num">${p.branchedFrom}</b>. The new run keeps that checkpoint as the warm start.</div>` : ""}
      ${problems.map((x) => `<div class="banner" role="alert">${x}</div>`).join("")}
      <p>Compared with the last finished run (${Plan.lastRun.names.length} names, ${Plan.lastRun.style}).</p>
      ${delta.length ? delta.map((d) => `<div class="plan"><div class="row"><span class="muted">${d.label}</span><b>${Plan.show(d.from)} → ${Plan.show(d.to)}</b></div></div>`).join("") : `<p class="muted">No difference from the last run. You can still start a fresh seed.</p>`}
      <p><b class="num">${e.credits} credits</b> · ${e.steps.toLocaleString()} steps · about ${e.minutes} min.</p>
      <p class="muted">Starting freezes this plan. The worker reads it once and does not take live edits.</p>`;
  }

  function planCard(p) {
    const e = Plan.estimate(p);
    const mark = (field) => (p.fromChat || []).includes(field) ? ` <span class="changed">Clawbot</span>` : "";
    const simple = !PR.isPro();
    const row = (field, control) => `<div class="row"><span class="muted">${(Plan.schema[field] && Plan.schema[field].label) || field}${mark(field)}</span><span>${control}</span></div>`;
    const reset = (field) => `<button class="btn btn-sm" data-reset="${field}" aria-label="Reset ${field}">Reset</button>`;
    return `<div class="card-h"><h2>Training plan</h2><span class="pill gold" id="cost">${e.credits} credits</span></div>
      <div class="card-b plan">
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button class="btn btn-sm" id="undo" ${Plan.canUndo() ? "" : "disabled"}>Undo</button>
          <button class="btn btn-sm" id="redo" ${Plan.canRedo() ? "" : "disabled"}>Redo</button>
          <button class="btn btn-sm" id="save-ver">Save version</button>
        </div>
        ${row("style", `<span class="seg">${["cautious", "balanced", "bold"].map((s) => `<button data-style="${s}" aria-pressed="${p.style === s}">${s[0].toUpperCase()}${s.slice(1)}</button>`).join("")}</span>`)}
        ${simple ? row("risk", `<input type="range" min="1" max="5" value="${p.risk}" data-risk style="width:120px"> <b class="num">${p.risk}</b>`) : ""}
        ${row("budget", `<span class="seg">${["quick", "standard", "deep"].map((s) => `<button data-budget="${s}" aria-pressed="${p.budget === s}">${s[0].toUpperCase()}${s.slice(1)}</button>`).join("")}</span>`)}
        ${row("names", `<b>${p.names.length} selected</b> ${reset("names")}`)}
        ${simple ? `<div class="lock-note" style="margin-top:8px">${PR.icon("risk", 16)} <span id="limits">Cap ${p.cap}% · stop ${p.stop}% · halt ${p.halt}%. Protected by PathRise.</span></div>` : `
          ${row("cap", `<b class="num">${p.cap}%</b> ${reset("cap")}`)}
          ${row("stop", `<b class="num">${p.stop}%</b> ${reset("stop")}`)}
          ${row("halt", `<b class="num">${p.halt}%</b> ${reset("halt")}`)}
          ${row("mentor", `<b>${p.mentor}</b>`)}
          ${row("lr", `<b class="num">${p.lr}</b>`)}
          ${row("entropy", `<b class="num">${p.entropy}</b>`)}
          ${row("seed", `<b class="num">${p.seed}</b>`)}
        `}
        <div class="row"><span class="muted">Steps</span><b class="num">${e.steps.toLocaleString()}</b></div>
        <div class="row"><span class="muted">Time</span><b>about ${e.minutes} min</b></div>
        <p class="muted" style="margin:8px 0 0">Ask Clawbot to change this. Nothing moves until you press Apply.</p>
      </div>`;
  }

  function paint() {
    if (step >= steps().length) step = steps().length - 1;
    const p = Plan.current();
    document.getElementById("pips").innerHTML = steps().map((n, i) => `<span class="step-pip ${i < step ? "done" : ""} ${i === step ? "on" : ""}"><span class="n">${i + 1}</span>${n}</span>`).join("");
    document.getElementById("step").innerHTML = wizard(p);
    document.getElementById("plan").innerHTML = planCard(p);
    const last = kind() === "Review";
    document.getElementById("go").textContent = last ? "Start training" : "Continue";
    document.getElementById("go").disabled = last && Plan.validate(p).length > 0;
    PR.hydrate(document.getElementById("plan"));
    PR.stampMode();
    bindPlan();
  }
  function bindPlan() {
    const u = document.getElementById("undo");
    const r = document.getElementById("redo");
    if (u) u.onclick = () => { Plan.undo(); paint(); };
    if (r) r.onclick = () => { Plan.redo(); paint(); };
    const sv = document.getElementById("save-ver");
    if (sv) sv.onclick = () => {
      const name = prompt("Name this version", Plan.current().versionName || "Draft");
      if (!name) return;
      Plan.saveVersion(name);
      PR.toast("Saved “" + name + "”. The draft is still editable.");
    };
  }
  function refreshPlan() {
    const p = Plan.current();
    const limits = document.getElementById("limits");
    if (limits) limits.textContent = `Cap ${p.cap}% · stop ${p.stop}% · halt ${p.halt}%. Protected by PathRise.`;
    const cost = document.getElementById("cost");
    if (cost) cost.textContent = Plan.estimate(p).credits + " credits";
  }

  const tech = ["NVDA", "MSFT", "AAPL", "GOOGL", "META", "AMD", "CRM", "ORCL"];
  const defensive = ["JNJ", "UNH", "PFE", "LLY", "ABBV", "KO", "WMT", "COST"];

  document.getElementById("step").addEventListener("click", (e) => {
    const t = e.target.closest("[data-tick]");
    if (t) {
      const p = Plan.current();
      const names = p.names.includes(t.dataset.tick) ? p.names.filter((n) => n !== t.dataset.tick) : p.names.concat(t.dataset.tick);
      Plan.set({ names: names });
      paint();
    }
    const uni = e.target.closest("[data-universe]");
    if (uni) {
      const map = { tech: tech, def: defensive, all: Plan.tickers };
      Plan.set({ names: map[uni.dataset.universe].slice() });
      paint();
    }
    const st = e.target.closest("[data-style]");
    if (st) { Plan.preset(st.dataset.style); paint(); }
    const m = e.target.closest("[data-mentor]");
    if (m) { Plan.set({ mentor: m.dataset.mentor }); paint(); }
    const b = e.target.closest("[data-budget]");
    if (b) { Plan.set({ budget: b.dataset.budget }); paint(); }
  });
  document.getElementById("step").addEventListener("input", (e) => {
    if (e.target.dataset.risk != null) {
      Plan.setRisk(+e.target.value);
      const b = e.target.previousElementSibling && e.target.previousElementSibling.querySelector("b");
      if (b) b.textContent = e.target.value + " / 5";
    }
    if (e.target.dataset.field) {
      Plan.set({ [e.target.dataset.field]: +e.target.value });
      const b = e.target.previousElementSibling && e.target.previousElementSibling.querySelector("b");
      if (b) b.textContent = e.target.value + "%";
    }
    if (e.target.dataset.h) Plan.set({ [e.target.dataset.h]: Number(e.target.value) });
    refreshPlan();
  });
  document.getElementById("plan").addEventListener("click", (e) => {
    const st = e.target.closest("[data-style]");
    if (st) { Plan.preset(st.dataset.style); paint(); }
    const b = e.target.closest("[data-budget]");
    if (b) { Plan.set({ budget: b.dataset.budget }); paint(); }
    const reset = e.target.closest("[data-reset]");
    if (reset) { Plan.resetField(reset.dataset.reset); paint(); }
  });
  document.getElementById("plan").addEventListener("input", (e) => {
    if (e.target.dataset.risk != null) {
      Plan.setRisk(+e.target.value);
      const b = e.target.parentElement.querySelector("b");
      if (b) b.textContent = e.target.value;
      refreshPlan();
    }
  });

  document.getElementById("back").onclick = () => { if (step === 0) location.href = "desk.html"; else { step -= 1; paint(); } };
  document.getElementById("draft").onclick = () => { Plan.saveVersion("Draft"); PR.toast("Draft saved on this desk. Nothing has started training."); };
  document.getElementById("go").onclick = async () => {
    const p = Plan.current();
    if (kind() === "Universe" && p.names.length < 4) { PR.toast("Pick at least 4 names."); return; }
    if (kind() !== "Review") { step += 1; paint(); return; }
    if (Plan.validate(p).length) return;
    const run = await PR.api.startRun();
    sessionStorage.setItem("pr-plan", JSON.stringify(p));
    location.href = "arena.html?start=1&run=" + run.id;
  };

  document.addEventListener("pr:mode", () => { step = 0; paint(); });
  document.addEventListener("pr:plan", () => { if (!document.getElementById("step")) return; });

  PR.onChat = async (text) => {
    const proposal = await PR.api.propose(text);
    const html = Plan.proposalHtml(proposal);
    const follow = `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px"><button class="pill" data-suggest="make it more cautious">Make it cautious</button><button class="pill" data-suggest="cap each name at 8%">Cap at 8%</button><button class="pill" data-suggest="train longer but spend under 40 credits">Longer, under 40 credits</button></div>`;
    return { html: html + follow };
  };

  document.addEventListener("click", async (e) => {
    const apply = e.target.closest("[data-apply]");
    if (apply) {
      const res = Plan.applyProposal(apply.dataset.apply);
      if (!res) { PR.toast("That proposal expired."); return; }
      paint();
      PR.pushChat(`Applied. <button class="btn btn-sm" id="chat-undo">Undo</button>`, "Clawbot");
      return;
    }
    if (e.target.closest("[data-undo-proposal]")) {
      Plan.dismiss(e.target.closest("[data-undo-proposal]").dataset.undoProposal);
      e.target.closest(".proposal").remove();
      return;
    }
    if (e.target.id === "chat-undo") { Plan.undo(); paint(); PR.toast("Undid the last plan change."); }
    const sug = e.target.closest("[data-suggest]");
    if (sug) {
      const input = document.getElementById("composer-input");
      if (input) { input.value = sug.dataset.suggest; document.getElementById("composer").requestSubmit(); }
    }
  });

  PR.seedChat(`<div class="msg"><div class="who">Clawbot</div>I can edit this training plan. Try “make it more cautious”, “drop energy stocks”, or <code>/set cap 8</code>. I will show a diff and wait for Apply. I cannot go live or place orders.</div>`);
  document.body.classList.add("dock-open");
  paint();
})();
