/* Single training plan. The builder, Clawbot tools, and api.js all call this. */
(function (root) {
  const FIELDS = {
    style: { label: "Style", enum: ["cautious", "balanced", "bold"] },
    risk: { label: "Risk comfort", min: 1, max: 5 },
    budget: { label: "Budget", enum: ["quick", "standard", "deep"] },
    cap: { label: "Position cap", min: 5, max: 15, unit: "%" },
    stop: { label: "Trailing stop", min: 4, max: 12, unit: "%" },
    halt: { label: "Session halt", min: 8, max: 25, unit: "%" },
    mentor: { label: "Mentor", enum: ["rules", "trend", "none"] },
    lr: { label: "Learning rate", min: 0.00005, max: 0.001, pro: true },
    nSteps: { label: "Rollout steps", min: 512, max: 8192, pro: true },
    batch: { label: "Batch", min: 64, max: 2048, pro: true },
    gamma: { label: "Gamma", min: 0.9, max: 0.999, pro: true },
    entropy: { label: "Entropy coef", min: 0, max: 0.1, pro: true },
    clip: { label: "Clip range", min: 0.05, max: 0.3, pro: true },
    tcnDepth: { label: "TCN depth", min: 2, max: 6, pro: true },
    seed: { label: "Seed", min: 0, max: 99999, pro: true },
    holdPenalty: { label: "Hold penalty", min: -0.5, max: 0, pro: true },
    tradeIncentive: { label: "Trading incentive", min: 0, max: 2, pro: true },
  };
  const ALIAS = { dd: "halt", drawdown: "halt", n_steps: "nSteps", nsteps: "nSteps", ent: "entropy", ent_coef: "entropy", learning_rate: "lr", tcn: "tcnDepth" };
  const TICKERS = "AAPL MSFT NVDA GOOGL META AMD JPM V BAC MA AMZN WMT HD NKE KO COST JNJ UNH PFE LLY ABBV XOM CVX VZ T DIS INTC QCOM ORCL CRM".split(" ");
  const ENERGY = ["XOM", "CVX"];
  const PRESETS = {
    cautious: { style: "cautious", risk: 2, cap: 8, stop: 6, halt: 12, holdPenalty: -0.03, tradeIncentive: 0.2 },
    balanced: { style: "balanced", risk: 4, cap: 15, stop: 8, halt: 25, holdPenalty: -0.1, tradeIncentive: 0.4 },
    bold: { style: "bold", risk: 5, cap: 15, stop: 10, halt: 22, holdPenalty: -0.2, tradeIncentive: 1.5 },
  };
  const RISK = {
    1: { cap: 5, stop: 4, halt: 8 },
    2: { cap: 8, stop: 6, halt: 12 },
    3: { cap: 12, stop: 8, halt: 18 },
    4: { cap: 15, stop: 8, halt: 22 },
    5: { cap: 15, stop: 10, halt: 25 },
  };
  const BUDGET = {
    quick: { label: "Quick", credits: 0, steps: 20000, minutes: 4 },
    standard: { label: "Standard", credits: 12, steps: 100000, minutes: 28 },
    deep: { label: "Deep", credits: 40, steps: 500000, minutes: 140 },
  };
  const HELP = {
    style: "Cautious trades less and locks the cap at 8%. Balanced is the research default. Bold pays the bot more for taking a trade.",
    risk: "Risk comfort from 1 to 5 sets the cap, trailing stop, and session halt together. Pro can still edit each limit.",
    budget: "Quick is free and finishes in a few minutes. Standard is 12 credits. Deep is 40 credits before the universe-size adjustment.",
    cap: "The most any one name can be of the paper book. The research default is 15%. PathRise will not accept a cap above 15%.",
    stop: "A winner is sold if it gives back this much from its high. A hard stop sits wider, near 10%.",
    halt: "New buys stop if the paper book falls this far in the session.",
    mentor: "Rules is a conservative teacher. Trend only acts when MACD and volume agree. None starts PPO from scratch.",
    lr: "How large each PPO update is. Too high and approx_kl spikes. The research default is 0.0003.",
    entropy: "How much the bot is paid to keep its options open. The coach raised this mid-run when HOLD collapsed.",
    names: "At least 4 symbols from the 30-name universe. One name just memorizes that chart.",
  };
  const LAST_RUN = {
    style: "balanced", risk: 4, budget: "quick",
    names: ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "AMZN", "AVGO"],
    cap: 15, stop: 8, halt: 25, mentor: "rules",
    lr: 0.0003, nSteps: 2048, batch: 512, gamma: 0.995, entropy: 0.02, clip: 0.15,
    tcnDepth: 4, seed: 42, holdPenalty: -0.1, tradeIncentive: 0.4,
  };

  const mem = { get: (k, d) => d, set: () => {} };
  try {
    if (root.localStorage) {
      mem.get = (k, d) => { try { return root.localStorage.getItem(k) ?? d; } catch { return d; } };
      mem.set = (k, v) => { try { root.localStorage.setItem(k, v); } catch {} };
    }
  } catch { /* private mode */ }

  function fresh() {
    return {
      style: "balanced", risk: 4, budget: "quick",
      names: ["NVDA", "MSFT", "AAPL", "GOOGL", "META", "AMD", "AMZN", "CRM"],
      cap: 15, stop: 8, halt: 25, mentor: "rules",
      lr: 0.0003, nSteps: 2048, batch: 512, gamma: 0.995, entropy: 0.02, clip: 0.15,
      tcnDepth: 4, seed: 42, holdPenalty: -0.1, tradeIncentive: 0.4,
      fromChat: [], versionName: "Draft", branchedFrom: "",
    };
  }
  function emit() {
    if (typeof document !== "undefined") document.dispatchEvent(new CustomEvent("pr:plan"));
  }
  let current = fresh();
  let undo = [];
  let redoStack = [];
  const pending = new Map();

  function load() {
    try {
      const raw = mem.get("pr-plan-v2", "");
      if (!raw) return;
      const parsed = JSON.parse(raw);
      current = { ...fresh(), ...parsed, names: parsed.names || fresh().names, fromChat: parsed.fromChat || [] };
    } catch { current = fresh(); }
  }
  function save() { mem.set("pr-plan-v2", JSON.stringify(current)); }
  load();

  function keyOf(name) {
    const n = String(name || "").toLowerCase();
    return ALIAS[n] || n;
  }
  function show(v) {
    if (Array.isArray(v)) return v.length + " names";
    if (typeof v === "number") {
      if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
      return String(Math.round(v * 10000) / 10000);
    }
    return String(v);
  }
  function clampInto(draft) {
    const clamped = [];
    Object.entries(FIELDS).forEach(([k, meta]) => {
      if (draft[k] == null) return;
      if (meta.enum && !meta.enum.includes(draft[k])) {
        clamped.push({ field: k, from: draft[k], to: current[k], reason: meta.label + " must be " + meta.enum.join(", ") + "." });
        draft[k] = current[k];
      }
      if (meta.min != null && typeof draft[k] === "number") {
        const c = Math.min(meta.max, Math.max(meta.min, draft[k]));
        if (c !== draft[k]) clamped.push({ field: k, from: draft[k], to: c, reason: meta.label + " stays between " + meta.min + " and " + meta.max + "." });
        draft[k] = c;
      }
    });
    if (draft.style === "cautious" && draft.cap > 8) {
      clamped.push({ field: "cap", from: draft.cap, to: 8, reason: "Cautious style locks the cap at 8%." });
      draft.cap = 8;
    }
    if (Array.isArray(draft.names)) {
      const uniq = [...new Set(draft.names.map((n) => String(n).toUpperCase()).filter((n) => TICKERS.includes(n)))];
      if (uniq.length !== draft.names.length) clamped.push({ field: "names", from: draft.names.join(","), to: uniq.join(","), reason: "Unknown symbols were dropped." });
      draft.names = uniq.slice(0, 30);
    }
    return clamped;
  }
  function commit(next, source, touched) {
    undo.push(JSON.stringify(current));
    if (undo.length > 50) undo.shift();
    redoStack = [];
    const fromChat = new Set(current.fromChat || []);
    (touched || []).forEach((k) => { if (source === "clawbot") fromChat.add(k); else fromChat.delete(k); });
    current = { ...next, fromChat: [...fromChat] };
    save();
    emit();
    return current;
  }
  function applyPatch(patch, source) {
    const draft = { ...current, ...patch, names: patch.names ? patch.names.slice() : current.names.slice() };
    const clamped = clampInto(draft);
    commit(draft, source, Object.keys(patch));
    return { plan: current, clamped };
  }
  function estimate(plan) {
    const p = plan || current;
    const b = BUDGET[p.budget] || BUDGET.quick;
    const scale = Math.max(1, (p.names ? p.names.length : 8) / 8);
    return {
      label: b.label,
      steps: b.steps,
      credits: b.credits === 0 ? 0 : Math.round(b.credits * scale),
      minutes: Math.max(2, Math.round(b.minutes * scale)),
    };
  }
  function diffs(a, b) {
    const keys = ["style", "risk", "budget", "names", "cap", "stop", "halt", "mentor", "lr", "nSteps", "batch", "gamma", "entropy", "clip", "tcnDepth", "seed", "holdPenalty", "tradeIncentive"];
    return keys.filter((k) => JSON.stringify(a[k]) !== JSON.stringify(b[k])).map((k) => ({
      field: k, label: (FIELDS[k] && FIELDS[k].label) || k, from: a[k], to: b[k],
    }));
  }
  function preview(patch) {
    const draft = { ...current, ...patch, names: patch.names ? patch.names.slice() : current.names.slice() };
    const clamped = clampInto(draft);
    return { draft, clamped, diffs: diffs(current, draft) };
  }
  function explain(field) {
    const k = keyOf(field || "");
    return HELP[k] || "I can change style, risk, budget, the universe, caps, stops, the mentor, and in Pro the PPO hyperparameters. Try “make it more cautious” or /set cap 8.";
  }
  function propose(text) {
    const raw = String(text || "").trim();
    const t = raw.toLowerCase();
    if (!raw) return { refused: true, message: "Tell me what to change." };
    if (/\b(go live|go-live|place an order|place order|buy now|sell now|broker key|api key|alpaca secret)\b/.test(t)) {
      return { refused: true, message: "I can't go live, place orders, or change broker settings. Those stay on Oversight and Release." };
    }
    if (/^\/explain\b/.test(t)) return { explain: explain(t.split(/\s+/)[1]) };
    if (t === "/estimate" || /how (much|long)|what does this cost/.test(t)) {
      const e = estimate();
      return { explain: e.label + " · " + e.steps.toLocaleString() + " steps · about " + e.minutes + " min · " + e.credits + " credits." };
    }
    const patch = {};
    const notes = [];
    const presetSlash = t.match(/^\/preset\s+(cautious|balanced|bold|aggressive)/);
    if (presetSlash) Object.assign(patch, PRESETS[presetSlash[1] === "aggressive" ? "bold" : presetSlash[1]]);
    const setM = raw.match(/^\/set\s+([a-z_]+)\s+([0-9.]+)/i);
    if (setM) {
      const k = keyOf(setM[1]);
      if (!FIELDS[k]) return { refused: true, message: "I don't have a field called " + setM[1] + "." };
      patch[k] = Number(setM[2]);
    }
    if (/\b(more cautious|safer|careful|cautious)\b/.test(t)) Object.assign(patch, PRESETS.cautious);
    if (/\b(more aggressive|bolder|bold|aggressive)\b/.test(t)) Object.assign(patch, PRESETS.bold);
    if (/\bbalanced\b/.test(t) && !patch.style) Object.assign(patch, PRESETS.balanced);
    const capM = t.match(/\bcap(?: each name| per name)?(?: at| to| of)?\s+(\d+(?:\.\d+)?)\s*%?/);
    if (capM) patch.cap = Number(capM[1]);
    const stopM = t.match(/\bstop(?: at| to)?\s+(\d+(?:\.\d+)?)\s*%?/);
    if (stopM) patch.stop = Number(stopM[1]);
    const haltM = t.match(/\bhalt(?: at| to)?\s+(\d+(?:\.\d+)?)\s*%?/);
    if (haltM) patch.halt = Number(haltM[1]);
    const riskM = t.match(/\brisk(?: comfort)?(?: at| to)?\s+([1-5])\b/);
    if (riskM) {
      const level = Number(riskM[1]);
      Object.assign(patch, { risk: level, ...RISK[level] });
    }
    if (/drop energy|remove energy|no energy|without energy/.test(t)) {
      const before = patch.names || current.names;
      const next = before.filter((n) => !ENERGY.includes(n));
      if (next.length === before.length) notes.push("No energy names are in this plan.");
      else { patch.names = next; notes.push("Energy names leave the universe."); }
    }
    const drop = t.match(/\b(?:drop|remove)\s+([a-z]{1,5})\b/);
    if (drop && drop[1] !== "energy") {
      const sym = drop[1].toUpperCase();
      if (TICKERS.includes(sym)) patch.names = (patch.names || current.names).filter((n) => n !== sym);
      else notes.push(sym + " is not in the 30-name universe.");
    }
    const add = t.match(/\badd\s+([a-z]{1,5})\b/);
    if (add) {
      const sym = add[1].toUpperCase();
      if (TICKERS.includes(sym)) patch.names = [...new Set([...(patch.names || current.names), sym])];
    }
    if (/rules mentor|mentor rules|\brules\b/.test(t) && /mentor|teacher|teach/.test(t)) patch.mentor = "rules";
    if (/trend mentor|\btrend\b/.test(t) && /mentor|teacher|teach/.test(t)) patch.mentor = "trend";
    if (/no mentor|skip mentor|from scratch/.test(t)) patch.mentor = "none";
    if (/train longer|more steps|deep train|longer run/.test(t)) patch.budget = "deep";
    if (/\bstandard run\b|\bstandard budget\b/.test(t)) patch.budget = "standard";
    if (/quick run|shorter|2-minute|two-minute|preview/.test(t)) patch.budget = "quick";
    const under = t.match(/under\s+(\d+)\s+credits/);
    if (under) {
      const max = Number(under[1]);
      let choice = "quick";
      ["quick", "standard", "deep"].forEach((id) => {
        const trial = { ...current, ...patch, budget: id };
        if (estimate(trial).credits <= max) choice = id;
      });
      if (patch.budget && estimate({ ...current, ...patch, names: patch.names || current.names }).credits > max) {
        notes.push(patch.budget + " would pass " + max + " credits, so this stays on " + choice + ".");
      }
      patch.budget = choice;
    }
    if (/suggest|fill the universe|mixed 30/.test(t)) {
      patch.names = TICKERS.slice();
      notes.push("That is the full 30-name research universe.");
    }
    if (!Object.keys(patch).length) {
      return { explain: notes.join(" ") || explain("") };
    }
    const seen = preview(patch);
    if (!seen.diffs.length) {
      const extra = notes.concat(seen.clamped.map((c) => c.reason)).filter(Boolean).join(" ");
      return { explain: extra || "That already matches the plan." };
    }
    const est = estimate(seen.draft);
    const id = "p" + Date.now();
    const proposal = {
      id,
      patch,
      diffs: seen.diffs,
      clamped: seen.clamped,
      note: notes.join(" ") || ("About " + est.minutes + " min · " + est.credits + " credits."),
      summary: seen.diffs.map((d) => d.label + " " + show(d.from) + " → " + show(d.to)).join(", "),
    };
    pending.set(id, proposal);
    return proposal;
  }
  function htmlFor(proposal) {
    if (proposal.refused) return proposal.message;
    if (proposal.explain) return proposal.explain;
    const rows = proposal.diffs.map((d) => `<div class="diff"><span>${d.label}</span><b>${show(d.from)} → ${show(d.to)}</b></div>`).join("");
    const clamps = (proposal.clamped || []).map((c) => `<div class="muted">${c.reason}</div>`).join("");
    return `<div class="proposal" data-proposal="${proposal.id}"><b>Proposed change</b>${rows}<div class="muted">${proposal.note}</div>${clamps}<div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn btn-gold btn-sm" data-apply="${proposal.id}">Apply</button><button class="btn btn-sm" data-undo-proposal="${proposal.id}">Dismiss</button></div></div>`;
  }

  root.Plan = {
    schema: FIELDS,
    tickers: TICKERS,
    energy: ENERGY,
    budgets: BUDGET,
    presets: PRESETS,
    riskMap: RISK,
    lastRun: LAST_RUN,
    help: HELP,
    current: () => JSON.parse(JSON.stringify(current)),
    estimate,
    validate(plan) {
      const p = plan || current;
      const errors = [];
      if (!p.names || p.names.length < 4) errors.push("Pick at least 4 names. A one-stock bot memorizes that chart.");
      if (p.names && p.names.length > 30) errors.push("The universe stops at 30 names.");
      return errors;
    },
    set(patch, source) { return applyPatch(patch, source || "form"); },
    preset(name) {
      const p = PRESETS[name === "aggressive" ? "bold" : name];
      if (!p) return null;
      return applyPatch(p, "form");
    },
    setRisk(level) { return applyPatch({ risk: level, ...RISK[level] }, "form"); },
    resetField(field) {
      const base = PRESETS[current.style] || PRESETS.balanced;
      if (field === "names") return applyPatch({ names: fresh().names }, "form");
      if (base[field] == null && fresh()[field] == null) return current;
      return applyPatch({ [field]: base[field] != null ? base[field] : fresh()[field] }, "form");
    },
    undo() {
      if (!undo.length) return false;
      redoStack.push(JSON.stringify(current));
      current = JSON.parse(undo.pop());
      save(); emit();
      return true;
    },
    redo() {
      if (!redoStack.length) return false;
      undo.push(JSON.stringify(current));
      current = JSON.parse(redoStack.pop());
      save(); emit();
      return true;
    },
    canUndo: () => undo.length > 0,
    canRedo: () => redoStack.length > 0,
    propose,
    proposalHtml: htmlFor,
    applyProposal(id) {
      const p = pending.get(id);
      if (!p) return null;
      pending.delete(id);
      return applyPatch(p.patch, "clawbot");
    },
    dismiss(id) { pending.delete(id); },
    saveVersion(name) {
      const all = JSON.parse(mem.get("pr-plan-versions", "[]") || "[]");
      all.unshift({ name: name || "Untitled", at: new Date().toISOString(), plan: current });
      mem.set("pr-plan-versions", JSON.stringify(all.slice(0, 12)));
      current.versionName = name || current.versionName;
      save();
      return all;
    },
    versions() { return JSON.parse(mem.get("pr-plan-versions", "[]") || "[]"); },
    restore(index) {
      const all = JSON.parse(mem.get("pr-plan-versions", "[]") || "[]");
      if (!all[index]) return false;
      return commit({ ...fresh(), ...all[index].plan }, "form", Object.keys(all[index].plan));
    },
    show,
    diffs,
    replace(plan) { return commit({ ...fresh(), ...plan, names: plan.names || current.names }, "form", Object.keys(plan)); },
  };
})(typeof window !== "undefined" ? window : globalThis);
