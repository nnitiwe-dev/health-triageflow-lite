"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---- state: everything about the CURRENT patient lives here and is reset together ---------
const S = {
  runId: null,            // the run awaiting approval (a fresh one per assessment)
  state: null,            // last public state from the agent
  signs: new Map(),       // code -> "present" | "absent"  (the nurse's decisions)
  candidates: [],         // suggestions from the last extraction
  candidatesFor: null,    // complaint text those suggestions were read from
  extraction: null,       // how they were found: {mode: "model+keywords" | "keywords", note}
  baseline: null,         // {vitals, at} when re-checking a patient (deterioration watch)
  arrival: null,
  measuredAt: null,       // kept only for unedited EMR-imported vitals (staleness check)
  awaiting: false,        // the server is paused at the approval gate for runId
  stale: false,           // intake edited since the result on screen
  assessing: false,
  confirmed: false,
};
let SIGN_GROUPS = [];     // full SATS list from /api/discriminators
let reassessTimer = null;

function newId() {
  const d = new Date();
  return `OPD-${String(d.getHours()).padStart(2, "0")}${String(d.getMinutes()).padStart(2, "0")}-${Math.floor(Math.random() * 900 + 100)}`;
}
function num(id) { const v = $(id).value; return v === "" ? null : Number(v); }
function str(id) { const v = $(id).value; return v === "" ? null : v; }
const VITAL_FIELDS = { respiratory_rate: "f-rr", heart_rate: "f-hr", systolic_bp: "f-sbp",
  diastolic_bp: "f-dbp", temperature_c: "f-temp", spo2: "f-spo2", glucose_mmol: "f-glucose",
  weight_kg: "f-weight", pain_score: "f-pain" };

function readVitals() {
  const v = {};
  for (const [k, id] of Object.entries(VITAL_FIELDS)) v[k] = num(id);
  v.avpu = str("f-avpu"); v.mobility = str("f-mobility");
  const t = $("f-trauma").value; v.trauma = t === "" ? null : t === "true";
  if (S.measuredAt) v.measured_at = S.measuredAt;
  return v;
}
function readForm() {
  return {
    encounter_id: $("f-id").value || newId(),
    age_years: num("f-age"), sex: str("f-sex"),
    presenting_complaint: $("f-complaint").value || "",
    arrival_time: S.arrival,
    discriminator_codes: [],            // signs only ever come from the nurse's own ticks
    vitals: readVitals(),
  };
}

// ---- new patient / loading ------------------------------------------------------------------
function resetPatient() {
  clearTimeout(reassessTimer);
  Object.assign(S, { runId: null, state: null, candidates: [], candidatesFor: null, extraction: null, baseline: null,
    arrival: new Date().toISOString(), measuredAt: null, awaiting: false, stale: false, confirmed: false });
  S.signs = new Map();
  for (const id of ["f-age", "f-sex", "f-complaint", "f-avpu", "f-mobility", "f-trauma", ...Object.values(VITAL_FIELDS)]) $(id).value = "";
  $("f-id").value = newId();
  $("a-override").value = ""; $("a-reason").value = ""; $("a-reason").classList.add("hidden");
  $("a-checked").checked = false;
  for (const id of ["proposal", "blocked", "recheck-note", "fhir-note", "extract-note"]) $(id).classList.add("hidden");
  $("empty-proposal").classList.remove("hidden");
  $("trace").innerHTML = "";
  renderSigns();
}
function fillForm(enc, { measuredAt = null } = {}) {
  resetPatient();
  $("f-id").value = newId();
  $("f-age").value = enc.age_years ?? ""; $("f-sex").value = enc.sex ?? "";
  $("f-complaint").value = enc.presenting_complaint || "";
  const v = enc.vitals || {};
  for (const [k, id] of Object.entries(VITAL_FIELDS)) $(id).value = v[k] ?? "";
  $("f-avpu").value = v.avpu ?? ""; $("f-mobility").value = v.mobility ?? "";
  $("f-trauma").value = v.trauma == null ? "" : String(!!v.trauma);
  if (enc.arrival_time) S.arrival = enc.arrival_time;
  S.measuredAt = measuredAt;
}

// ---- signs: one source of truth, three views (emergency ticks, suggestions, full list) ------
function presentCodes() { return [...S.signs].filter(([, v]) => v === "present").map(([k]) => k); }
function absentCodes() { return [...S.signs].filter(([, v]) => v === "absent").map(([k]) => k); }
function fromVitals(code) { return S.candidates.some(c => c.code === code && c.status === "from_vitals"); }
function unreviewed() { return S.candidates.filter(c => !S.signs.has(c.code) && c.status !== "from_vitals"); }

function setSign(code, value) {
  if (value == null) S.signs.delete(code); else S.signs.set(code, value);
  renderSigns();
  // A sign changes the priority: re-score straight away. An emergency tick is SATS step 1 —
  // it must work with no vitals entered and no extra button press.
  if (S.runId || value === "present") scheduleReassess();
  else updateGate();
}
function scheduleReassess() {
  markStale();
  clearTimeout(reassessTimer);
  reassessTimer = setTimeout(() => assess(), 350);
}

function renderSigns() {
  document.querySelectorAll(".emerg-tick").forEach(el => { el.checked = S.signs.get(el.value) === "present"; });

  const box = $("candidates");
  $("candidates-empty").classList.toggle("hidden", S.candidates.length > 0);
  box.innerHTML = S.candidates.map(c => {
    const st = S.signs.get(c.code) || (c.status === "from_vitals" ? "from_vitals" : "unreviewed");
    const via = c.via === "model" ? "suggested by the model" : c.via === "vitals" ? "because of the glucose reading" : "matched in the complaint";
    const control = st === "from_vitals"
      ? `<span class="via">✓ already counted from the vitals</span>`
      : `<span class="seg"><button data-code="${esc(c.code)}" data-v="present" class="${st === "present" ? "on-yes" : ""}">Present</button><button data-code="${esc(c.code)}" data-v="absent" class="${st === "absent" ? "on-no" : ""}">Not present</button></span>`;
    return `<div class="cand ${st === "unreviewed" ? "unreviewed" : ""}">
      <div><div class="text-[13px] font-medium">${esc(c.label)} <span class="lvl band-${esc(c.implies)}">${esc(c.implies)}</span></div>
      <div class="via">${via}</div></div>${control}</div>`;
  }).join("");
  box.querySelectorAll("button[data-code]").forEach(b => b.onclick = () => {
    const cur = S.signs.get(b.dataset.code);
    setSign(b.dataset.code, cur === b.dataset.v ? null : b.dataset.v);
  });

  const n = presentCodes().length;
  $("checklist-count").textContent = n ? `· ${n} ticked` : "· tick anything the suggestions missed";
  $("checklist").innerHTML = SIGN_GROUPS.map(g => `
    <div><div class="chk-group-h qprio ${esc(g.level)}">${esc(g.title)}</div>
      <div class="grid grid-cols-1 gap-1">${g.signs.map(s => `
        <label class="tick"><input type="checkbox" class="chk" value="${esc(s.code)}" ${S.signs.get(s.code) === "present" || fromVitals(s.code) ? "checked" : ""} ${fromVitals(s.code) ? "disabled" : ""}/><span>${esc(s.label)}</span></label>`).join("")}
      </div></div>`).join("");
  $("checklist").querySelectorAll(".chk").forEach(el => el.onchange = () => setSign(el.value, el.checked ? "present" : null));
  updateGate();
}

// ---- streaming assess -----------------------------------------------------------------------
async function assess() {
  if (S.assessing) { scheduleReassess(); return; }
  clearTimeout(reassessTimer);
  const enc = readForm();
  $("f-id").value = enc.encounter_id;
  const complaint = enc.presenting_complaint.trim();
  const reuse = S.candidatesFor !== null && S.candidatesFor === complaint;   // nothing new to read
  S.assessing = true; S.awaiting = false; S.stale = false; S.confirmed = false; S.runId = null;
  $("agent-spin").classList.remove("hidden"); $("btn-assess").disabled = true;
  $("trace").innerHTML = ""; $("blocked").classList.add("hidden");
  for (const id of ["confirmed", "after", "audit", "btn-audit", "handover", "stale"]) $(id).classList.add("hidden");
  $("gate").classList.remove("hidden");
  if (!reuse) { S.candidates = []; S.candidatesFor = null; S.extraction = null; $("a-checked").checked = false; $("extract-note").classList.add("hidden"); renderSigns(); }
  updateGate();

  let step = 1;
  try {
    const resp = await fetch("/api/assess", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        encounter: enc, nurse_confirmed_codes: presentCodes(), nurse_dismissed_codes: absentCodes(),
        prior_candidates: reuse ? S.candidates.map(({ status, ...c }) => c) : null,
        prior_extraction: reuse ? S.extraction : null,
        baseline_vitals: S.baseline ? S.baseline.vitals : null,
      }),
    });
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n"); buf = parts.pop();
      for (const p of parts) {
        const line = p.split("\n").find(l => l.startsWith("data: "));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "run") S.runId = ev.run_id;
        else if (ev.type === "trace") addTrace(step++, ev.entry);
        else if (ev.type === "validation") { if (ev.blocked) renderBlocked(ev.validation); }
        else if (ev.type === "proposal") renderProposal(ev.state);          // before any model call
        else if (ev.type === "candidates") {
          S.candidates = ev.candidates || []; S.candidatesFor = complaint;
          S.extraction = { mode: (ev.extraction || {}).mode, note: (ev.extraction || {}).note || "" };
          $("extract-note").textContent = S.extraction.note; $("extract-note").classList.toggle("hidden", !S.extraction.note);
          if (unreviewed().length) $("checklist-wrap").open = false;
          renderSigns();
        } else if (ev.type === "gate") {
          S.state = ev.state; S.awaiting = !!ev.awaiting_approval;
          if (ev.state.result) renderProposal(ev.state);
        } else if (ev.type === "error") renderError(ev.message);
      }
    }
  } catch (e) { renderError(String(e)); }
  S.assessing = false;
  $("agent-spin").classList.add("hidden"); $("btn-assess").disabled = false;
  updateGate();
  const st = window._status || {};
  if (S.awaiting && st.llm_enabled && st.handover_note) requestHandover(S.runId);   // opt-in
}

function addTrace(n, e) {
  const li = document.createElement("li");
  li.className = "trace-item";
  li.innerHTML = `<span class="trace-dot">${n}</span>
    <div><div class="flex items-center gap-2"><span class="font-medium">${esc(e.step)}</span>
    <span class="trace-server">${esc(e.server)}</span></div>
    <div class="text-subtle text-[12px]">${esc(e.summary)}</div></div>`;
  $("trace").appendChild(li);
}

function renderBlocked(validation) {
  const errs = (validation.issues || []).filter(i => i.severity === "error");
  $("empty-proposal").classList.add("hidden"); $("proposal").classList.add("hidden");
  $("blocked").classList.remove("hidden");
  $("blocked").innerHTML = `<div class="font-semibold mb-1">No priority given — please correct ${errs.length === 1 ? "this entry" : "these entries"}:</div>
    <ul class="list-disc pl-5 space-y-0.5">${errs.map(i => `<li>${esc(i.message)}</li>`).join("")}</ul>
    <div class="mt-1.5">If the patient has an emergency sign, tick it above — that gives RED immediately.</div>`;
}
function renderError(msg) {
  $("empty-proposal").classList.add("hidden"); $("proposal").classList.add("hidden");
  $("blocked").classList.remove("hidden");
  $("blocked").innerHTML = `<div class="font-semibold">The assessment could not be completed.</div><div>${esc(msg)}</div><div class="mt-1">Nothing was scored — try again.</div>`;
}

function renderProposal(state) {
  const r = state.result || {}; const sum = state.summary;
  if (!sum) return;
  $("empty-proposal").classList.add("hidden"); $("blocked").classList.add("hidden");
  $("proposal").classList.remove("hidden");

  const band = $("band"); const ready = sum.status === "ready";
  band.className = "rounded-lg p-4 " + (ready ? `text-white band-${sum.priority}` : "bg-gray-100 text-ink");
  $("band-label").textContent = sum.headline;
  $("band-time").textContent = ready ? sum.target : (sum.status === "incomplete" ? "More information needed" : "Use the paediatric SATS chart");
  $("band-disp").textContent = ready ? sum.where : "";
  $("explanation").textContent = sum.sentence;

  const det = state.deterioration; const d = $("deterioration");
  d.classList.toggle("hidden", !det);
  if (det) {
    d.className = det.escalated ? "note-orange" : "note-amber";
    d.innerHTML = det.escalated
      ? `<b>⚠ Worse than the earlier check — needs a second look.</b><ul class="list-disc pl-5 mt-1">${det.reasons.map(x => `<li>${esc(x.reason)}</li>`).join("")}</ul>${det.incomplete ? `<div class="mt-1">${esc(det.note)}</div>` : ""}`
      : `<b>No worsening against the earlier check.</b>${det.incomplete ? ` ${esc(det.note)}` : ""}`;
  }

  $("why-wrap").classList.toggle("hidden", !sum.reasons.length);
  $("why").innerHTML = sum.reasons.map(x => `
    <li class="why ${x.deciding ? "deciding" : ""}"><span class="why-dot ${esc(x.level || "")}"></span>
      <span class="txt">${esc(x.text)}</span><span class="detail">${esc(x.detail)}${x.deciding ? " · decides the priority" : ""}</span></li>`).join("");

  $("actions-wrap").classList.toggle("hidden", !sum.actions.length);
  $("actions").innerHTML = sum.actions.map(a => `<li class="act"><div>${esc(a.text)}</div><div class="reason">${esc(a.reason)}</div></li>`).join("");

  $("cautions-wrap").classList.toggle("hidden", !sum.cautions.length);
  $("cautions").innerHTML = sum.cautions.map(c => `<li class="caution">${esc(c)}</li>`).join("");

  const t = sum.tews;
  $("tews-title").textContent = t.complete ? `TEWS working — total ${t.total} (${t.band_name.toLowerCase()} range on its own)`
    : `TEWS working — incomplete (${t.missing.length} still needed)`;
  $("tews-rows").innerHTML = t.rows.map(x => `<tr class="${x.points ? "" : "zero"}"><td>${esc(x.label)}</td><td>${esc(x.value)}</td><td class="pts">${x.points ? "+" + x.points : "0"}</td></tr>`).join("")
    + (t.complete ? `<tr class="total"><td>Total</td><td></td><td class="pts">${t.total}</td></tr>` : "");

  const ev = r.evidence || [];
  $("sources-title").textContent = `Sources — ${ev.length} cited rule${ev.length === 1 ? "" : "s"}`;
  $("evidence").innerHTML = ev.map(e => {
    const what = e.kind === "tews" ? `${esc(e.parameter.replace(/_/g, " "))} = ${esc(e.observed)} (${esc(e.band_label)}), +${e.points}`
      : `${esc(e.observed)} → ${esc(e.implies)}`;
    return `<li class="ev"><div><div>${what}</div><div class="text-subtle text-[11px] mt-0.5">${esc(e.source)}</div></div></li>`;
  }).join("") + (r.actions || []).map(a => `<li class="ev"><div><div>Action: ${esc(a.text)}</div><div class="text-subtle text-[11px] mt-0.5">${esc(a.source)}</div></div></li>`).join("");

  const v = state.verification || {};
  $("verification").innerHTML =
    `<span class="vbadge ${v.all_cited ? "ok" : "bad"}">${v.all_cited ? "✓ every point cited" : "✗ a citation is missing"}</span>
     <span class="vbadge ${v.grounded_in_protocol ? "ok" : "bad"}" title="${esc((v.notes || []).join("; "))}">${v.grounded_in_protocol ? "✓ matches the protocol text" : "✗ does not match the protocol text"}</span>`
    + ((v.local_rules || []).length ? `<span class="vbadge info" title="${esc(v.local_rules.join("; "))}">ⓘ ${v.local_rules.length} local safety rule (not SATS)</span>` : "");
  setStale(S.stale);
}

async function requestHandover(runId) {
  try {
    const data = await (await fetch("/api/narrate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run_id: runId }) })).json();
    if (data.note && runId === S.runId) {
      $("handover").classList.remove("hidden");
      $("handover").innerHTML = `<div class="sec-h">Handover note <span class="font-normal normal-case">(written by the model, checked against the result)</span></div>${esc(data.note)}`;
    }
  } catch { /* optional — the deterministic summary stands on its own */ }
}

// ---- stale + gate ---------------------------------------------------------------------------
function setStale(on) {
  S.stale = on;
  $("stale").classList.toggle("hidden", !on);
  for (const id of ["band", "explanation", "why-wrap", "actions-wrap"]) $(id).classList.toggle("stale-dim", on);
}
function markStale() { if (S.runId && !S.confirmed) { setStale(true); updateGate(); } }

function updateGate() {
  const override = $("a-override").value;
  $("a-reason").classList.toggle("hidden", !override);
  // The model did not read the complaint -> the nurse confirms they checked the sign list.
  const needCheck = !override && S.awaiting && !S.confirmed && !!S.extraction && S.extraction.mode !== "model+keywords";
  $("a-checked-wrap").classList.toggle("hidden", !needCheck);
  const sum = (S.state || {}).summary || {};
  const ver = (S.state || {}).verification || {};
  let why = "";
  if (S.confirmed) why = "";
  else if (S.assessing) why = "Checking the complaint for signs…";
  else if (S.stale) why = "The intake changed — assess again before confirming.";
  else if (!S.awaiting) why = "Assess the patient first.";
  else if (!override && unreviewed().length) why = `Mark ${unreviewed().length === 1 ? "the suggested sign" : "each suggested sign"} as present or not present (panel 2).`;
  else if (!override && sum.status !== "ready") why = "No priority to approve yet — complete the vitals. A senior clinician can override with a reason.";
  else if (!override && !(ver.all_cited && ver.grounded_in_protocol)) why = "The self-check failed — a senior clinician must override.";
  else if (needCheck && !$("a-checked").checked) why = "Check the full sign list, then tick the box above.";
  $("gate-hint").textContent = why;
  $("btn-approve").disabled = !!why || S.confirmed;
  // A result with suggested signs still unreviewed is PROVISIONAL — say so on the band itself.
  const n = unreviewed().length; const pending = n > 0 && !S.assessing && !S.confirmed;
  $("pending").classList.toggle("hidden", !pending);
  if (pending) $("pending").innerHTML = `<b>Provisional.</b> ${n} suggested sign${n === 1 ? "" : "s"} not reviewed yet (${esc(unreviewed().map(c => c.label).join("; "))}). The priority may rise — mark ${n === 1 ? "it" : "each"} in panel 2.`;
  $("band").classList.toggle("provisional", pending);
  $("btn-approve").textContent = override ? `Override to ${override} & place in queue` : "Confirm priority & place in queue";
}

async function approve() {
  const name = $("a-name").value.trim();
  if (!name) { $("gate-hint").textContent = "Enter your name to confirm."; $("a-name").focus(); return; }
  const override = $("a-override").value; const reason = $("a-reason").value.trim();
  if (override && !reason) { $("gate-hint").textContent = "An override needs a written reason."; $("a-reason").focus(); return; }
  $("btn-approve").disabled = true;
  let data;
  try {
    const resp = await fetch("/api/approve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: S.runId, clinician_name: name,
        decision: override ? "override" : "approve", override_priority: override || null, reason,
        signs_checked: $("a-checked").checked }),
    });
    data = await resp.json();
  } catch (e) { data = { error: String(e) }; }
  const c = data.confirmed || {};
  if (data.error || !c.confirmed_priority) {      // never show a confirmation that did not happen
    $("gate-hint").textContent = data.error || "The decision was not recorded — assess again.";
    $("btn-approve").disabled = false;
    return;
  }
  S.confirmed = true; S.awaiting = false;
  $("gate").classList.add("hidden");
  $("confirmed").classList.remove("hidden");
  $("confirmed").innerHTML = `✓ <b>${esc(c.confirmed_priority)}</b> confirmed by ${esc(c.clinician)}`
    + (c.decision === "override" ? ` — override of the proposed ${esc(c.proposed_priority || "no priority")}. Reason: ${esc(c.reason)}` : "")
    + `. Placed in the queue and written to the audit trail.`;
  $("after").classList.remove("hidden"); $("btn-audit").classList.remove("hidden");
  loadQueue($("f-id").value);
}

async function showAudit() {
  const rows = await (await fetch(`/api/encounter/${encodeURIComponent($("f-id").value)}/audit`)).json();
  $("audit").classList.toggle("hidden");
  $("audit").textContent = rows.map(r =>
    `${r.ts.slice(11, 19)}  ${r.event.padEnd(14)} ${(r.tool || "").padEnd(26)} ${r.actor}`).join("\n");
}

function startRecheck() {
  const v = readVitals(); delete v.measured_at;
  S.baseline = { vitals: v, at: new Date() };
  S.measuredAt = null;
  $("recheck-note").classList.remove("hidden");
  $("recheck-note").innerHTML = `<b>Re-check mode.</b> Enter the new vitals and press <b>Assess priority</b>. They will be compared with the ${S.baseline.at.toTimeString().slice(0, 5)} reading and any worsening will be flagged.`;
  $("after").classList.add("hidden"); $("confirmed").classList.add("hidden");
  S.confirmed = false; $("gate").classList.remove("hidden");
  $("a-checked").checked = false;            // a new assessment needs its own sign check
  markStale(); $("f-rr").focus();
}

// ---- queue ----------------------------------------------------------------------------------
async function loadQueue(highlightId) {
  const q = await (await fetch("/api/queue")).json();
  $("queue").innerHTML = q.map((item, i) => {
    const p = item.priority; const id = item.encounter.encounter_id;
    const badge = item.is_next ? `<span class="qnext">NEXT →</span>` : (item.position === item.total ? `<span class="qlast">last</span>` : "");
    const tag = item.demo ? `<span class="qtag">demo</span>` : `<span class="qtag mine">✓ ${esc(item.clinician)} ${esc(item.confirmed_at)}</span>`;
    return `<div class="qcard q-${esc(p)} ${item.demo ? "demo" : ""} ${id === highlightId ? "mine" : ""}" data-i="${i}" title="${esc(item.complaint)}">
      <div class="flex items-center justify-between mb-1.5">
        <div class="flex items-center gap-1.5"><span class="qpos">${item.position}</span>${badge}</div>${tag}</div>
      <div class="qprio ${esc(p)}">${esc(p)} · ${esc(item.acuity)}</div>
      <div class="qcomplaint mt-0.5">${esc(item.complaint || "—")}</div>
      <div class="text-[11px] text-subtle mt-1">${esc(id)} · TEWS ${item.tews_total ?? "–"}</div>
    </div>`;
  }).join("");
  window._queue = q;
  document.querySelectorAll(".qcard").forEach(el => el.onclick = () => fillForm(window._queue[+el.dataset.i].encounter));
}

async function loadSynthetic() {
  const item = await (await fetch("/api/synthetic")).json();   // random acuity; signs NOT pre-applied
  fillForm(item.encounter);
}

// ---- model switcher + download --------------------------------------------------------------
let _catalogKey = "";
function renderModelSelect(s) {
  const cat = s.catalog || [];
  const key = JSON.stringify(cat.map(c => [c.id, c.installed])) + "|" + (s.model || "");
  if (key === _catalogKey) return;        // avoid clobbering an open dropdown on every poll
  _catalogKey = key;
  $("model-select").innerHTML = cat.map(c => {
    const tag = c.installed ? "✓" : `⤓ ${c.size_gb}GB`;
    const active = (c.id === s.model) ? " (active)" : "";
    return `<option value="${esc(c.id)}" ${c.id === s.model ? "selected" : ""}>${esc(c.label)} · ${tag}${active}</option>`;
  }).join("") || `<option>${esc(s.model || "none")}</option>`;
}

async function selectModel(modelId) {
  $("model-banner").classList.remove("hidden");
  $("pull-progress").classList.remove("hidden");
  $("pull-progress").textContent = "preparing " + modelId + "…";
  const resp = await fetch("/api/model/select", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: modelId }),
  });
  const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n"); buf = parts.pop();
    for (const p of parts) {
      const line = p.split("\n").find(l => l.startsWith("data: ")); if (!line) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.type === "progress") $("pull-progress").textContent = ev.message;
      else if (ev.type === "error") { $("pull-progress").textContent = "error: " + ev.message; _catalogKey = ""; refreshStatus(); }
      else if (ev.type === "done") {
        $("pull-progress").textContent = "ready: " + (ev.model || modelId);
        _catalogKey = "";
        await refreshStatus();
        setTimeout(() => { $("pull-progress").classList.add("hidden"); refreshStatus(); }, 1500);
      }
    }
  }
}

let _statusTimer = null;
async function refreshStatus() {
  let s;
  try { s = await (await fetch("/api/status")).json(); } catch { return; }
  renderModelSelect(s);
  const ready = !!s.model_ready || !!s.llm_off_by_config;
  // Auto-hide the banner the moment a model is ready; poll until then so a late/cold-starting
  // runtime clears it on its own (no manual reload, no falsely-stuck banner).
  $("model-banner").classList.toggle("hidden", ready && !s.warming);
  $("btn-pull").classList.toggle("hidden", !!s.warming);
  if (s.warming) {
    $("banner-msg").innerHTML = `Language model is loading into memory (about a minute, once). <b>Priorities already work</b>; until it finishes, signs are suggested by keyword only.`;
    if (!_statusTimer) _statusTimer = setInterval(refreshStatus, 3000);
    window._status = s;
    return;
  }
  if (!ready) {
    $("banner-msg").innerHTML = `Language model not ready yet (<b>${esc(s.backend)}</b>). ${esc(s.backend_hint || "")} <b>Priorities still work</b> — they are calculated without it. Until it is ready, signs are suggested by keyword only, so check the full sign list.`;
  }
  if (!ready && !_statusTimer) _statusTimer = setInterval(refreshStatus, 3000);
  if (ready && _statusTimer) { clearInterval(_statusTimer); _statusTimer = null; }
  if (s.fhir && s.fhir.enabled) {
    $("btn-fhir").classList.remove("hidden");
    $("chip-fhir").textContent = "FHIR: " + (s.fhir.base_url || "").replace(/^https?:\/\//, "");
    $("chip-fhir").classList.remove("hidden");
  }
  window._status = s;
}

function downloadDefaultModel() {
  const cat = (window._status && window._status.catalog) || [];
  const std = cat.find(c => c.installed) || cat.find(c => c.tier === "standard") || cat[0];
  if (std) selectModel(std.id);
}

async function importFromFhir() {
  $("btn-fhir").disabled = true;
  const prev = $("btn-fhir").textContent;
  $("btn-fhir").textContent = "Importing…";
  try {
    const data = await (await fetch("/api/fhir/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })).json();
    if (data.error) {
      $("fhir-note").classList.remove("hidden");
      $("fhir-note").textContent = "FHIR import failed: " + data.error + " — use manual/synthetic instead.";
      return;
    }
    fillForm(data.encounter, { measuredAt: (data.encounter.vitals || {}).measured_at || null });
    $("fhir-note").classList.remove("hidden");
    $("fhir-note").innerHTML = `Imported vitals from <b>${esc(data.source)}</b> (patient ${esc(data.patient_id)}). Check them, then assess.`;
  } finally {
    $("btn-fhir").disabled = false;
    $("btn-fhir").textContent = prev;
  }
}

// ---- wire up --------------------------------------------------------------------------------
$("btn-assess").onclick = () => assess();
$("btn-synthetic").onclick = loadSynthetic;
$("btn-fhir").onclick = importFromFhir;
$("btn-approve").onclick = approve;
$("btn-audit").onclick = showAudit;
$("btn-new").onclick = resetPatient;
$("btn-new2").onclick = resetPatient;
$("btn-recheck").onclick = startRecheck;
$("btn-pull").onclick = downloadDefaultModel;
$("model-select").onchange = (e) => selectModel(e.target.value);
$("a-override").onchange = updateGate;
$("a-checked").onchange = updateGate;
document.querySelectorAll(".emerg-tick").forEach(el => el.onchange = () => setSign(el.value, el.checked ? "present" : null));
// Any edit to the intake makes the result on screen out of date — say so, and lock the gate.
for (const id of ["f-age", "f-sex", "f-complaint", "f-avpu", "f-mobility", "f-trauma", ...Object.values(VITAL_FIELDS)]) {
  $(id).addEventListener("input", () => { if (document.querySelector(`#${id}.vital`)) S.measuredAt = null; markStale(); });
}
(async () => {
  try { SIGN_GROUPS = await (await fetch("/api/discriminators")).json(); } catch { SIGN_GROUPS = []; }
  resetPatient();
  loadQueue(); refreshStatus();
})();
