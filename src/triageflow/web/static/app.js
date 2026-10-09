"use strict";
const $ = (id) => document.getElementById(id);
const PRIORITY_ORDER = { RED: 0, ORANGE: 1, YELLOW: 2, GREEN: 3, BLUE: 4, null: 5 };
let currentEncounter = null;
let lastProposal = null;

function newId() {
  const d = new Date();
  return `OPD-${d.getHours()}${String(d.getMinutes()).padStart(2,"0")}-${Math.floor(Math.random()*900+100)}`;
}
function num(id) { const v = $(id).value; return v === "" ? null : Number(v); }
function str(id) { const v = $(id).value; return v === "" ? null : v; }

function readForm() {
  return {
    encounter_id: $("f-id").value || newId(),
    age_years: num("f-age"),
    presenting_complaint: $("f-complaint").value || "",
    discriminator_codes: currentEncounter?.discriminator_codes || [],
    vitals: {
      respiratory_rate: num("f-rr"), heart_rate: num("f-hr"), systolic_bp: num("f-sbp"),
      temperature_c: num("f-temp"), spo2: num("f-spo2"), glucose_mmol: num("f-glucose"),
      avpu: str("f-avpu"), mobility: str("f-mobility"), trauma: $("f-trauma").checked,
    },
  };
}
function fillForm(enc) {
  currentEncounter = enc;
  $("f-id").value = enc.encounter_id || newId();
  $("f-age").value = enc.age_years ?? "";
  $("f-complaint").value = enc.presenting_complaint || "";
  const v = enc.vitals || {};
  $("f-rr").value = v.respiratory_rate ?? ""; $("f-hr").value = v.heart_rate ?? "";
  $("f-sbp").value = v.systolic_bp ?? ""; $("f-temp").value = v.temperature_c ?? "";
  $("f-spo2").value = v.spo2 ?? ""; $("f-glucose").value = v.glucose_mmol ?? "";
  $("f-avpu").value = v.avpu ?? ""; $("f-mobility").value = v.mobility ?? "";
  $("f-trauma").checked = !!v.trauma;
}

// ---- streaming assess -------------------------------------------------------
async function assess() {
  const enc = readForm();
  $("f-id").value = enc.encounter_id;
  currentEncounter = enc;
  resetPanels();
  $("agent-spin").classList.remove("hidden");
  $("btn-assess").disabled = true;

  const confirmed = [...document.querySelectorAll(".cand:checked")].map(c => c.value);
  const resp = await fetch("/api/assess", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ encounter: enc, nurse_confirmed_codes: confirmed,
                           recheck_vitals: enc._recheck || null }),
  });
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "", step = 1;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n"); buf = parts.pop();
    for (const p of parts) {
      const line = p.split("\n").find(l => l.startsWith("data: "));
      if (!line) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.type === "trace") addTrace(step++, ev.entry);
      else if (ev.type === "candidates") renderCandidates(ev.candidates);
      else if (ev.type === "proposal") renderProposal(ev.state);
      else if (ev.type === "error") addTrace(step++, { step: "error", server: "system", summary: ev.message });
    }
  }
  $("agent-spin").classList.add("hidden");
  $("btn-assess").disabled = false;
}

function resetPanels() {
  $("trace").innerHTML = ""; $("candidates").innerHTML = "";
  $("candidates-wrap").classList.add("hidden"); $("btn-rescore").classList.add("hidden");
  $("proposal").classList.add("hidden"); $("empty-proposal").classList.remove("hidden");
  $("confirmed").classList.add("hidden"); $("audit").classList.add("hidden");
  $("btn-audit").classList.add("hidden"); $("deterioration").classList.add("hidden");
}

function addTrace(n, e) {
  const li = document.createElement("li");
  li.className = "trace-item";
  li.innerHTML = `<span class="trace-dot">${n}</span>
    <div><div class="flex items-center gap-2"><span class="font-medium">${e.step}</span>
    <span class="trace-server">${e.server}</span></div>
    <div class="text-subtle text-[12px]">${e.summary}</div></div>`;
  $("trace").appendChild(li);
}

function renderCandidates(cands) {
  if (!cands || !cands.length) return;
  $("candidates-wrap").classList.remove("hidden");
  $("btn-rescore").classList.remove("hidden");
  const confirmed = new Set(currentEncounter?.discriminator_codes || []);
  $("candidates").innerHTML = cands.map(c => `
    <label class="flex items-start gap-2 text-[13px]">
      <input type="checkbox" class="cand mt-0.5 h-4 w-4" value="${c.code}" ${confirmed.has(c.code) ? "checked" : ""}/>
      <span>${c.label} <span class="text-subtle">→ ${c.implies}${c.via === "llm" ? " · via model" : ""}</span></span>
    </label>`).join("");
}

function renderProposal(state) {
  lastProposal = state;
  const r = state.result || {};
  $("empty-proposal").classList.add("hidden");
  $("proposal").classList.remove("hidden");

  const band = $("band");
  band.className = "rounded-lg p-4 text-white band-" + (r.proposed_priority || "GREEN");
  if (!r.complete || !r.proposed_priority) {
    band.className = "rounded-lg p-4 bg-gray-100 text-ink";
    $("band-label").textContent = "Incomplete";
    $("band-time").textContent = ""; $("band-disp").textContent = (r.warnings || []).join(" ");
  } else {
    $("band-label").textContent = r.proposed_priority;
    $("band-time").textContent = r.target_time || "";
    $("band-disp").textContent = r.disposition || "";
  }
  $("explanation").textContent = state.explanation || "";

  const det = state.deterioration;
  if (det && det.escalated) {
    $("deterioration").classList.remove("hidden");
    $("deterioration").innerHTML = `⚠ <b>Deterioration flag</b> — ${det.reasons.map(x=>x.reason).join("; ")}. Raised for a second look (the agent cannot move the patient).`;
  }

  $("evidence").innerHTML = (r.evidence || []).map(e => {
    const left = e.kind === "discriminator"
      ? `<span class="ev-pts">sign</span>`
      : `<span class="ev-pts">${e.points > 0 ? "+" : ""}${e.points ?? ""}</span>`;
    const what = e.kind === "discriminator"
      ? `${e.observed} <span class="text-subtle">→ ${e.implies}</span>`
      : `${e.parameter} = ${e.observed} <span class="text-subtle">(${e.band_label})</span>`;
    return `<li class="ev">${left}<div><div>${what}</div><div class="text-subtle text-[11px] mt-0.5">${e.source}</div></div></li>`;
  }).join("");

  const v = state.verification || {};
  $("verification").innerHTML =
    `<span class="vbadge ${v.all_cited ? "ok":"bad"}">${v.all_cited ? "✓ all sourced":"✗ missing citation"}</span>
     <span class="vbadge ${v.grounded_in_protocol ? "ok":"bad"}">${v.grounded_in_protocol ? "✓ grounded in protocol":"✗ not grounded"}</span>
     <span class="vbadge">${v.citation_count ?? 0} citations</span>`;
  $("a-override").value = "";
}

// ---- approval ---------------------------------------------------------------
async function approve() {
  const name = $("a-name").value.trim();
  if (!name) { $("a-name").focus(); return; }
  const override = $("a-override").value;
  const reason = $("a-reason").value.trim();
  if (override && !reason) { $("a-reason").focus(); return; }
  $("btn-approve").disabled = true;
  const resp = await fetch("/api/approve", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      encounter_id: currentEncounter.encounter_id, clinician_name: name,
      decision: override ? "override" : "approve",
      override_priority: override || null, reason,
    }),
  });
  const data = await resp.json();
  $("btn-approve").disabled = false;
  if (data.error) { alert(data.error); return; }
  const c = data.confirmed || {};
  $("confirmed").classList.remove("hidden");
  $("confirmed").innerHTML = `✓ Confirmed <b>${c.confirmed_priority}</b> by ${c.clinician}
    ${override ? `(override · ${reason})` : ""} — placed in queue. Logged to audit trail.`;
  $("btn-audit").classList.remove("hidden");
}

async function showAudit() {
  const resp = await fetch(`/api/encounter/${currentEncounter.encounter_id}/audit`);
  const rows = await resp.json();
  $("audit").classList.toggle("hidden");
  $("audit").textContent = rows.map(r =>
    `${r.ts.slice(11,19)}  ${r.event.padEnd(14)} ${(r.tool||"").padEnd(26)} ${r.actor}`).join("\n");
}

// ---- queue ------------------------------------------------------------------
async function loadQueue() {
  const resp = await fetch("/api/queue");
  const q = await resp.json();
  $("queue").innerHTML = q.map((item, i) => {
    const p = item.proposed_priority;
    return `<div class="qcard" data-i="${i}">
      <div class="flex items-center justify-between mb-1">
        <span class="text-[11px] text-subtle">${item.encounter.encounter_id}</span>
        <span class="qband band-${p}"></span></div>
      <div class="text-[13px] font-medium leading-snug">${item.complaint}</div>
      <div class="text-[11px] text-subtle mt-1">${p} · TEWS ${item.tews_total ?? "–"}</div>
    </div>`;
  }).join("");
  window._queue = q;
  document.querySelectorAll(".qcard").forEach(el =>
    el.onclick = () => fillForm(window._queue[+el.dataset.i].encounter));
}

async function loadSynthetic() {
  const resp = await fetch("/api/queue?n=1&seed=" + Math.floor(Math.random()*9999));
  const q = await resp.json();
  const enc = q[0].encounter; enc.encounter_id = newId();
  fillForm(enc);
}

// ---- model download (self-healing) -----------------------------------------
async function pullModel(forceClean) {
  $("pull-progress").classList.remove("hidden");
  $("pull-progress").textContent = "starting…";
  $("btn-pull").disabled = true; $("btn-repair").disabled = true;
  const resp = await fetch("/api/model/pull", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force_clean: !!forceClean }),
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
      else if (ev.type === "done") {
        $("pull-progress").textContent = ev.message || "done";
        if (ev.model) { await refreshStatus(); setTimeout(()=>$("model-banner").classList.add("hidden"), 1500); }
      }
    }
  }
  $("btn-pull").disabled = false; $("btn-repair").disabled = false;
}

async function refreshStatus() {
  const s = await (await fetch("/api/status")).json();
  $("chip-model").textContent = `model: ${s.model || "none"}${s.using_fallback ? " (fallback)" : ""}`;
  if (!s.model_ready) $("model-banner").classList.remove("hidden");
}

// ---- wire up ----------------------------------------------------------------
$("btn-assess").onclick = assess;
$("btn-synthetic").onclick = loadSynthetic;
$("btn-approve").onclick = approve;
$("btn-audit").onclick = showAudit;
$("btn-rescore").onclick = assess;
$("btn-refresh-queue").onclick = loadQueue;
$("btn-pull").onclick = () => pullModel(false);
$("btn-repair").onclick = () => pullModel(true);
$("f-id").value = newId();
loadQueue(); refreshStatus();
