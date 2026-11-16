"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pill = (p) => `<span class="pill ${p || "none"}">${esc(p || "—")}</span>`;
function when(ts) {                       // ISO (UTC) -> local short "15 Nov 14:03"
  if (!ts) return "—";
  const d = new Date(ts);
  if (isNaN(d)) return esc(ts);
  return d.toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}
function clock(ts) { const d = new Date(ts); return isNaN(d) ? esc(ts) : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }

// ---- tabs -----------------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === t));
  const tab = t.dataset.tab;
  $("tab-runs").classList.toggle("hidden", tab !== "runs");
  $("tab-queue").classList.toggle("hidden", tab !== "queue");
});

// ---- patient runs ---------------------------------------------------------------------------
async function loadRuns() {
  let rows;
  try { rows = await (await fetch("/api/history/encounters")).json(); } catch { rows = []; }
  if (!rows.length) { $("runs-body").innerHTML = `<div class="muted text-[13px] py-6">No patient runs recorded yet. Assess a patient in the workspace, and it will appear here.</div>`; return; }
  $("runs-body").innerHTML = `<table class="htable"><thead><tr>
      <th>Encounter</th><th>Complaint</th><th>Proposed → confirmed</th><th>Clinician</th>
      <th>Last activity</th><th>Steps</th></tr></thead><tbody>${rows.map(r => `
      <tr class="clickable" data-id="${esc(r.encounter_id)}">
        <td class="font-mono text-[12px]">${esc(r.encounter_id)}</td>
        <td style="max-width:320px">${esc(r.complaint || "—")}${r.flags ? ` <span class="pill-o">⚑ ${r.flags} flag${r.flags > 1 ? "s" : ""}</span>` : ""}</td>
        <td>${pill(r.proposed_priority)} <span class="muted">→</span> ${r.confirmed ? pill(r.confirmed_priority) : '<span class="muted text-[12px]">not confirmed</span>'}${r.decision === "override" ? ' <span class="pill-o override">override</span>' : ""}</td>
        <td>${esc(r.clinician || "—")}</td>
        <td class="muted">${when(r.last_ts)}</td>
        <td class="muted">${r.events ?? "—"}</td>
      </tr>`).join("")}</tbody></table>`;
  $("runs-body").querySelectorAll("tr.clickable").forEach(tr => tr.onclick = () => showDetail(tr.dataset.id));
}

// ---- queue history --------------------------------------------------------------------------
async function loadQueue() {
  let data;
  try { data = await (await fetch("/api/history/queue")).json(); } catch { data = { rows: [] }; }
  const rows = data.rows || [];
  const order = ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE"];
  const stats = [`<div class="stat"><div class="n">${data.total || 0}</div><div class="l">confirmed total</div></div>`]
    .concat(order.filter(p => (data.by_priority || {})[p]).map(p =>
      `<div class="stat"><div class="n" style="color:var(--x)">${data.by_priority[p]}</div><div class="l">${p}</div></div>`))
    .concat(data.overrides ? [`<div class="stat"><div class="n">${data.overrides}</div><div class="l">overrides</div></div>`] : []);
  $("queue-stats").innerHTML = stats.join("");
  if (!rows.length) { $("queue-body").innerHTML = `<div class="muted text-[13px] py-6">No confirmed dispositions yet. Confirm a patient in the workspace and it will be traced here — separately from the live waiting queue.</div>`; return; }
  $("queue-body").innerHTML = `<table class="htable"><thead><tr>
      <th>Confirmed</th><th>Encounter</th><th>Priority</th><th>Decision</th><th>Clinician</th><th>Reason</th></tr></thead><tbody>${rows.map(r => `
      <tr class="clickable" data-id="${esc(r.encounter_id)}">
        <td class="muted">${when(r.ts)}</td>
        <td class="font-mono text-[12px]">${esc(r.encounter_id)}</td>
        <td>${pill(r.confirmed_priority)}${r.decision === "override" && r.proposed_priority ? ` <span class="muted text-[11px]">(was ${esc(r.proposed_priority)})</span>` : ""}</td>
        <td>${r.decision === "override" ? '<span class="pill-o override">override</span>' : '<span class="pill-o">approved</span>'}</td>
        <td>${esc(r.clinician || "—")}</td>
        <td class="muted" style="max-width:260px">${esc(r.reason || "—")}</td>
      </tr>`).join("")}</tbody></table>`;
  $("queue-body").querySelectorAll("tr.clickable").forEach(tr => tr.onclick = () => showDetail(tr.dataset.id));
}

// ---- one encounter's full trace -------------------------------------------------------------
const EVENT_LABEL = {
  tool_call: "tool call", tool_result: "tool result", tool_error: "tool error",
  await_approval: "waiting for a person", approval: "clinician decision", flag: "deterioration flag",
  narration: "handover note",
};
function summarise(e) {
  const p = e.payload || {};
  if (e.event === "approval") return `${esc(p.clinician || e.actor || "clinician")} → ${esc(p.confirmed_priority || "—")}${p.decision === "override" ? ` (override of ${esc(p.proposed_priority || "—")}${p.reason ? `: ${esc(p.reason)}` : ""})` : ""}`;
  if (e.event === "flag") return `escalated: ${p.escalated ? "yes" : "no"}`;
  if (p.proposed_priority) return `proposed ${esc(p.proposed_priority)}`;
  if (e.tool) return esc(e.tool);
  return "";
}
async function showDetail(id) {
  let data;
  try { data = await (await fetch(`/api/history/encounter/${encodeURIComponent(id)}`)).json(); } catch { return; }
  const trail = data.trail || [];
  const appr = [...trail].reverse().find(e => e.event === "approval");
  const p = appr ? (appr.payload || {}) : {};
  $("detail-head").innerHTML = `
    <div class="flex items-center justify-between flex-wrap gap-2">
      <div><div class="text-[11px] text-subtle uppercase tracking-wide">Encounter</div>
        <div class="font-mono text-[15px] font-semibold">${esc(id)}</div></div>
      <div class="text-right">${appr ? `${pill(p.confirmed_priority)} <span class="muted text-[12px]">confirmed by ${esc(p.clinician || appr.actor)}</span>` : '<span class="muted text-[13px]">not confirmed</span>'}</div>
    </div>
    <div class="muted text-[12px] mt-2">${trail.length} recorded step${trail.length === 1 ? "" : "s"}${trail.length ? ` · ${when(trail[0].ts)} → ${when(trail[trail.length - 1].ts)}` : ""}</div>`;
  $("detail-trace").innerHTML = trail.map(e => `
    <div class="trace-row">
      <div class="t">${clock(e.ts)}</div>
      <div class="e">${esc(EVENT_LABEL[e.event] || e.event)}</div>
      <div><span class="muted">${esc(e.actor || "")}</span> ${summarise(e)}</div>
    </div>`).join("") || `<div class="muted text-[13px] p-4">No steps.</div>`;
  $("list-view").classList.add("hidden");
  $("detail-view").classList.remove("hidden");
  window.scrollTo(0, 0);
}
$("back").onclick = () => { $("detail-view").classList.add("hidden"); $("list-view").classList.remove("hidden"); };

loadRuns(); loadQueue();
