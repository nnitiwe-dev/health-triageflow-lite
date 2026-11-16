"use strict";
// Bottom navbar: just the sign-out action. Everything else is plain links.
(function () {
  const btn = document.getElementById("tfnav-logout");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try { await fetch("/api/session/logout", { method: "POST" }); } catch (_) { /* ignore */ }
    window.location.href = "/login";
  });
})();
