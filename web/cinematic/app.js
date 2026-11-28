/* Scam Guard — cinematic case file. Display only: it never computes or changes a verdict,
   score or advice, and never fetches a URL found in a message. All message-derived text is
   rendered with textContent. No third-party code. */
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Nine canonical steps, with friendly case-file labels and the three model-decided steps.
  const STEPS = [
    { name: "message_in", label: "INPUT", model: false },
    { name: "ocr", label: "OCR", model: false },
    { name: "sanitize", label: "CLEAN", model: false },
    { name: "extract", label: "EXTRACT", model: true },
    { name: "verify", label: "VERIFY", model: false },
    { name: "detect", label: "DETECT", model: true },
    { name: "guards", label: "CUES", model: false },
    { name: "gate", label: "GATE", model: false },
    { name: "advise", label: "ADVISE", model: true },
  ];
  const MODEL_GLYPH = "◉"; // ◉
  const CODE_GLYPH = "□"; // □

  // Five display states: a word + glyph, so meaning never rests on colour alone.
  // benign must read "no red flags found", never "safe".
  const STATE = {
    scam: { word: "SCAM", glyph: "◉" },
    suspicious: { word: "SUSPICIOUS", glyph: "▲" },
    benign: { word: "No red flags found", glyph: "○" },
    unsure: { word: "UNSURE", glyph: "？" },
    incomplete: { word: "Incomplete input", glyph: "◌" },
  };

  function deriveState(result) {
    const d = result.decision;
    if (d === "unsure" && result.reason === "need_clearer_input") return "incomplete";
    return STATE[d] ? d : "unsure";
  }

  // ---- cold open ----
  const intro = $("intro");
  const stage = $("stage");
  function closeIntro() {
    if (intro.hidden) return;
    stage.hidden = false;
    if (REDUCED) { intro.hidden = true; return; }
    intro.classList.add("closing");
    setTimeout(() => (intro.hidden = true), 600);
  }
  if (REDUCED) { intro.hidden = true; stage.hidden = false; }
  else { stage.hidden = false; setTimeout(closeIntro, 3800); }
  $("skip-intro").addEventListener("click", closeIntro);
  intro.addEventListener("click", closeIntro);

  // ---- cue highlighting (DOM + textContent only) ----
  function highlightEvidence(message, cues) {
    const frag = document.createDocumentFragment();
    const ranges = [];
    (cues || []).forEach((cue) => {
      if (!cue) return;
      const idx = message.indexOf(cue);
      if (idx >= 0) ranges.push([idx, idx + cue.length]);
    });
    ranges.sort((a, b) => a[0] - b[0]);
    const clean = [];
    let lastEnd = -1;
    ranges.forEach((r) => { if (r[0] >= lastEnd) { clean.push(r); lastEnd = r[1]; } });

    let cursor = 0;
    clean.forEach(([start, end]) => {
      if (start > cursor) frag.appendChild(document.createTextNode(message.slice(cursor, start)));
      const mark = document.createElement("mark");
      mark.className = "cue";
      const hl = document.createElement("span");
      hl.className = "hl";
      hl.setAttribute("aria-hidden", "true");
      const tx = document.createElement("span");
      tx.className = "tx";
      tx.textContent = message.slice(start, end); // exact substring, inert text
      mark.append(hl, tx);
      frag.appendChild(mark);
      cursor = end;
    });
    if (cursor < message.length) frag.appendChild(document.createTextNode(message.slice(cursor)));
    return frag;
  }

  // ---- the case run ----
  const announce = $("announce");
  let runToken = 0;

  function runCase(result, ctx) {
    const token = ++runToken;
    const stateKey = deriveState(result);
    const info = STATE[stateKey];
    // Prefer the typed text; otherwise show the screenshot's transcribed (OCR) text.
    const message = ctx.message || result.transcript || "[screenshot — no readable text found]";

    // SAFETY FIRST: announce verdict + advice immediately, before any animation.
    announce.textContent = "Verdict: " + info.word + ". Advice: " + (result.advice || "");

    // case-file header
    const caseNo = String(Math.floor(1000 + Math.random() * 9000));
    const modeText = ctx.mode === "live" ? "LIVE CASE" : "MOCK DATA" + (ctx.note ? " · " + ctx.note : "");
    $("case-meta").textContent = "CASE FILE " + caseNo + " · " + modeText;

    // reveal the case
    $("error").hidden = true;
    const caseEl = $("case");
    caseEl.hidden = false;
    caseEl.setAttribute("data-verdict", stateKey);

    // verdict text (present immediately; animation only scales it in)
    const vState = $("v-state");
    vState.textContent = info.word;
    vState.classList.remove("in");
    $("v-score").textContent =
      stateKey === "incomplete" ? "not enough evidence to judge" : "risk score " + Number(result.score).toFixed(2);

    // quoted-cues heading + tag line (real cues only)
    const cuesArr = result.cues || [];
    const foundCount = cuesArr.filter((c) => c && message.indexOf(c) >= 0).length;
    $("cues-head").textContent =
      cuesArr.length ? "Quoted cues · " + foundCount + " of " + cuesArr.length + " found in text" : "Quoted cues";
    const tags = $("cue-tags");
    tags.textContent = "";
    tags.classList.remove("none");
    if (cuesArr.length) {
      cuesArr.forEach((c, i) => {
        if (i) tags.appendChild(document.createTextNode("  ·  "));
        const s = document.createElement("span");
        s.textContent = c; // inert text
        tags.appendChild(s);
      });
    } else {
      tags.classList.add("none");
      tags.textContent = "none quoted";
    }

    // advice lines (closing-credits stagger), exactly as the API returned it
    const adviceEl = $("advice");
    adviceEl.textContent = "";
    const lines = String(result.advice || "—").split(/(?<=[.!؟?])\s+/).filter((s) => s.trim());
    const lineEls = (lines.length ? lines : [String(result.advice || "—")]).map((ln) => {
      const div = document.createElement("div");
      div.className = "line";
      div.textContent = ln;
      adviceEl.appendChild(div);
      return div;
    });

    // film strip
    const strip = $("strip");
    strip.textContent = "";
    const byStep = {};
    (result.trace || []).forEach((t) => (byStep[t.step] = t));
    const frameEls = STEPS.map((s, i) => {
      const li = document.createElement("li");
      li.className = "frame" + (s.model ? " model" : "");
      const g = document.createElement("div"); g.className = "fglyph"; g.textContent = s.model ? MODEL_GLYPH : CODE_GLYPH;
      const nm = document.createElement("div"); nm.className = "fname"; nm.textContent = s.label;
      const st = document.createElement("div"); st.className = "fstatus"; st.textContent = "";
      li.append(g, nm, st);
      strip.appendChild(li);
      return li;
    });
    function applyFrame(el, i) {
      const t = byStep[i + 1];
      const st = el.querySelector(".fstatus");
      if (!t) { st.textContent = "—"; return; }
      el.classList.add("is-" + t.status);
      st.textContent = t.status === "ok" ? "ok" : t.status;
    }

    const exhibit = document.querySelector(".exhibit");
    const ev = $("evidence");

    if (REDUCED) {
      exhibit.classList.add("in");
      ev.textContent = ""; ev.appendChild(highlightEvidence(message, cuesArr));
      ev.querySelectorAll("mark.cue").forEach((m) => m.classList.add("sweep"));
      frameEls.forEach((el, i) => { el.classList.add("in"); applyFrame(el, i); });
      vState.classList.add("in");
      lineEls.forEach((el) => el.classList.add("in"));
      return;
    }

    exhibit.classList.remove("in"); void exhibit.offsetWidth; exhibit.classList.add("in");

    // 1) type the message, then sweep the highlighters
    ev.textContent = "";
    typeText(ev, message, token, () => {
      if (token !== runToken) return;
      ev.textContent = "";
      ev.appendChild(highlightEvidence(message, cuesArr));
      const marks = ev.querySelectorAll("mark.cue");
      marks.forEach((m, i) => setTimeout(() => token === runToken && m.classList.add("sweep"), 120 + i * 320));

      // 2) light the film strip frame by frame
      frameEls.forEach((el, i) =>
        setTimeout(() => { if (token !== runToken) return; el.classList.add("in"); applyFrame(el, i); }, i * 110)
      );

      // 3) verdict word, then advice credits
      const afterStrip = frameEls.length * 110 + 160;
      setTimeout(() => {
        if (token !== runToken) return;
        vState.classList.add("in");
        lineEls.forEach((el, i) => setTimeout(() => el.classList.add("in"), 300 + i * 260));
      }, afterStrip);
    });
  }

  function typeText(el, text, token, done) {
    const n = text.length;
    if (!n) return done();
    const per = Math.max(10, Math.min(24, Math.floor(900 / n)));
    const chunk = Math.max(1, Math.ceil(n / (900 / per)));
    let i = 0;
    const id = setInterval(() => {
      if (token !== runToken) return clearInterval(id);
      i += chunk;
      el.textContent = text.slice(0, Math.min(i, n));
      if (i >= n) { clearInterval(id); done(); }
    }, per);
  }

  // ---- presenter scenarios ----
  function loadScenario(name) {
    fetch("mock/" + name + ".json", { cache: "no-store" })
      .then((r) => { if (!r.ok) throw new Error("mock " + r.status); return r.json(); })
      .then((data) => runCase(data, { mode: "demo", message: data._message || "" }))
      .catch(() => showError("Could not load the recorded demo file."));
  }
  document.querySelectorAll(".scenario").forEach((btn) =>
    btn.addEventListener("click", () => loadScenario(btn.dataset.scenario))
  );
  const SCEN_KEYS = { 1: "scam", 2: "blurry", 3: "injection" };
  document.addEventListener("keydown", (e) => {
    const tag = (document.activeElement && document.activeElement.tagName) || "";
    if (tag === "TEXTAREA" || tag === "INPUT") return;
    if (SCEN_KEYS[e.key]) { e.preventDefault(); loadScenario(SCEN_KEYS[e.key]); }
  });

  // ---- live submit ----
  const form = $("form");
  const submitBtn = $("submit");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const textVal = $("text").value.trim();
    const fileInput = $("image");
    const hasImage = fileInput.files && fileInput.files.length > 0;
    if (!textVal && !hasImage) { showError("Hand over a message or a screenshot first."); return; }

    const data = new FormData();
    if (textVal) data.append("text", textVal);
    if (hasImage) data.append("image", fileInput.files[0]);
    const sid = $("session_id").value.trim();
    if (sid) data.append("session_id", sid);

    submitBtn.disabled = true;
    submitBtn.textContent = "Examining…";
    fetch("/analyze", { method: "POST", body: data })
      .then((r) => {
        if (!r.ok) return r.json().then((j) => Promise.reject(j.detail || "Request failed")).catch(() => Promise.reject("Request failed (" + r.status + ")"));
        return r.json();
      })
      .then((result) => runCase(result, { mode: "live", message: textVal }))
      .catch(() => {
        fetch("mock/scam.json", { cache: "no-store" })
          .then((r) => r.json())
          .then((data) => runCase(data, { mode: "demo", note: "API OFFLINE", message: data._message || "" }))
          .catch(() => showError("The API is unreachable and the recorded demo could not load."));
      })
      .finally(() => { submitBtn.disabled = false; submitBtn.textContent = "Hand over evidence"; });
  });

  $("reset").addEventListener("click", () => {
    form.reset();
    $("case").hidden = true;
    $("error").hidden = true;
    $("case-meta").textContent = "CASE FILE 0000 · STANDBY";
    announce.textContent = "";
    runToken++;
  });

  function showError(msg) { const el = $("error"); el.textContent = msg; el.hidden = false; }
})();
