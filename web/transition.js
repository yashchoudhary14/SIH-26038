/* Page transition shared by the dossier and the console.

   Leaving: a curtain closes over the page, the destination's name is drawn in
   outline and then filled, and only then does the browser navigate.
   Arriving: the page starts under the same curtain, which withdraws as a
   circle collapsing to the right edge, uncovering the new page from the left.

   Opt in per link with  data-transition="Title"  (and optionally
   data-eyebrow="LABEL"). The title is set in Inter, the typeface both pages
   share. Honours prefers-reduced-motion by doing nothing.

   Each page also carries a one-line inline script in <head> that adds the
   html.dr-entering class before first paint, so the arriving page is never
   visible un-curtained for a frame. */
(function () {
  "use strict";
  var KEY = "dr-transition";
  var REDUCED = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  var EASE = "cubic-bezier(.19,1,.22,1)";

  var css = [
    ".drx{position:fixed;inset:0;z-index:2147483647;background:#08080A;",
    "  display:flex;flex-direction:column;justify-content:center;",
    "  padding:0 clamp(20px,6vw,90px);opacity:0;pointer-events:none;",
    "  transition:opacity .42s " + EASE + ";will-change:opacity,clip-path}",
    ".drx.on{opacity:1;pointer-events:all}",
    ".drx-eye{display:flex;align-items:center;gap:16px;margin-bottom:18px;",
    "  font:400 10.5px/1 'IBM Plex Mono',ui-monospace,monospace;letter-spacing:.14em;color:#8F918D}",
    ".drx-eye i{display:block;height:1px;width:0;background:rgba(242,242,238,.28);",
    "  transition:width .9s " + EASE + " .12s}",
    ".drx.drawn .drx-eye i{width:min(260px,40vw)}",
    ".drx-t{margin:0;font:100 clamp(48px,9.5vw,150px)/.95 Inter,system-ui,-apple-system,'Segoe UI',sans-serif;",
    "  text-transform:uppercase;letter-spacing:-.05em;color:transparent;-webkit-text-stroke:1px rgba(242,242,238,.55);",
    "  transition:color .55s " + EASE + ",-webkit-text-stroke-color .55s " + EASE + "}",
    ".drx-t b{display:inline-block;white-space:nowrap;font-weight:inherit}",
    ".drx-t span{display:inline-block;opacity:0;transform:translateY(.18em);",
    "  transition:opacity .5s " + EASE + ",transform .7s " + EASE + "}",
    ".drx.drawn .drx-t span{opacity:1;transform:none}",
    ".drx.filled .drx-t{color:#F2F2EE;-webkit-text-stroke-color:rgba(242,242,238,0)}",
    ".drx.leave{transition:clip-path .9s " + EASE + "}",
    // an arriving curtain is born finished: nothing in it may animate in again
    ".drx.still,.drx.still *{transition:none!important}"
  ].join("");

  function injectStyle() {
    if (document.getElementById("drx-style")) return;
    var st = document.createElement("style");
    st.id = "drx-style";
    st.textContent = css;
    document.head.appendChild(st);
  }

  // Letters animate one by one, but each word is held together, so a title can
  // only ever break between words; and the title is sized to fit on one line
  // when it can. `settled` builds it already drawn and filled (the arriving
  // side), so measuring it cannot commit an unfinished state to animate from.
  function build(title, eyebrow, settled) {
    injectStyle();
    var el = document.createElement("div");
    el.className = settled ? "drx on drawn filled still" : "drx";
    el.setAttribute("aria-hidden", "true");
    var k = 0;
    var words = String(title).split(/\s+/).filter(Boolean).map(function (w) {
      return "<b>" + w.split("").map(function (ch) {
        return '<span style="transition-delay:' + (80 + (k++) * 26) + 'ms">' + ch.replace(/[&<>]/g, "") + "</span>";
      }).join("") + "</b>";
    });
    el.innerHTML = '<div class="drx-eye">' + String(eyebrow || "").replace(/[&<>]/g, "") + "<i></i></div>" +
      '<p class="drx-t">' + words.join(" ") + "</p>";
    document.body.appendChild(el);
    fit(el);
    return el;
  }
  function fit(el) {
    var t = el.querySelector(".drx-t"), cs = getComputedStyle(el);
    var avail = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    t.style.whiteSpace = "nowrap";
    var w = t.scrollWidth, fs = parseFloat(getComputedStyle(t).fontSize);
    if (w > avail && w > 0) t.style.fontSize = Math.max(34, Math.floor(fs * avail / w * 0.97)) + "px";
    if (t.scrollWidth > avail) t.style.whiteSpace = "normal";   // at the floor: break between words only
  }

  // ---- leaving ------------------------------------------------------------
  function go(href, title, eyebrow) {
    if (REDUCED) { location.href = href; return; }
    try { sessionStorage.setItem(KEY, JSON.stringify({ t: title, e: eyebrow || "" })); } catch (e) {}
    var el = build(title, eyebrow);
    el.getBoundingClientRect();              // commit the start state
    el.classList.add("on");
    setTimeout(function () { el.classList.add("drawn"); }, 260);
    setTimeout(function () { el.classList.add("filled"); }, 760);
    setTimeout(function () { location.href = href; }, 1180);
  }

  document.addEventListener("click", function (ev) {
    var a = ev.target.closest && ev.target.closest("a[data-transition]");
    if (!a) return;
    if (ev.defaultPrevented || ev.button !== 0 || ev.metaKey || ev.ctrlKey ||
        ev.shiftKey || ev.altKey || a.target === "_blank") return;
    ev.preventDefault();
    go(a.href, a.getAttribute("data-transition"), a.getAttribute("data-eyebrow"));
  });

  // ---- arriving -----------------------------------------------------------
  function arrive() {
    var raw = null;
    try { raw = sessionStorage.getItem(KEY); sessionStorage.removeItem(KEY); } catch (e) {}
    var root = document.documentElement;
    if (!raw || REDUCED) { root.classList.remove("dr-entering"); return; }
    var d; try { d = JSON.parse(raw); } catch (e) { d = { t: "", e: "" }; }
    var el = build(d.t, d.e, true);
    el.style.clipPath = "circle(150% at 100% 50%)";
    root.classList.remove("dr-entering");     // the real curtain now covers
    // Timers, not animation frames: a page restored into a background tab must
    // still end up uncovered.
    setTimeout(function () {
      el.classList.remove("still");
      el.classList.add("leave");
      el.style.clipPath = "circle(0% at 100% 50%)";
    }, 340);
    setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 1400);
  }

  // Back/forward can restore a page with the curtain still drawn over it.
  window.addEventListener("pageshow", function (ev) {
    if (!ev.persisted) return;
    document.documentElement.classList.remove("dr-entering");
    var el = document.querySelectorAll(".drx");
    for (var i = 0; i < el.length; i++) el[i].parentNode.removeChild(el[i]);
  });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", arrive);
  else arrive();

  window.DRTransition = { go: go };
})();
