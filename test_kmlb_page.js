// Node test for the KMLB page: stubbed DOM, real page JS, the harness site (truncated volume).
//   node test_kmlb_page.js [site_dir] [index.html]
const fs = require("fs"), vm = require("vm"), path = require("path");
const SITE = process.argv[2] || "site";
const html = fs.readFileSync(process.argv[3] || path.join(__dirname, "web/index.html"), "utf8");
const js = html.split("<script>")[1].split("</script>")[0];
const els = {};
function ctx2d(){ const b = { createImageData: (w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h * 4)) }),
  measureText: t => ({ width: t.length * 6 }) };
  return new Proxy(b, { get: (t, k) => (k in t ? t[k] : () => ({})), set: (t, k, v) => (t[k] = v, true) }); }
function mk(id){ const el = { id, _l: {}, innerHTML: "", textContent: "", hidden: false, value: 0, max: 0,
  checked: id === "rings", naturalWidth: 1100, naturalHeight: 1250, clientWidth: 1100, style: { setProperty(){} },
  classList: { add(){}, remove(){} }, setAttribute(){}, scrollIntoView(){},
  addEventListener(t, fn){ (this._l[t] = this._l[t] || []).push(fn); },
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 1100, height: 1250 }) };
  const c = ctx2d(); el.getContext = () => c; return el; }
const ctx = { document: { getElementById: id => els[id] || (els[id] = mk(id)), querySelectorAll: () => [] },
  console, Math, Date, JSON, Promise, Uint8Array, Infinity, encodeURIComponent, isFinite, parseInt, Number, String,
  Object, Array, Set, Blob: class {}, matchMedia: () => ({ matches: false }), addEventListener(){}, setInterval(){},
  clearInterval(){}, Image: class { set src(v){} },
  fetch: async url => { const p = path.join(SITE, url.split("?")[0]); if (!fs.existsSync(p)) return { ok: false, status: 404 };
    const b = fs.readFileSync(p); return { ok: true, json: async () => JSON.parse(b),
      arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.length) }; } };
ctx.window = ctx; vm.createContext(ctx); vm.runInContext(js, ctx);
const strip = h => h.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
const run = c => vm.runInContext(c, ctx);
setTimeout(async () => {
  let bad = 0; const ok = (c, m) => { console.log((c ? "  ok    " : "  FAIL  ") + m); if (!c) bad++; };
  ok(run("M && M.frames.length") > 0, "manifest loaded");
  ok(/Page kmlb-v2/.test(els.method.innerHTML) && /Radar\./.test(els.method.innerHTML), "footer: kmlb-v2 and the radar method note");
  const sl = strip(els.srcline.innerHTML);
  ok(/^Radar: KMLB \d\d:\d\dZ, VCP 215, to 6\.4° — sampled to only 6\.8 km over LC-39A, so the pads use MRMS/.test(sl), "source line: " + sl.split("Scored")[0]);
  const S = run("M.sites");
  run(`readout(${S["LC-39A"][0]}, ${S["LC-39A"][1]}, "LC-39A")`); await new Promise(r => setTimeout(r, 40));
  ok(/Radar MRMS here - KMLB samples this column only to 6\.8 km/.test(strip(els.readout.innerHTML)), "LC-39A readout: MRMS here, KMLB only to 6.8 km");
  run(`readout(28.15, -80.65, null)`); await new Promise(r => setTimeout(r, 40));
  ok(/Radar MRMS only - KMLB does not sample this column|Radar MRMS here - KMLB samples/.test(strip(els.readout.innerHTML)), "over the radar: " + (strip(els.readout.innerHTML).match(/Radar [^A-Z]*[A-Za-z ,.0-9-]*/) || [""])[0]);
  run(`readout(29.55, -81.55, null)`); await new Promise(r => setTimeout(r, 40));
  ok(/Radar KMLB through the column/.test(strip(els.readout.innerHTML)), "190 km out, the 6.4° beam is above 15 km: 'through the column'");
  run(`frameAt(idx).kmlb = Object.assign({}, frameAt(idx).kmlb, {mode: "full", top_deg: 19.5, vcp: 212, lc39a_top_km: 20.2}); srcLine(frameAt(idx))`);
  ok(/Radar: KMLB \d\d:\d\dZ, VCP 212, to 19\.5° , full depth over the pads/.test(strip(els.srcline.innerHTML)), "full-depth wording");
  run(`frameAt(idx).kmlb = {mode: "none", note: "no KMLB volume within 6 min of this frame"}; srcLine(frameAt(idx)); readout(${S["LC-39A"][0]}, ${S["LC-39A"][1]}, "LC-39A")`);
  await new Promise(r => setTimeout(r, 40));
  ok(/Radar: KMLB unavailable \(no KMLB volume within 6 min of this frame\), MRMS only/.test(strip(els.srcline.innerHTML)), "unavailable wording");
  ok(/Radar MRMS only: no KMLB volume within 6 min/.test(strip(els.readout.innerHTML)), "readout says MRMS only, with the reason");
  run(`DATA["old.bin"] = DATA[frameAt(idx).data].slice(0, 14 * M.grid.ny * M.grid.nx); frameAt(idx).data = "old.bin"; delete frameAt(idx).kmlb; show(idx); readout(${S["LC-39A"][0]}, ${S["LC-39A"][1]}, "LC-39A")`);
  await new Promise(r => setTimeout(r, 40));
  ok(!/KMLB samples this column|KMLB through the column/.test(els.readout.innerHTML), "frame without plane 14 / kmlb info: no radar row, no crash");
  ok(!/Radar:/.test(els.srcline.innerHTML), "and no radar item in the source line");
  console.log(`\n${bad} failure(s)`); process.exit(bad ? 1 : 0);
}, 300);
