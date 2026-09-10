import fs from "node:fs";
// Minimal DOM shim: enough to run the real render path and surface any throw.
const mk = (tag) => ({
  tag, attrs: {}, children: [], style: {}, dataset: {}, _text: "",
  setAttribute(k, v) { if (v === undefined || v === null || Number.isNaN(v)) throw new Error(`${tag}.${k} = ${v}`); this.attrs[k] = v; },
  appendChild(c) { this.children.push(c); return c; },
  replaceChildren(...c) { this.children = c; },
  querySelector() { return mk("stub"); },
  scrollIntoView() {},
  set textContent(v) { this._text = String(v); },
  get textContent() { return this._text; },
  set innerHTML(v) { this._html = v; },
  set onclick(f) { this._onclick = f; },
});
const byId = {};
for (const id of ["sub","diagnosis","tiles","timeline","ttft","tpot","rows","detail"]) byId[id] = mk("div");
globalThis.document = {
  createElementNS: (_ns, t) => mk(t),
  createElement: (t) => mk(t),
  getElementById: (id) => byId[id],
  documentElement: {},
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#123456" });
globalThis.fetch = async () => ({ ok: true, json: async () => ({}) });
globalThis.setInterval = () => {};

let src = fs.readFileSync(process.argv[2], "utf8");
src = src.replace(/^tick\(\);$/m, "").replace(/^setInterval\(tick, 2000\);$/m, "");
const mod = new Function(src + "\nreturn {render, timelineChart, histogram};")();

let failures = 0;
for (const f of fs.readdirSync(process.argv[3]).filter(n => n.startsWith("payload-"))) {
  const p = JSON.parse(fs.readFileSync(`${process.argv[3]}/${f}`, "utf8"));
  try {
    mod.render(p);
    const svg = mod.timelineChart(p.timeline);
    const h = mod.histogram(p.tpot, "#000");
    const rows = byId.rows.children.length;
    const tiles = byId.tiles.children.length;
    console.log(`  OK   ${f.padEnd(34)} rows=${String(rows).padStart(3)} tiles=${tiles} ` +
                `timeline_nodes=${svg.children.length} hist_bars=${h.children.length}`);
  } catch (e) {
    failures++;
    console.log(`  FAIL ${f}: ${e.message}`);
  }
}
process.exit(failures ? 1 : 0);
