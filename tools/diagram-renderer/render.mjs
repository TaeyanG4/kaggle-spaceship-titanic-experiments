// Render docs/diagrams/*.dot to docs/assets/<name>.svg and .png and record hashes.
// Uses Viz.js (Graphviz compiled to WebAssembly) and sharp; no model training or network access.
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { instance } from "@viz-js/viz";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const sourceDir = join(root, "docs", "diagrams");
const assetDir = join(root, "docs", "assets");
const sha = (buffer) => createHash("sha256").update(buffer).digest("hex");
const viz = await instance();
const manifest = { renderer: `@viz-js/viz ${viz.graphvizVersion}`, diagrams: {} };

for (const file of readdirSync(sourceDir).filter((f) => f.endsWith(".dot")).sort()) {
  const name = file.replace(/\.dot$/, "");
  const source = readFileSync(join(sourceDir, file), "utf8");
  const svg = viz.renderString(source, { format: "svg", engine: "dot" });
  writeFileSync(join(assetDir, `${name}.svg`), svg);
  const png = await sharp(Buffer.from(svg), { density: 144 })
    .resize({ width: 1800, height: 1800, fit: "inside", withoutEnlargement: true })
    .flatten({ background: "#ffffff" }).png().toBuffer();
  writeFileSync(join(assetDir, `${name}.png`), png);
  const meta = await sharp(png).metadata();
  manifest.diagrams[name] = {
    source_sha256: sha(Buffer.from(source)),
    svg_sha256: sha(Buffer.from(svg)),
    png_sha256: sha(png),
    png_size: [meta.width, meta.height],
  };
  console.log(`rendered ${name} (${meta.width}x${meta.height})`);
}
writeFileSync(join(sourceDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
