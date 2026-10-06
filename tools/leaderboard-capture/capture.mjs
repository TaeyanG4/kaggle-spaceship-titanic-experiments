// Capture the public Spaceship Titanic leaderboard (top rows down to this project's team) as PNGs.
// Uses the locally installed Chrome through puppeteer-core with a fresh temporary profile (no login,
// no cookies kept). The cookie notice is hidden for the screenshot only; nothing is clicked or accepted.
// Usage: npm ci && node capture.mjs [teamName]   (CHROME_PATH overrides the Chrome location)
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const team = process.argv[2] ?? "Taeyang";
const url = "https://www.kaggle.com/competitions/spaceship-titanic/leaderboard";
const assets = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "docs", "assets");
const chrome = process.env.CHROME_PATH ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const profile = mkdtempSync(join(tmpdir(), "lb-capture-"));

const browser = await puppeteer.launch({ executablePath: chrome, headless: true, userDataDir: profile,
  args: ["--no-first-run", "--no-default-browser-check"] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1600, deviceScaleFactor: 2 });
  await page.goto(url, { waitUntil: "networkidle2", timeout: 90000 });
  await page.waitForFunction((t) => document.body.innerText.includes(t), { timeout: 60000 }, team);

  const box = await page.evaluate((t) => {
    // hide the cookie notice and fixed overlays for the screenshot only
    for (const el of document.querySelectorAll("body *")) {
      const style = getComputedStyle(el);
      if ((style.position === "fixed" || style.position === "sticky") && el.innerText?.includes("cookies")) {
        el.style.display = "none";
      }
    }
    const leaves = [...document.querySelectorAll("body *")].filter(
      (el) => el.children.length === 0 && el.textContent.trim() === t);
    if (!leaves.length) return null;
    let row = leaves[0];
    while (row.parentElement && row.getBoundingClientRect().width < 700) row = row.parentElement;
    const header = [...document.querySelectorAll("body *")].find(
      (el) => el.children.length === 0 && el.textContent.trim() === "This leaderboard is calculated with all of the test data.");
    const r = row.getBoundingClientRect();
    const h = header ? header.getBoundingClientRect() : r;
    return { rowTop: r.top + scrollY, rowBottom: r.bottom + scrollY, left: r.left, width: r.width,
             top: h.top + scrollY - 70 };
  }, team);
  if (!box) throw new Error(`team "${team}" not found on the leaderboard page`);

  const pad = 12;
  await page.screenshot({ path: join(assets, "kaggle-leaderboard.png"),
    clip: { x: box.left - pad, y: box.top, width: box.width + 2 * pad,
            height: box.rowBottom - box.top + pad } });
  await page.screenshot({ path: join(assets, "kaggle-leaderboard-row.png"),
    clip: { x: box.left - pad, y: box.rowTop - pad, width: box.width + 2 * pad,
            height: box.rowBottom - box.rowTop + 2 * pad } });
  console.log("captured", new Date().toISOString(), box);
} finally {
  await browser.close();
  rmSync(profile, { recursive: true, force: true });
}
