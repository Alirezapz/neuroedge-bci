/* Optional browser QA. Requires Playwright; never needed for EEG training.
 * node scripts/check_demo.cjs runs/main/demo/replay.html
 * NEUROEDGE_PLAYWRIGHT_MODULE and NEUROEDGE_CHROMIUM_BINARY can select local QA dependencies.
 */
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const assert = require('node:assert/strict');
const pw = require(process.env.NEUROEDGE_PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const filename = path.resolve(process.argv[2] || 'runs/main/demo/replay.html');
  const browser = await pw.chromium.launch({
    executablePath: process.env.NEUROEDGE_CHROMIUM_BINARY || undefined,
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
           '--disable-software-rasterizer', '--disable-gpu-compositing', '--use-gl=disabled']
  });
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 1000}});
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto(pathToFileURL(filename).href);
    assert.match(await page.title(), /NeuroEdge/);
    assert.equal(await page.locator('#truth').textContent(), '');
    await page.locator('#reveal').check();
    assert.match(await page.locator('#truth').textContent(), /Task:/);
    await page.locator('#next').click();
    assert.equal(await page.locator('#trial-select').inputValue(), '1');
    await page.locator('#play').click();
    assert.match(await page.locator('#decision').textContent(), /Acquiring/);
    assert.equal(await page.locator('#truth').textContent(), '');
    await page.locator('#play').click();
    await page.locator('#timeline').evaluate(el => {
      el.value = '3'; el.dispatchEvent(new Event('input', {bubbles:true}));
    });
    assert.match(await page.locator('#status').textContent(), /Complete window/);
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    const result = {
      status: 'PASS', browser: browser.version(),
      checks: ['page loads without script errors', 'task label hidden by default',
               'ground truth reveal', 'next trial selection', 'replay begins',
               'prediction and task label withheld during partial replay',
               'pause/seek controls', 'mobile layout without horizontal overflow'],
      desktop_viewport: [1280, 1000], mobile_viewport: [390, 844],
      note: 'UI software checks; not evidence of scientific or clinical validity.'
    };
    fs.writeFileSync(path.join(path.dirname(filename), 'browser_qa.json'), JSON.stringify(result,null,2));
    console.log(JSON.stringify(result,null,2));
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exitCode = 1; });

