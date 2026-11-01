// Setup: npm install --prefix dashboard/data/browser-tools playwright --no-audit --no-fund
// Start local gateway with a synthetic workspace, then: node code/tests/test_browser.mjs
import { chromium } from '../../dashboard/data/browser-tools/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
try {
  await page.goto('http://127.0.0.1:8000/');
  await page.getByRole('heading', { name: 'Your hands, in 3D' }).waitFor();
  await page.getByRole('button', { name: 'Upload a clip', exact: true }).click();
  await page.getByRole('heading', { name: 'Add a head-camera clip' }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Play', exact: true }).isDisabled(), true);
  await page.getByRole('button', { name: 'Collapse navigation' }).click();
  assert.equal(Math.round((await page.locator('.app-sidebar').boundingBox()).width), 66);
  await page.getByRole('button', { name: 'Expand navigation' }).click();
  await page.goto('http://127.0.0.1:8000/#/workspace/bbbbbbbbbbbb');
  await page.getByRole('button', { name: 'Play', exact: true }).waitFor();
  await page.locator('.scene-pane canvas').waitFor();
  await page.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
  assert.equal(await page.locator('.quality-lane').count(), 6);
  await page.getByRole('tab', { name: 'Quality', exact: true }).click();
  await page.getByRole('heading', { name: 'Clip rating', exact: true }).waitFor();
  await page.getByRole('tab', { name: 'Clip', exact: true }).click();
  await page.getByRole('heading', { name: 'Processing times', exact: true }).waitFor();
  await page.getByRole('tab', { name: 'Frame', exact: true }).click();
  await page.getByRole('heading', { name: 'Left hand', exact: true }).waitFor();
  const splitter = page.getByRole('button', { name: 'Resize split between video and 3D view' });
  const divider = await splitter.boundingBox();
  const originalWidth = (await page.locator('.video-pane').boundingBox()).width;
  await page.mouse.move(divider.x + divider.width / 2, divider.y + 100);
  await page.mouse.down(); await page.mouse.move(divider.x - 70, divider.y + 100); await page.mouse.up();
  assert.ok((await page.locator('.video-pane').boundingBox()).width < originalWidth - 20);
  const handle = await page.getByRole('button', { name: 'Resize timeline', exact: true }).boundingBox();
  const heightBefore = (await page.locator('.timeline-dock').boundingBox()).height;
  await page.mouse.move(handle.x + 100, handle.y + 5); await page.mouse.down();
  await page.mouse.move(handle.x + 100, handle.y - 50); await page.mouse.up();
  assert.ok((await page.locator('.timeline-dock').boundingBox()).height > heightBefore + 20);
  for (const preset of ['Top', 'Side', 'Behind', "Wearer's eye", 'Reset']) {
    await page.getByLabel('Camera view').selectOption(preset);
    await page.waitForTimeout(100);
  }
  const slider = page.getByRole('slider', { name: 'Video frame' });
  await slider.fill('150');
  await page.waitForFunction(() => Math.abs(document.querySelector('video').currentTime - 5) < .1);
  await page.getByRole('button', { name: 'Previous frame' }).click();
  assert.equal(await slider.inputValue(), '149');
  await page.getByRole('button', { name: 'Next frame' }).click();
  assert.equal(await slider.inputValue(), '150');
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('video').currentTime > 5, { timeout: 10000 });
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.locator('.segment').first().click();
  const canvas = page.locator('.scene-pane canvas');
  await page.screenshot({ path: "dashboard/data/browser-qa.png", fullPage: true });
  const rect = await canvas.boundingBox();
  await page.mouse.move(rect.x + 100, rect.y + 100);
  await page.mouse.down(); await page.mouse.move(rect.x + 220, rect.y + 160, { steps: 10 }); await page.mouse.up();
  await page.mouse.wheel(0, 100);
  await page.mouse.down({ button: 'right' }); await page.mouse.move(rect.x + 260, rect.y + 170, { steps: 5 }); await page.mouse.up({ button: 'right' });
  // Record one second through actual MediaRecorder and gateway conversion.
  await page.getByRole('button', { name: 'Exports', exact: true }).click();
  await page.getByLabel('From frame').fill('0');
  await page.getByLabel('To frame').fill('29');
  const downloaded = page.waitForEvent('download', { timeout: 45000 });
  await page.getByRole('button', { name: 'Export my 3D view MP4' }).click();
  const download = await downloaded;
  assert.equal(download.suggestedFilename(), 'view.mp4');
  assert.equal(await page.locator('.error').count(), 0);
  await page.getByRole('link', { name: 'Library', exact: true }).click();
  await page.getByRole('link', { name: 'Synthetic QA' }).waitFor();
  await page.getByRole('link', { name: 'Settings', exact: true }).click();
  await page.getByRole('button', { name: 'Test connection', exact: true }).waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2));
  assert.deepEqual(errors, []);
  console.log('BROWSER CHECKS PASSED: reference landing/empty/ready layouts, inspector tabs, collapsible navigation, viewer/timeline resize, presets, seek, frame stepping, playback, lanes, orbit/pan/zoom, real MediaRecorder -> MP4, navigation, mobile width');
} finally { await browser.close(); }
