// Prerequisite: local dashboard on :8000; ignored Playwright installation as in code/tests/test_browser.mjs.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from '../data/browser-tools/node_modules/playwright/index.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
execFileSync(root + '.venv/Scripts/python.exe', ['tools/generate_room_fixture.py', '--video'], { cwd: root });
const fixture = new URL('../frontend/src/__fixtures__/room-twin/synthetic-room/', import.meta.url);
const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const errors = []; page.on('pageerror', e => errors.push(String(e)));
  await page.route('**/api/rooms**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/api/rooms') return route.fulfill({ json: [{ id: 'synthetic-room', has_video: true, has_render: true, has_robot: true }] });
    const name = pathname.split('/').pop();
    const file = (name === 'video' || name === 'slam') ? 'original.mp4' : name === 'stats' ? 'room_stats.json' : `web/${name}`;
    try {
      const body = await readFile(new URL(file, fixture));
      if (file.endsWith('.mp4')) {
        const match = /bytes=(\d+)-(\d*)/.exec(route.request().headers().range ?? '');
        if (match) {
          const start = Number(match[1]), end = match[2] ? Math.min(Number(match[2]), body.length - 1) : body.length - 1;
          return route.fulfill({ status: 206, body: body.subarray(start, end + 1), contentType: 'video/mp4',
            headers: { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${body.length}`, 'Content-Length': String(end - start + 1) } });
        }
      }
      await route.fulfill({ body, contentType: file.endsWith('.mp4') ? 'video/mp4' : file.endsWith('.json') ? 'application/json' : 'application/octet-stream' });
    } catch { await route.fulfill({ status: 404, body: 'Missing fixture' }); }
  });
  await page.goto('http://127.0.0.1:8000/#/room/synthetic-room');
  const hands = page.locator('.room-hands-view'), robot = page.locator('.room-robot-view');
  await hands.locator('canvas[data-room-scene="ready"]').waitFor();
  assert.equal(await robot.count(), 0, 'Default view must be interactive hands only');
  const slider = page.getByRole('slider', { name: 'Room frame' });
  const handCanvas = hands.locator('canvas');
  await hands.getByRole('button', { name: 'Pan', exact: true }).click();
  const bounds = await handCanvas.boundingBox();
  const handBefore = await handCanvas.screenshot();
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down(); await page.mouse.move(bounds.x + bounds.width / 2 + 70, bounds.y + bounds.height / 2 + 15, { steps: 8 }); await page.mouse.up();
  await page.waitForTimeout(600);
  assert.ok(!handBefore.equals(await handCanvas.screenshot()), 'Hand view must support panning');
  await hands.getByRole('button', { name: 'Pan', exact: true }).click();
  await hands.getByRole('button', { name: 'Rotate camera', exact: true }).click();
  const rotatingHand = await handCanvas.screenshot(); await page.waitForTimeout(700);
  assert.ok(!rotatingHand.equals(await handCanvas.screenshot()), 'Hand camera must rotate while playback is paused');
  await hands.getByRole('button', { name: 'Reset view', exact: true }).click();
  await page.getByRole('button', { name: 'Show robot replay', exact: true }).click();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await robot.locator('canvas[data-room-scene="ready"]').waitFor();
  await page.getByLabel('Hands camera view').selectOption('top');
  await page.getByLabel('Robot camera view').selectOption('top');
  await page.waitForTimeout(700);
  const robotCanvas = robot.locator('canvas');
  const stationaryRobot = await robotCanvas.screenshot();
  const handsBox = await handCanvas.boundingBox();
  await page.mouse.move(handsBox.x + handsBox.width / 2, handsBox.y + handsBox.height / 2);
  await page.mouse.down(); await page.mouse.move(handsBox.x + handsBox.width / 2 + 60, handsBox.y + handsBox.height / 2 + 20, { steps: 10 }); await page.mouse.up();
  await page.waitForTimeout(700);
  assert.ok(stationaryRobot.equals(await robotCanvas.screenshot()), 'Orbiting hands must not move the robot camera');
  await page.waitForTimeout(500);
  const stationaryHands = await handCanvas.screenshot();
  await robot.getByRole('button', { name: 'Rotate camera', exact: true }).click();
  const rotatingRobot = await robotCanvas.screenshot(); await page.waitForTimeout(700);
  assert.ok(!rotatingRobot.equals(await robotCanvas.screenshot()), 'Robot camera must rotate');
  assert.ok(stationaryHands.equals(await handCanvas.screenshot()), 'Robot auto-rotation must not move the hands camera');
  await robot.getByRole('button', { name: 'Reset view', exact: true }).click();
  await slider.fill('0'); await page.waitForTimeout(100); const pose0 = await robotCanvas.screenshot();
  await slider.fill('9'); await page.waitForTimeout(100);
  assert.ok(!pose0.equals(await robotCanvas.screenshot()), 'Slider must move robot links');
  await page.getByRole('button', { name: 'Robot L Out of reach frames 4–5', exact: true }).press('Enter');
  assert.equal(await slider.inputValue(), '4');
  await page.getByRole('button', { name: 'Original video', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.room-video video')?.readyState >= 2);
  await slider.fill('45');
  await page.waitForFunction(() => Math.abs(document.querySelector('.room-video video').currentTime - 1.5) < .05);
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.getByRole('button', { name: 'Hide robot replay', exact: true }).click();
  assert.equal(await robot.count(), 0); assert.equal(await handCanvas.count(), 1);
  assert.equal(await page.getByRole('button', { name: 'Pause', exact: true }).count(), 1);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.getByRole('button', { name: 'Show robot replay', exact: true }).click();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await slider.fill('70');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Record my view', exact: true }).click();
  assert.ok((await download).suggestedFilename().endsWith('.webm'));
  assert.equal(errors.length, 0, errors.join('\n'));
  await page.setViewportSize({ width: 600, height: 900 });
  assert.ok((await hands.boundingBox()).width < 600);
  console.log('PASS: hands pan/orbit/auto-rotation, independent robot camera, shared replay/grades, original video sync, recording and mobile');
} finally { await browser.close(); }
