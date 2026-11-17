const puppeteer = require('../frontend/node_modules/puppeteer-core');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE_URL = 'http://127.0.0.1:8000';

const ROUTES = [
  '/',
  '/live',
  '/image',
  '/video',
  '/cnn',
  '/yolo',
  '/history',
  '/analytics',
  '/models',
  '/settings',
  '/camera'
];

async function verifyAllRoutes() {
  console.log('🚀 Starting Automated Headless QA with Microsoft Edge...');
  const browser = await puppeteer.launch({
    executablePath: EDGE_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  let allPassed = true;
  const results = [];

  for (const route of ROUTES) {
    const url = `${BASE_URL}${route}`;
    const errors = [];
    const warnings = [];

    const onConsole = (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
      else if (msg.type() === 'warning') warnings.push(msg.text());
    };
    const onPageError = (err) => {
      errors.push(err.toString());
    };

    page.on('console', onConsole);
    page.on('pageerror', onPageError);

    try {
      const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 5000 });
      const status = response ? response.status() : 'No response';
      
      // Wait a moment for React effects and renders
      await new Promise(r => setTimeout(r, 800));

      const title = await page.title();
      const rootText = await page.evaluate(() => document.getElementById('root')?.innerText || '');
      const hasContent = rootText.length > 20;

      // Filter out expected WebSocket or mock camera warnings if any
      const criticalErrors = errors.filter(e => !e.includes('favicon') && !e.includes('MediaStream'));

      const passed = status === 200 && criticalErrors.length === 0 && hasContent;
      if (!passed) allPassed = false;

      results.push({
        route,
        status,
        passed,
        errors: criticalErrors,
        snippet: rootText.slice(0, 80).replace(/\n/g, ' ')
      });

      console.log(`[${passed ? 'PASS' : 'FAIL'}] ${route.padEnd(12)} Status: ${status} | Errors: ${criticalErrors.length} | Preview: ${rootText.slice(0, 45).replace(/\n/g, ' ')}...`);
      if (criticalErrors.length > 0) {
        console.log(`   ⚠️ Errors in ${route}:`, criticalErrors);
      }
    } catch (err) {
      allPassed = false;
      console.log(`[FAIL] ${route.padEnd(12)} Exception: ${err.message}`);
      results.push({ route, passed: false, error: err.message });
    } finally {
      page.off('console', onConsole);
      page.off('pageerror', onPageError);
    }
  }

  await browser.close();
  console.log('\n' + '='.repeat(60));
  if (allPassed) {
    console.log('🎉 ALL 11 ROUTES RENDERED CLEANLY WITHOUT CONSOLE ERRORS!');
  } else {
    console.log('⚠️ SOME ROUTES HAD ERRORS');
  }
  console.log('='.repeat(60));
  process.exit(allPassed ? 0 : 1);
}

verifyAllRoutes().catch(err => {
  console.error('Test runner fatal error:', err);
  process.exit(1);
});
