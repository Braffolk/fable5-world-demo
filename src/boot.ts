/** Pre-boot entry point — quality-preset selection BEFORE the engine loads.
 *
 * This module must stay dependency-free of the engine graph (only
 * core/Quality, which is pure DOM/URL code): the whole point is that the
 * main module graph — where many knobs are read from location.search — only
 * loads AFTER the preset params are injected into the URL. `import('./main')`
 * is the barrier.
 *
 * HARNESS LAW: automation boots must be completely unaffected — no UI, no
 * delay. Skipped (auto-resolve) when:
 *   - navigator.webdriver (any Playwright/Selenium boot), or
 *   - ?quality= present (explicit choice — also what the picker stamps), or
 *   - ?scene= present (every probe URL passes ?scene=…; the product URL is
 *     bare — humans never type it).
 * Auto-resolve WITHOUT ?quality is MEDIUM (the shipped default).
 */

import {
  PRESET_LABELS,
  applyPreset,
  normalizeTier,
  setResolvedTier,
  type QualityTier,
} from './core/Quality';

const AUTO_CONTINUE_S = 5;
const STORAGE_KEY = 'laas.quality';

/** Last interactive choice (localStorage) — used to PRESELECT the picker on a
 *  return visit and to seed the countdown's auto-pick. The picker still shows on
 *  every interactive boot (mandate: "asks again" + 5 s auto-continue); the memory
 *  only changes which preset is highlighted/auto-continued. Absent/invalid ⇒ null. */
function loadRemembered(): QualityTier | null {
  try {
    return normalizeTier(localStorage.getItem(STORAGE_KEY));
  } catch {
    return null; // private-mode / disabled storage — degrade to no memory
  }
}
function saveRemembered(tier: QualityTier): void {
  try {
    localStorage.setItem(STORAGE_KEY, tier);
  } catch {
    /* storage unavailable — non-fatal, just no memory next boot */
  }
}

/** Minimal DOM overlay: three buttons, `preselect` highlighted, 5 s countdown
 *  auto-continue so returning users aren't blocked. The chosen tier is persisted
 *  (localStorage) and preselected next interactive boot; the picker still appears
 *  every time. Any key/pointer interaction cancels the countdown; click (or Enter)
 *  confirms. Styling matches the #boot overlay in index.html. */
function showQualityPicker(preselect: QualityTier): Promise<QualityTier> {
  return new Promise((resolve) => {
    const tiers: QualityTier[] = ['low', 'medium', 'high'];
    let selected = preselect;
    let countdown: number | null = AUTO_CONTINUE_S;

    const root = document.createElement('div');
    root.id = 'quality-picker';
    root.style.cssText = [
      'position:fixed', 'inset:0', 'z-index:20', 'display:flex',
      'flex-direction:column', 'align-items:center', 'justify-content:center',
      'background:#06080a', 'color:#8aa39b',
      "font-family:ui-monospace,'SF Mono',Menlo,monospace",
      'font-size:13px', 'letter-spacing:0.08em',
    ].join(';');

    const title = document.createElement('div');
    title.textContent = 'L A A S';
    title.style.cssText = 'font-size:22px;color:#c8d8d0;letter-spacing:0.35em;margin-bottom:6px';
    const sub = document.createElement('div');
    sub.textContent = 'quality';
    sub.style.cssText = 'margin-bottom:22px;color:#5a6f66';
    root.append(title, sub);

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:14px;flex-wrap:wrap;justify-content:center;max-width:820px';
    root.append(row);

    const buttons = new Map<QualityTier, HTMLButtonElement>();
    const paint = (): void => {
      for (const [t, b] of buttons) {
        const on = t === selected;
        b.style.borderColor = on ? '#5fae8f' : '#1a2420';
        b.style.background = on ? '#0c1512' : '#080d0b';
        b.style.color = on ? '#c8d8d0' : '#8aa39b';
      }
    };

    const hint = document.createElement('div');
    hint.style.cssText = 'margin-top:24px;color:#5a6f66;min-height:16px';
    const paintHint = (): void => {
      hint.textContent =
        countdown !== null
          ? `starting ${PRESET_LABELS[selected].title} in ${countdown}s — click to change`
          : 'click a preset to start';
    };

    const finish = (tier: QualityTier): void => {
      if (timer !== null) clearInterval(timer);
      window.removeEventListener('keydown', onKey, true);
      saveRemembered(tier);
      root.remove();
      resolve(tier);
    };
    const cancelCountdown = (): void => {
      if (countdown === null) return;
      countdown = null;
      if (timer !== null) clearInterval(timer);
      paintHint();
    };

    for (const t of tiers) {
      const b = document.createElement('button');
      b.type = 'button';
      b.style.cssText = [
        'cursor:pointer', 'width:220px', 'padding:16px 14px', 'text-align:left',
        'border:1px solid #1a2420', 'border-radius:2px', 'background:#080d0b',
        'font:inherit', 'letter-spacing:inherit', 'transition:all 0.15s ease',
      ].join(';');
      const h = document.createElement('div');
      h.textContent = PRESET_LABELS[t].title;
      h.style.cssText = 'font-size:15px;letter-spacing:0.25em;margin-bottom:8px';
      const d = document.createElement('div');
      d.textContent = PRESET_LABELS[t].desc;
      d.style.cssText = 'font-size:11px;line-height:1.5;color:#5a6f66';
      b.append(h, d);
      b.addEventListener('pointerenter', cancelCountdown);
      b.addEventListener('click', () => finish(t));
      row.append(b);
      buttons.set(t, b);
    }

    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Enter') return finish(selected);
      cancelCountdown();
      const i = tiers.indexOf(selected);
      if (e.key === 'ArrowLeft') selected = tiers[Math.max(0, i - 1)]!;
      else if (e.key === 'ArrowRight') selected = tiers[Math.min(tiers.length - 1, i + 1)]!;
      else if (e.key === '1') selected = 'low';
      else if (e.key === '2') selected = 'medium';
      else if (e.key === '3') selected = 'high';
      paint();
      paintHint();
    };
    window.addEventListener('keydown', onKey, true);

    root.append(hint);
    paint();
    paintHint();
    const timer = window.setInterval(() => {
      if (countdown === null) return;
      countdown -= 1;
      if (countdown <= 0) finish(selected);
      else paintHint();
    }, 1000);

    // the boot progress overlay sits under us (z-index 10 vs 20) — leave it;
    // it becomes visible the instant the picker resolves and boot proceeds
    document.body.append(root);
  });
}

async function preboot(): Promise<void> {
  const q = new URLSearchParams(window.location.search);
  const explicit = normalizeTier(q.get('quality'));
  let tier: QualityTier;
  if (explicit) {
    tier = explicit; // ?quality= always wins and always skips the UI
  } else if (navigator.webdriver || q.has('scene') || q.has('quality')) {
    tier = 'medium'; // automation/probe boot — zero delay, shipped default
  } else {
    tier = await showQualityPicker(loadRemembered() ?? 'medium');
  }
  setResolvedTier(tier); // downstream grid sizing + bootcache key read this
  applyPreset(tier);
  await import('./main'); // main.ts self-runs its boot() on import
}

preboot().catch((e: unknown) => {
  // main.ts has its own failLoud; this only covers preboot/import failures
  const el = document.getElementById('boot-msg');
  if (el) el.textContent = `boot failed: ${e instanceof Error ? e.message : String(e)}`;
  // eslint-disable-next-line no-console
  console.error('[laas] preboot failed', e);
});
