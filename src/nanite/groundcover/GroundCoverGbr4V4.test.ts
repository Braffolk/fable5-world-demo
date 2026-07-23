import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import {
  CALAMAGROSTIS_GBR4_V4_REFERENCE_URL,
  Gbr4V4RecordMode,
  parseGbr4V4Profile,
} from './GroundCoverGbr4V4';

test('loads the honest Calamagrostis GBR4/v4 reference and charges upload bytes', () => {
  const bytes = readFileSync(fileURLToPath(CALAMAGROSTIS_GBR4_V4_REFERENCE_URL));
  const profile = parseGbr4V4Profile(bytes);
  assert.equal(profile.profileId, 2);
  assert.equal(profile.faces.length, 6);
  assert.equal(profile.levels.length, 4);
  assert.equal(profile.boundaryCells, 2);
  assert.equal(profile.directionCells, 4);
  assert.equal(profile.fineBoundaryCells, 4096);
  assert.equal(profile.fineDirectionCells, 65535);
  assert.equal(profile.recordModeCounts[Gbr4V4RecordMode.CERTIFIED_REGULAR], 0);
  assert.equal(profile.recordModeCounts[Gbr4V4RecordMode.FILTERED_MIXED], profile.addressRecordCount);
  assert.equal(profile.correctionCount, 8);
  assert.ok(profile.correctionSeed > 0);
  assert.equal(profile.regularRecordCount, 8);
  assert.equal(profile.maximumTerminalReads, 6);
  assert.equal(profile.publicationStatus, 'REFERENCE_RED');
  assert.equal(profile.runtimeBindAllowed, false);
  assert.ok(profile.gpuResidentBytes > 0);
  assert.ok(profile.gpuResidentBytes < profile.containerBytes);
  profile.addressTexture.dispose();
  profile.mixedPayloadTexture.dispose();
  profile.regularPayloadTexture.dispose();
  profile.correctionTexture.dispose();
});

test('rejects a correction payload outside the regular table', () => {
  const source = readFileSync(fileURLToPath(CALAMAGROSTIS_GBR4_V4_REFERENCE_URL));
  const bytes = new Uint8Array(source);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const correctionOffset = view.getUint32(280, true);
  const correctionTexels = view.getUint32(292, true) * view.getUint32(296, true);
  for (let slot = 0; slot < correctionTexels; slot++) {
    const payload = correctionOffset + slot * 16 + 8;
    if (view.getUint32(payload, true) === 0) continue;
    view.setUint32(payload, 0xffff_ffff, true);
    assert.throws(() => parseGbr4V4Profile(bytes), /absent regular payload/);
    return;
  }
  assert.fail('reference asset contains no populated correction');
});

test('rejects a RED reference bit-flipped to GREEN without terrain/motion closure', () => {
  const source = readFileSync(fileURLToPath(CALAMAGROSTIS_GBR4_V4_REFERENCE_URL));
  const bytes = new Uint8Array(source);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  view.setUint32(312, 1, true);
  assert.throws(
    () => parseGbr4V4Profile(bytes),
    /GREEN regular payload \d+ lacks a positive terrain\/motion firstness margin/,
  );
});
