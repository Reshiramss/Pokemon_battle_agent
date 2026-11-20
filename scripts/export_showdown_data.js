#!/usr/bin/env node
/* Export Showdown's compiled data tables to JSON for the agent's static-KB.
 *
 * Usage:
 *   node export_showdown_data.js <showdown-dist-data-dir> <out-dir>
 *        [--overlay champions|championsregmb ...] [--formats]
 *        [--random-sets champions]
 *
 * Base tables are written to <out-dir>/showdown/. Each --overlay writes the
 * compiled mod's override tables (inherit: true entries) to
 * <out-dir>/showdown_<mod>/; the Python ShowdownData loader merges them over
 * the base bundle with Showdown's inheritance semantics. --formats writes
 * <out-dir>/format_profiles.json with the resolved rules flags (mod,
 * levelClause, evLimit, gameType) used by FormatProfile.
 *
 * Reads the CommonJS modules the Showdown build produces under
 * pokemon-showdown/dist/, so no TypeScript transpilation is needed.
 * Rerun whenever the pokemon-showdown checkout is updated.
 */

'use strict';

const fs = require('fs');
const path = require('path');

const positional = [];
const overlays = [];
const randomSetMods = [];
let exportFormats = false;
const cliArgs = process.argv.slice(2);
for (let i = 0; i < cliArgs.length; i++) {
	const arg = cliArgs[i];
	if (arg === '--formats') {
		exportFormats = true;
	} else if (arg === '--overlay') {
		if (cliArgs[i + 1]) overlays.push(cliArgs[++i]);
	} else if (arg.startsWith('--overlay=')) {
		overlays.push(arg.slice('--overlay='.length));
	} else if (arg === '--random-sets') {
		if (cliArgs[i + 1]) randomSetMods.push(cliArgs[++i]);
	} else if (arg.startsWith('--random-sets=')) {
		randomSetMods.push(arg.slice('--random-sets='.length));
	} else if (arg.startsWith('-')) {
		// ignore unknown flags
	} else {
		positional.push(arg);
	}
}

const [distDir, outDir] = positional;
if (!distDir || !outDir) {
	console.error('usage: node export_showdown_data.js <showdown-dist-data-dir> <out-dir> [--overlay <mod>] [--formats] [--random-sets <mod>]');
	process.exit(1);
}

// exportName -> compiled module (relative to dist/data)
const BASE_TABLES = {
	moves: ['Moves', 'moves.js'],
	pokedex: ['Pokedex', 'pokedex.js'],
	items: ['Items', 'items.js'],
	typechart: ['TypeChart', 'typechart.js'],
	aliases: ['Aliases', 'aliases.js'],
	formats: ['FormatsData', 'formats-data.js'],
	text_moves: ['MovesText', 'text/moves.js'],
	text_pokedex: ['PokedexText', 'text/pokedex.js'],
	abilities: ['Abilities', 'abilities.js'],
	// zh-cn ability files carry the official Chinese name only (no desc);
	// descriptions come from the EN text table
	text_abilities: ['AbilitiesText', 'text/abilities.js'],
	zh_pokedex: ['PokedexText', 'text/zh-cn/pokedex.js'],
	zh_moves: ['MovesText', 'text/zh-cn/moves.js'],
	zh_abilities: ['AbilitiesText', 'text/zh-cn/abilities.js'],
};

// mod override table -> compiled module (relative to dist/data/mods/<mod>/)
const MOD_TABLES = {
	moves: ['Moves', 'moves.js'],
	abilities: ['Abilities', 'abilities.js'],
	items: ['Items', 'items.js'],
	formats: ['FormatsData', 'formats-data.js'],
};

function requireTable(file, exportName) {
	const mod = require(file);
	const table = mod[exportName];
	if (!table || typeof table !== 'object') {
		throw new Error(`no export "${exportName}" in ${file}`);
	}
	return table;
}

function mergeOverride(base, extra) {
	// used only to layer championsregmb over champions: shallow per-entry
	// merge with the later mod winning. `inherit` stays true.
	const out = { ...base };
	for (const [key, value] of Object.entries(extra)) {
		const prev = out[key];
		if (prev && typeof prev === 'object' && value && typeof value === 'object') {
			out[key] = { ...prev, ...value, inherit: prev.inherit || value.inherit };
		} else {
			out[key] = value;
		}
	}
	return out;
}

function writeJson(file, value) {
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, JSON.stringify(value));
	console.log(`${path.basename(path.dirname(file))}/${path.basename(file)}: ${Object.keys(value).length} entries`);
}

// --- base bundle -------------------------------------------------------------
const baseOut = path.join(outDir, 'showdown');
const base = {};
for (const [outName, [exportName, moduleFile]] of Object.entries(BASE_TABLES)) {
	const table = requireTable(path.join(distDir, moduleFile), exportName);
	base[outName] = table;
	writeJson(path.join(baseOut, `${outName}.json`), table);
}

// --- mod overlays ------------------------------------------------------------
for (const rawMod of overlays) {
	const modId = String(rawMod || '').trim();
	if (!modId) continue;
	const chain = modId === 'championsregmb' ? ['champions', 'championsregmb'] : [modId];
	const merged = {};
	for (const link of chain) {
		const modDir = path.join(distDir, 'mods', link);
		for (const [outName, [exportName, moduleFile]] of Object.entries(MOD_TABLES)) {
			const file = path.join(modDir, moduleFile);
			if (!fs.existsSync(file)) continue;
			const table = requireTable(file, exportName);
			merged[outName] = mergeOverride(merged[outName] || {}, table);
		}
	}
	const bundleOut = path.join(outDir, `showdown_${modId}`);
	for (const [outName, table] of Object.entries(merged)) {
		writeJson(path.join(bundleOut, `${outName}.json`), table);
	}
}

// --- curated random-battle sets (cold-start priors) --------------------------
// Source: <repo>/data/random-battles/<mod>/{sets,doubles-sets}.json. These are
// plain JSON (not compiled modules), so read them from the source checkout.
for (const modId of randomSetMods) {
	const mod = String(modId || '').trim();
	if (!mod) continue;
	const srcDir = path.join(distDir, '..', '..', 'data', 'random-battles', mod);
	const outDirForMod = path.join(outDir, 'seed', mod);
	for (const file of ['sets.json', 'doubles-sets.json']) {
		const src = path.join(srcDir, file);
		if (!fs.existsSync(src)) continue;
		try {
			const table = JSON.parse(fs.readFileSync(src, 'utf8'));
			writeJson(path.join(outDirForMod, file), table);
		} catch (err) {
			console.warn(`[random-sets] ${src}: ${err.message}`);
		}
	}
}

// --- resolved format profiles ------------------------------------------------
if (exportFormats) {
	const { Dex } = require(path.join(distDir, '..', 'sim', 'dex'));
	const profiles = {};
	for (const format of Dex.formats.all()) {
		let ruleTable = null;
		try {
			ruleTable = Dex.formats.getRuleTable(format);
		} catch {
			ruleTable = null;
		}
		profiles[format.id] = {
			name: format.name,
			mod: format.mod,
			gameType: format.gameType,
			team: format.team || null,
			levelClause: ruleTable ? ruleTable.has('levelclausemod') : undefined,
			evLimit: ruleTable ? ruleTable.evLimit : undefined,
			ruleset: format.ruleset || [],
		};
	}
	writeJson(path.join(outDir, 'format_profiles.json'), profiles);
}