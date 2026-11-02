#!/usr/bin/env node
/* Export Showdown's compiled data tables to JSON for the agent's static-KB.
 *
 * Usage: node export_showdown_data.js <showdown-dist-data-dir> <output-dir>
 * Reads the CommonJS modules that the Showdown build produces under
 * pokemon-showdown/dist/data/, so no TypeScript transpilation is needed.
 * Rerun whenever the pokemon-showdown checkout is updated.
 */

'use strict';

const fs = require('fs');
const path = require('path');

const [distDir, outDir] = process.argv.slice(2);
if (!distDir || !outDir) {
	console.error('usage: node export_showdown_data.js <showdown-dist-data-dir> <output-dir>');
	process.exit(1);
}

// exportName -> compiled module (relative to dist/data)
const TABLES = {
	moves: ['Moves', 'moves.js'],
	pokedex: ['Pokedex', 'pokedex.js'],
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

fs.mkdirSync(outDir, { recursive: true });
for (const [outName, [exportName, moduleFile]] of Object.entries(TABLES)) {
	const mod = require(path.join(distDir, moduleFile));
	const table = mod[exportName];
	if (!table || typeof table !== 'object') {
		console.error(`FAIL ${moduleFile}: no export "${exportName}"`);
		process.exit(1);
	}
	const outFile = path.join(outDir, `${outName}.json`);
	fs.writeFileSync(outFile, JSON.stringify(table));
	console.log(`${outName}: ${Object.keys(table).length} entries -> ${outFile}`);
}
