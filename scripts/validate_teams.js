/* Validate the packed teams in teams/ against our custom battle format.
 *
 * Usage:
 *   node scripts/validate_teams.js [formatId] [teamsDir]
 *   node scripts/validate_teams.js                    # defaults below
 * Defaults: formatId=gen9vgc2025regihidden, teamsDir=teams
 * Requires a built pokemon-showdown checkout (dist/); path via SHOWDOWN_DIR env
 * or ../pokemon-showdown relative to this repo.
 */

'use strict';

const fs = require('fs');
const path = require('path');

const showdownDir = process.env.SHOWDOWN_DIR ||
	path.resolve(__dirname, '..', '..', 'pokemon-showdown');

const formatId = process.argv[2] || 'gen9vgc2025regihidden';
const teamsDir = process.argv[3] || path.resolve(__dirname, '..', 'teams');

const {Dex} = require(path.join(showdownDir, 'dist', 'sim', 'dex.js'));
const {TeamValidator} = require(path.join(showdownDir, 'dist', 'sim', 'team-validator.js'));

const readFileMeta = file => {
	const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/);
	const packed = lines.filter(l => l && !l.startsWith('#')).join('').trim();
	if (!packed) throw new Error(`no packed team found in ${file}`);
	let formatTag = null;
	for (const line of lines) {
		const m = /^#\s*format\s*:\s*(.+)$/i.exec(line.trim());
		if (m) formatTag = m[1].trim().toLowerCase();
	}
	return {packed, formatTag};
};

const targetChampions = formatId.includes('champions');
const tagMatches = tag => {
	if (!tag) return true; // wildcard
	if (['champions', 'champion'].includes(tag)) return targetChampions;
	if (['standard', 'gen9', 'regular'].includes(tag)) return !targetChampions;
	return true;
};

const readPacked = file => {
	const {packed} = readFileMeta(file);
	return packed;
};

const unpack = packed => {
	const {Teams} = require(path.join(showdownDir, 'dist', 'sim', 'teams.js'));
	return Teams.unpack(packed);
};

let failures = 0;
const files = fs.readdirSync(teamsDir).filter(f => f.endsWith('.txt')).sort();
for (const file of files) {
	const full = path.join(teamsDir, file);
	const {packed, formatTag} = readFileMeta(full);
	if (!tagMatches(formatTag)) {
		console.log(`SKIP ${file} (format: ${formatTag})`);
		continue;
	}
	let problems = [];
	try {
		const team = unpack(packed);
		if (!team) {
			problems.push('could not unpack (bad packed format)');
		} else {
			problems = TeamValidator.get(formatId).validateTeam(team) || [];
		}
	} catch (e) {
		problems.push(`error: ${e.message}`);
	}
	if (problems.length) {
		failures++;
		console.log(`FAIL ${file}`);
		for (const p of problems) console.log(`  - ${p}`);
	} else {
		console.log(`OK   ${file}`);
	}
}
process.exit(failures ? 1 : 0);
