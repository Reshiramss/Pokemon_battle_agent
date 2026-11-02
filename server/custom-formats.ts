/**
 * Custom format for the PokeAgent framework.
 *
 * Install (one-time):
 *   cp server/custom-formats.ts /path/to/pokemon-showdown/config/custom-formats.ts
 *   cd /path/to/pokemon-showdown && node build
 *   node pokemon-showdown start --no-security --skip-build 8964
 *
 * Same rules as "[Gen 9] VGC 2025 Reg I" minus Open Team Sheets: doubles,
 * bring 6 / pick 4 with lead selection, level 50. At team preview each side
 * only sees the opponent's six species — moves, items and abilities are
 * hidden until revealed in battle.
 */
exports.Formats = [
	{
		name: "[Gen 9] VGC 2025 Reg I (Hidden)",
		mod: 'gen9',
		gameType: 'doubles',
		searchShow: false,
		bestOfDefault: true,
		ruleset: ['Flat Rules', '!! Adjust Level = 50', 'Min Source Gen = 9', 'VGC Timer', 'Limit Two Restricted'],
		restricted: ['Restricted Legendary'],
	},
];
