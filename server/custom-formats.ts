/**
 * Custom format for the PokeAgent framework.
 *
 * Install (one-time):
 *   cp server/custom-formats.ts /path/to/pokemon-showdown/config/custom-formats.ts
 *   cd /path/to/pokemon-showdown && node build
 *   node pokemon-showdown start --no-security --skip-build 8964
 *
 * Same rules as "[Gen 9] VGC 2025 Reg I": doubles, bring 6 / pick 4 with
 * lead selection, level 50. At team preview each side only sees the
 * opponent's six species — moves, items and abilities stay hidden until
 * revealed in battle.
 *
 * 'Open Team Sheets' is kept ON DELIBERATELY. In current Showdown it is
 * strictly opt-in (both sides must run /acceptopenteamsheets during team
 * preview); poke-env clients send /rejectopenteamsheets for any "vgc"
 * format, which both guarantees sheets stay hidden for our battles and
 * avoids the server error the same command raises in a format without
 * the rule.
 *
 * The one deliberate deviation from official Reg I: the VGC Timer
 * (55s per turn, 90s preview, 7-minute bank) is replaced by an AI-friendly
 * one. An agentic opponent needs multiple LLM calls per turn and a human
 * opponent needs thinking time — under the official timer both get
 * force-moved or forfeited. Team legality is unaffected (Flat Rules etc.
 * stay intact).
 */
exports.Formats = [
	{
		name: "[Gen 9] VGC 2025 Reg I (Hidden)",
		mod: 'gen9',
		gameType: 'doubles',
		searchShow: false,
		bestOfDefault: true,
		ruleset: [
			'Flat Rules', '!! Adjust Level = 50', 'Min Source Gen = 9',
			'Timer Starting = 1200', 'Timer Grace = 300', 'Timer Add Per Turn = 30',
			'Timer Max Per Turn = 1200', 'Timer Max First Turn = 1200',
			'Timeout Auto Choose', 'DC Timer Bank',
			'Open Team Sheets', 'Limit Two Restricted',
		],
		restricted: ['Restricted Legendary'],
	},
];
