# Team library

Packed teams for team-preview formats (default: `gen9vgc2025regihidden`, our
custom VGC 2025 Reg I doubles format without Open Team Sheets). Both players in
a VGC battle must bring a team — challenges without one are rejected.

## Built-in teams

| File | Team | Archetype |
|---|---|---|
| `shadowbolt.txt` | Shadow Bolt | hyper offense (Calyrex-Shadow + Miraidon) |
| `crowned-balance.txt` | Crowned Balance | balance (Zacian/Zamazenta crowned core) |
| `rain-rush.txt` | Rain Rush | rain offense (Kyogre + Swift Swim) |
| `slow-sun.txt` | Slow Sun | Trick Room (Calyrex-Ice + Torkoal) |

Each file is one Showdown **packed team** on the non-comment lines, with `#`
comment headers (`# name:`, `# archetype:`, `# description:`) parsed by
`pokeagent/teams.py`. Validate against the format before running:

```sh
node scripts/validate_teams.js                    # all files
node scripts/validate_teams.js gen9vgc2025regihidden
```

## Using a team

```sh
python scripts/run_battle.py --team "Shadow Bolt"          # opponent auto-picks a different team
python scripts/run_battle.py --team shadowbolt --opponent-team "Rain Rush"
```

`--team` accepts either a **library name** (matched case-insensitively against
the `# name:` header, or the file stem) or a **raw packed team string**
(anything containing `|`). The same values work as the `team` key in
`config.toml`.

## Importing your own teams

Drop additional `.txt` files in the same layout (headers + one packed team)
into a directory and point `teams_dir` at it:

```toml
[agent]
team = "my-team"          # name from your file's `# name:` header
teams_dir = "/path/to/my/teams"
```

Files in your import directory shadow built-ins with the same file name. To
build a packed string from a Showdown teambuilder export, paste the team into
https://play.pokemonshowdown.com/teambuilder and copy the "packed format" — or
`join('|', ...)` each set yourself; the per-set field order is
`name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|misc`,
sets joined by `]`, 6 sets max.
