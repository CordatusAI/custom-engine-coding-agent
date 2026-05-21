# Installing Custom Engine Coding Agent for OpenCode

## Prerequisites

- [OpenCode.ai](https://opencode.ai) installed

## Installation

Add custom-engine-coding-agent to the `plugin` array in your `opencode.json` (global or project-level):

```json
{
  "plugin": ["custom-engine-coding-agent@git+https://github.com/<org>/custom-engine-coding-agent.git"]
}
```

Restart OpenCode. The plugin installs through OpenCode's plugin manager and registers all skills.

Verify by asking: "What Custom Engine skills do you have?"

OpenCode uses its own plugin install. If you also use Claude Code, Cursor, Codex, or another harness, install Custom Engine Coding Agent separately for each one.

## Usage

Use OpenCode's native `skill` tool:

```
use skill tool to list skills
use skill tool to load custom-engine-dev
```

## Updating

OpenCode installs Custom Engine Coding Agent through a git-backed package spec. Some OpenCode and Bun versions pin that resolved git dependency in a lockfile or cache, so a restart may not pick up the newest commit. If updates do not appear, clear OpenCode's package cache or reinstall the plugin.

To pin a specific version:

```json
{
  "plugin": ["custom-engine-coding-agent@git+https://github.com/<org>/custom-engine-coding-agent.git#v1.0.0"]
}
```

## Troubleshooting

### Plugin not loading

1. Check logs: `opencode run --print-logs "hello" 2>&1 | grep -i custom-engine`
2. Verify the plugin line in your `opencode.json`
3. Make sure you're running a recent version of OpenCode

### Windows install issues

Some Windows OpenCode builds have upstream installer issues with git-backed plugin specs. If OpenCode cannot install the plugin, try installing with system npm and pointing OpenCode at the local package:

```powershell
npm install custom-engine-coding-agent@git+https://github.com/<org>/custom-engine-coding-agent.git --prefix "$HOME\.config\opencode"
```

Then use the installed package path in `opencode.json`:

```json
{
  "plugin": ["~/.config/opencode/node_modules/custom-engine-coding-agent"]
}
```

### Skills not found

1. Use `skill` tool to list what's discovered
2. Check that the plugin is loading (see above)

### Tool mapping

When skills reference Claude Code tools:
- `TodoWrite` → `todowrite`
- `Task` with subagents → `@mention` syntax
- `Skill` tool → OpenCode's native `skill` tool
- File operations → your native tools
