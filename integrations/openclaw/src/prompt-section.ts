/**
 * Builds the system prompt section for hybrid memory mode.
 *
 * Guides the agent to:
 * 1. Use memory_search for recall (searches both MambaMemory + files)
 * 2. Use memory_ingest to store important cognitive state
 * 3. Let flush handle file-based archiving automatically
 */

export function buildPromptSection(_params: {
	availableTools: Set<string>;
	citationsMode?: string;
}): string[] {
	const lines: string[] = [];

	lines.push("## Memory (Hybrid: Files + Cognitive State)");
	lines.push("");
	lines.push("You have two memory systems working together:");
	lines.push("- **File memory**: MEMORY.md + memory/*.md files (long-term archive, auto-flushed)");
	lines.push("- **Cognitive memory**: MambaMemory L2 slots (active knowledge with decay)");
	lines.push("");

	if (_params.availableTools.has("memory_search")) {
		lines.push("### Recalling memories");
		lines.push("Before answering questions about prior work, use `memory_search`.");
		lines.push("It searches both cognitive state (fast, ranked by relevance + recency)");
		lines.push("and file-based memory (comprehensive, full-text).");
		lines.push("");
	}

	if (_params.availableTools.has("memory_ingest")) {
		lines.push("### Storing important information");
		lines.push("When the user shares important information, use `memory_ingest` to store it");
		lines.push("in cognitive memory. You ARE the compressor — summarize before ingesting.");
		lines.push("");
		lines.push("**What to ingest** (high-value cognitive state):");
		lines.push("- Decisions: technology choices, architecture decisions");
		lines.push("- Facts with data: IPs, ports, configs, connection strings");
		lines.push("- Corrections: fixing previous wrong information");
		lines.push("- Preferences: user's coding style, tool preferences");
		lines.push("- Action items: TODOs, deadlines, next steps");
		lines.push("");
		lines.push("**What NOT to ingest** (handled by file flush, or discard):");
		lines.push("- Greetings, acknowledgments, small talk");
		lines.push("- Long code blocks (use files instead)");
		lines.push("- Information already in MEMORY.md");
		lines.push("");
		lines.push("Example:");
		lines.push("```");
		lines.push('memory_ingest({');
		lines.push('  content: "user discussed database migration at length",');
		lines.push('  summary: "Decided PostgreSQL 16, port 5432, pool max 50",');
		lines.push('  entities: ["PostgreSQL"],');
		lines.push("})");
		lines.push("```");
		lines.push("");
		lines.push("The gate automatically filters noise — if you're unsure, ingest anyway.");
		lines.push("");
	}

	return lines;
}
