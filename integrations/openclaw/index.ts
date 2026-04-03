/**
 * OpenClaw plugin: MambaMemory — Hybrid memory with cognitive state layer.
 *
 * HYBRID MODE: Doesn't replace memory-core, enhances it.
 *
 * Architecture:
 *   ┌─────────────────────────────────────────────────────┐
 *   │  memory-core (file-based)                           │
 *   │  MEMORY.md + memory/*.md → file search + flush      │
 *   │  Handles: long-term documentation, bootstrap files  │
 *   └─────────────┬───────────────────────────────────────┘
 *                  │  ← memory-mamba enhances this layer
 *   ┌─────────────┴───────────────────────────────────────┐
 *   │  MambaMemory (cognitive state)                      │
 *   │  L2 slots + gate + decay → semantic recall          │
 *   │  Handles: decisions, preferences, corrections,      │
 *   │           facts, active knowledge that evolves       │
 *   └─────────────────────────────────────────────────────┘
 *
 * How they work together:
 *   WRITE: Agent conversation → memory-core flushes to files (unchanged)
 *          → MambaMemory also ingests key facts via memory_ingest tool
 *   READ:  Agent queries → memory_search hits BOTH systems
 *          → MambaMemory results (fast, ranked by activation + decay)
 *          → memory-core results (comprehensive, file-based)
 *          → Merged, deduplicated, returned to agent
 *
 * Setup:
 *   1. pip install mamba-memory[google]
 *   2. openclaw config set plugins.slots.memory memory-mamba
 *   3. Set GOOGLE_API_KEY in environment
 *
 * The flush plan delegates to memory-core's file-based flush (unchanged).
 * MambaMemory adds a SECOND path: the agent uses memory_ingest to store
 * high-value cognitive state in L2, guided by the prompt section.
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { buildMemoryFlushPlan } from "./src/flush-plan.js";
import { MambaBridge, type MambaConfig } from "./src/mcp-bridge.js";
import { buildPromptSection } from "./src/prompt-section.js";
import { createMemoryIngestTool, createMemorySearchTool } from "./src/tools.js";

export default definePluginEntry({
	id: "memory-mamba",
	name: "Memory (MambaMemory Hybrid)",
	description:
		"Hybrid memory: memory-core file search + MambaMemory cognitive state " +
		"with selective gating, Ebbinghaus decay, and self-evolving neural gate",
	kind: "memory",

	register(api) {
		const pluginConfig = api.pluginConfig as MambaConfig | undefined;
		const bridge = new MambaBridge({
			dbPath: pluginConfig?.dbPath,
			embeddingProvider: pluginConfig?.embeddingProvider,
			pythonPath: pluginConfig?.pythonPath,
		});

		// -- Service: MambaMemory subprocess ----------------------------

		api.registerService({
			id: "mamba-memory-bridge",
			async start() {
				await bridge.start();
			},
			async stop() {
				await bridge.stop();
			},
		});

		// -- Prompt section: guides agent to use BOTH systems -----------

		api.registerMemoryPromptSection(buildPromptSection);

		// -- Flush plan: delegates to file-based flush (same as core) ---
		// memory-core's flush writes to memory/*.md files.
		// MambaMemory adds a separate cognitive layer via memory_ingest.

		api.registerMemoryFlushPlan(buildMemoryFlushPlan);

		// -- Tools: search + ingest ------------------------------------

		// memory_search: hybrid search across MambaMemory + fallback
		api.registerTool(() => createMemorySearchTool(bridge), {
			names: ["memory_search"],
		});

		// memory_ingest: store cognitive state in MambaMemory L2
		// This is the key addition over memory-core — the agent can
		// proactively store important facts with pre-compressed summaries.
		api.registerTool(() => createMemoryIngestTool(bridge), {
			names: ["memory_ingest"],
		});

		// -- CLI -------------------------------------------------------

		api.registerCli(
			({ program }) => {
				const cmd = program.command("mamba-memory").description("MambaMemory cognitive memory engine");

				cmd.command("status").description("Show cognitive memory status").action(async () => {
					try {
						const status = await bridge.status("slots");
						console.log(JSON.stringify(status, null, 2));
					} catch (e) {
						console.error("Failed to get status:", e);
					}
				});

				cmd.command("compact").description("Trigger memory compaction").action(async () => {
					try {
						const result = await bridge.compact("all");
						console.log(JSON.stringify(result, null, 2));
					} catch (e) {
						console.error("Failed to compact:", e);
					}
				});
			},
			{
				descriptors: [
					{
						name: "mamba-memory",
						description: "MambaMemory cognitive memory engine",
						hasSubcommands: true,
					},
				],
			},
		);
	},
});
