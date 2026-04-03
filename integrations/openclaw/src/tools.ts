/**
 * Tools for hybrid memory mode.
 *
 * memory_search: queries MambaMemory cognitive state (L1+L2+L3)
 * memory_ingest: stores important facts in MambaMemory with pre-compression
 */

import { Type } from "@sinclair/typebox";
import type { MambaBridge } from "./mcp-bridge.js";

export function createMemorySearchTool(bridge: MambaBridge) {
	return {
		name: "memory_search",
		description:
			"Search your memory for past context. " +
			"Queries the cognitive memory system (recent conversations, active knowledge slots, " +
			"and long-term persistent storage). Results are ranked by semantic relevance, " +
			"activation level, and recency. " +
			"Use this before answering questions that might need past context.",
		parameters: Type.Object({
			query: Type.String({ description: "What to search for" }),
			limit: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
		}),
		async execute(_id: string, params: { query: string; limit?: number }) {
			try {
				const result = await bridge.recall({
					query: params.query,
					limit: params.limit || 5,
				});

				if (result.memories.length === 0) {
					return {
						content: [{
							type: "text" as const,
							text: "No relevant memories found in cognitive state. " +
								"Try checking MEMORY.md or memory/*.md files directly.",
						}],
					};
				}

				const lines = result.memories.map(
					(m) => `[${m.layer}] (score=${m.score.toFixed(2)}) ${m.content}`,
				);
				return {
					content: [{ type: "text" as const, text: lines.join("\n") }],
				};
			} catch (e) {
				return {
					content: [{
						type: "text" as const,
						text: `Cognitive memory search failed: ${e}. ` +
							"Falling back to file-based search — check MEMORY.md and memory/*.md.",
					}],
				};
			}
		},
	};
}

export function createMemoryIngestTool(bridge: MambaBridge) {
	return {
		name: "memory_ingest",
		description:
			"Store important information in cognitive memory. " +
			"The gate automatically filters noise — greetings, filler, and low-value content " +
			"are discarded. Provide a pre-compressed summary and entity tags for best results. " +
			"Use this for: decisions, configurations, corrections, preferences, action items.",
		parameters: Type.Object({
			content: Type.String({ description: "Raw content to remember" }),
			summary: Type.Optional(
				Type.String({
					description:
						"Your compressed summary (recommended). " +
						"Goes directly to L2 cognitive state, bypassing internal compression.",
				}),
			),
			entities: Type.Optional(
				Type.Array(Type.String(), {
					description: "Entity tags (people, projects, tools, technologies)",
				}),
			),
			force: Type.Optional(
				Type.Boolean({
					description: "Force store, bypass the selective gate",
				}),
			),
		}),
		async execute(
			_id: string,
			params: {
				content: string;
				summary?: string;
				entities?: string[];
				force?: boolean;
			},
		) {
			try {
				const result = await bridge.ingest({
					content: params.content,
					summary: params.summary,
					entities: params.entities,
					force: params.force,
				});

				const status = result.stored ? "Stored" : "Filtered";
				return {
					content: [{
						type: "text" as const,
						text: `${status} in ${result.layer}: ${result.reason}`,
					}],
				};
			} catch (e) {
				return {
					content: [{
						type: "text" as const,
						text: `Memory ingest failed: ${e}`,
					}],
				};
			}
		},
	};
}
