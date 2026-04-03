/**
 * Memory flush plan — hybrid mode.
 *
 * File-based flush works exactly like memory-core: when the conversation
 * gets too long, the agent compresses and writes to memory/*.md files.
 *
 * The key addition: the flush prompt also instructs the agent to
 * memory_ingest the most important facts into MambaMemory cognitive state.
 * This way both systems stay in sync without double-storing everything.
 */

export function buildMemoryFlushPlan() {
	return {
		softThresholdTokens: 4000,
		forceFlushTranscriptBytes: 2 * 1024 * 1024,
		reserveTokensFloor: 1000,
		prompt:
			"The conversation is getting long. Do TWO things:\n\n" +
			"1. **File flush**: Summarize the key information from this conversation " +
			"and append it to the memory file (same as usual).\n\n" +
			"2. **Cognitive ingest**: For the most important decisions, facts, and " +
			"action items, also call `memory_ingest` with a compressed summary and " +
			"entity tags. This stores them in the fast cognitive layer for quick recall.\n\n" +
			"Focus memory_ingest on: decisions made, configurations discussed, " +
			"corrections to previous knowledge, and explicit 'remember this' requests.\n" +
			"Skip: greetings, chitchat, long explanations (those go in the file only).",
		systemPrompt:
			"You are performing a memory flush. Write key facts to the memory file " +
			"AND call memory_ingest for the most important cognitive state changes.",
		relativePath: "memory/mamba-flush.md",
	};
}
