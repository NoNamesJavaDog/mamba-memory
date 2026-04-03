/**
 * MCP bridge to the MambaMemory Python server.
 *
 * Spawns `mamba-memory serve --mcp` as a child process and communicates
 * via stdio using the MCP protocol. All tool calls are proxied through
 * this bridge.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { once } from "node:events";

export interface MambaConfig {
	dbPath?: string;
	embeddingProvider?: string;
	pythonPath?: string;
}

export interface IngestParams {
	content: string;
	tags?: string[];
	force?: boolean;
	summary?: string;
	entities?: string[];
}

export interface RecallParams {
	query: string;
	limit?: number;
	layers?: string[];
	min_score?: number;
}

export interface IngestResult {
	stored: boolean;
	layer: string;
	slot_id?: number;
	reason: string;
}

export interface RecallItem {
	content: string;
	topic: string;
	score: number;
	layer: string;
	source_id?: string;
}

export interface RecallResult {
	count: number;
	total_tokens: number;
	memories: RecallItem[];
}

export interface MemoryStatus {
	l1_window_turns: number;
	l1_compressed_segments: number;
	l2_active_slots: number;
	l2_total_slots: number;
	l2_step_count: number;
	l3_total_records: number;
	l3_archived_records: number;
	l3_entity_count: number;
}

/**
 * MCP bridge client that manages the MambaMemory subprocess.
 *
 * Uses a simple JSON-RPC-over-stdio protocol: each request/response
 * is a single line of JSON followed by a newline.
 */
export class MambaBridge {
	private proc: ChildProcess | null = null;
	private config: MambaConfig;
	private requestId = 0;
	private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
	private buffer = "";
	private initialized = false;

	constructor(config: MambaConfig = {}) {
		this.config = config;
	}

	async start(): Promise<void> {
		if (this.proc) return;

		const python = this.config.pythonPath || "python3";
		const args = ["-m", "mamba_memory.cli", "serve", "--mcp"];
		if (this.config.dbPath) {
			args.push("--db", this.config.dbPath);
		}

		this.proc = spawn(python, args, {
			stdio: ["pipe", "pipe", "pipe"],
			env: {
				...process.env,
				PYTHONUNBUFFERED: "1",
			},
		});

		this.proc.stdout!.on("data", (data: Buffer) => {
			this.buffer += data.toString();
			this.processBuffer();
		});

		this.proc.stderr!.on("data", (data: Buffer) => {
			// Log MambaMemory stderr for debugging
			const msg = data.toString().trim();
			if (msg) {
				console.error(`[mamba-memory] ${msg}`);
			}
		});

		this.proc.on("exit", (code) => {
			console.error(`[mamba-memory] process exited with code ${code}`);
			this.proc = null;
			// Reject all pending requests
			for (const [, { reject }] of this.pending) {
				reject(new Error(`MambaMemory process exited (code ${code})`));
			}
			this.pending.clear();
		});

		// Send MCP initialize
		await this.sendRpc("initialize", {
			protocolVersion: "2024-11-05",
			capabilities: {},
			clientInfo: { name: "openclaw-memory-mamba", version: "0.1.0" },
		});

		// Send initialized notification
		this.sendNotification("notifications/initialized", {});
		this.initialized = true;
	}

	async stop(): Promise<void> {
		if (this.proc) {
			this.proc.kill("SIGTERM");
			this.proc = null;
		}
		this.initialized = false;
	}

	async ingest(params: IngestParams): Promise<IngestResult> {
		return (await this.callTool("memory_ingest", params)) as IngestResult;
	}

	async recall(params: RecallParams): Promise<RecallResult> {
		return (await this.callTool("memory_recall", params)) as RecallResult;
	}

	async forget(query: string): Promise<{ forgotten: number }> {
		return (await this.callTool("memory_forget", { query })) as { forgotten: number };
	}

	async status(detail?: string): Promise<MemoryStatus> {
		return (await this.callTool("memory_status", { detail: detail || "summary" })) as MemoryStatus;
	}

	async compact(layer?: string): Promise<Record<string, number>> {
		return (await this.callTool("memory_compact", { layer: layer || "all" })) as Record<string, number>;
	}

	private async callTool(name: string, arguments_: Record<string, unknown>): Promise<unknown> {
		const result = (await this.sendRpc("tools/call", { name, arguments: arguments_ })) as {
			content: Array<{ type: string; text: string }>;
		};
		if (result.content?.[0]?.text) {
			return JSON.parse(result.content[0].text);
		}
		return result;
	}

	private async sendRpc(method: string, params: unknown): Promise<unknown> {
		if (!this.proc?.stdin) {
			throw new Error("MambaMemory process not running");
		}

		const id = ++this.requestId;
		const request = JSON.stringify({ jsonrpc: "2.0", id, method, params });

		return new Promise((resolve, reject) => {
			this.pending.set(id, { resolve, reject });
			this.proc!.stdin!.write(request + "\n");
		});
	}

	private sendNotification(method: string, params: unknown): void {
		if (!this.proc?.stdin) return;
		const notification = JSON.stringify({ jsonrpc: "2.0", method, params });
		this.proc.stdin.write(notification + "\n");
	}

	private processBuffer(): void {
		const lines = this.buffer.split("\n");
		this.buffer = lines.pop() || "";

		for (const line of lines) {
			const trimmed = line.trim();
			if (!trimmed) continue;

			try {
				const msg = JSON.parse(trimmed);
				if (msg.id !== undefined && this.pending.has(msg.id)) {
					const { resolve, reject } = this.pending.get(msg.id)!;
					this.pending.delete(msg.id);
					if (msg.error) {
						reject(new Error(msg.error.message || JSON.stringify(msg.error)));
					} else {
						resolve(msg.result);
					}
				}
			} catch {
				// Not valid JSON, skip
			}
		}
	}
}
