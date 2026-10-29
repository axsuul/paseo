import { OpenCodeRuntimeClient } from "./opencode/runtime-client.js";
import { mkdir, mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";

import { createOpencodeClient } from "@opencode-ai/sdk/v2/client";
import { expect, test } from "vitest";

import { execCommand } from "../../../utils/spawn.js";
import { createTestLogger } from "../../../test-utils/test-logger.js";
import {
  getRealProviderConfig,
  getRealProviderRuntimeSettings,
} from "../../daemon-e2e/real-provider-test-config.js";
import type { PaseoToolCatalog } from "../tools/types.js";
import { AgentManager } from "../agent-manager.js";
import { AgentStorage } from "../agent-storage.js";
import { ProviderSnapshotManager } from "../provider-snapshot-manager.js";
import { createPaseoAgentToolManifest, createPaseoToolCatalog } from "../tools/paseo-tools.js";
import { serializePaseoToolInputParameters } from "../tools/paseo-tool-serialization.js";
import { OpenCodeAgentClient } from "./opencode-agent.js";
import { OpenCodeV2AgentClient } from "./opencode/v2/agent.js";
import { OpenCodeBridge } from "./opencode/bridge.js";
import { OpenCodeServerManager } from "./opencode/server-manager.js";
import { createDaemonTestContext, type DaemonTestContext } from "../../test-utils/index.js";
import {
  drainPersistedTimeline,
  readAssistantText,
  requireSessionId,
} from "./opencode/test-utils/v2-local-e2e-helpers.js";

test("real OpenCode server advertises agent-scoped notification descriptions", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "paseo-opencode-manifest-real-"));
  const observed = path.join(root, "observed.json");
  const plugin = path.join(root, "observer.js");
  let modelRequests = 0;
  let releaseModel: (() => void) | undefined;
  let toolAttempt = false;
  const modelServer = createServer((request, response) => {
    modelRequests++;
    request.resume();
    if (toolAttempt) {
      releaseModel = () => {
        response.writeHead(200, { "Content-Type": "text/event-stream" });
        const delta = {
          role: "assistant",
          tool_calls: [
            {
              index: 0,
              id: "call_canceled",
              type: "function",
              function: {
                name: "bash",
                arguments: JSON.stringify({
                  command: `touch ${path.join(root, "must-not-exist")}`,
                  description: "Cancellation probe",
                }),
              },
            },
          ],
        };
        response.write(
          `data: ${JSON.stringify({ id: "test", object: "chat.completion.chunk", created: 1, model: "test", choices: [{ index: 0, delta, finish_reason: null }] })}\n\n`,
        );
        response.write(
          `data: ${JSON.stringify({ id: "test", object: "chat.completion.chunk", created: 1, model: "test", choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
        );
        response.end("data: [DONE]\n\n");
      };
      return;
    }
    response.writeHead(500, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ error: { message: "The canceled command reached the model" } }));
  });
  await new Promise<void>((resolve) => modelServer.listen(0, "127.0.0.1", resolve));
  const modelAddress = modelServer.address();
  if (!modelAddress || typeof modelAddress === "string")
    throw new Error("Model server has no address");
  await writeFile(
    plugin,
    `import { writeFile, readFile } from "node:fs/promises";
export default async () => {
  let command;
  return {
    "command.execute.before": async (input) => { command = input; },
    "chat.message": async (input, output) => {
      const delivery = output[Symbol.for("paseo.commandDelivery")];
      await writeFile(${JSON.stringify(observed)}, JSON.stringify({command, input, output, delivery}));
      if (command.arguments === "pause after claim") {
        while (await readFile(${JSON.stringify(path.join(root, "release"))}, "utf8").catch(() => "") !== "yes") await new Promise(resolve => setTimeout(resolve, 10));
        return;
      }
      if (command.arguments === "pause model") return;
      throw new Error("PASEO_COMMAND_OBSERVED");
    },
  };
};`,
  );
  const logger = createTestLogger();
  const providerSnapshotManager = new ProviderSnapshotManager({ logger });
  const dependencies = {
    agentManager: new AgentManager({ logger }),
    agentStorage: new AgentStorage(path.join(root, "agents"), logger),
    providerSnapshotManager,
    logger,
  };
  const bridge = new OpenCodeBridge({ paseoHome: root, logger });
  bridge.setManifestCatalog(createPaseoAgentToolManifest(dependencies));
  const manager = new OpenCodeServerManager({
    logger,
    resolveHomeDir: () => root,
    runtimeSettings: {
      env: {
        XDG_CONFIG_HOME: path.join(root, "config"),
        XDG_DATA_HOME: path.join(root, "data"),
        XDG_CACHE_HOME: path.join(root, "cache"),
        OPENCODE_DISABLE_AUTO_UPDATE: "1",
        OPENCODE_CONFIG_CONTENT: JSON.stringify({
          plugin: [pathToFileURL(plugin).href],
          command: { auto: { template: "$ARGUMENTS" } },
          provider: {
            local: {
              npm: "@ai-sdk/openai-compatible",
              name: "Local",
              options: { baseURL: `http://127.0.0.1:${modelAddress.port}/v1`, apiKey: "test" },
              models: { test: { name: "test", limit: { context: 10000, output: 1000 } } },
            },
          },
        }),
      },
    },
    decorateServerEnv: (env) => bridge.decorateServerEnv(env),
  });
  let inspection: Awaited<ReturnType<OpenCodeServerManager["acquireCurrent"]>> | undefined;
  let daemon: DaemonTestContext | undefined;
  try {
    await bridge.start();
    inspection = await manager.acquireCurrent();
    const sdk = createOpencodeClient({ baseUrl: inspection.server.url, directory: root });
    const response = await sdk.tool.list({ directory: root, provider: "openai", model: "gpt-5.4" });
    expect(response.error).toBeUndefined();
    const agentCatalog = createPaseoToolCatalog({ ...dependencies, callerAgentId: "caller" });
    for (const name of ["create_agent", "send_agent_prompt"]) {
      const actual = response.data!.find((tool) => tool.id === `paseo_${name}`)!;
      expect(actual).toBeDefined();
      const expected = serializePaseoToolInputParameters(agentCatalog.getTool(name)!);
      expect(actual.parameters).toMatchObject({
        properties: {
          notifyOnFinish: {
            description: (expected.properties as Record<string, { description: string }>)
              .notifyOnFinish.description,
          },
        },
      });
    }
    daemon = await createDaemonTestContext({
      agentClients: {
        opencode: new OpenCodeAgentClient(logger, undefined, { serverManager: manager, bridge }),
      },
    });
    const agent = await daemon.client.createAgent({
      provider: "opencode",
      cwd: root,
      model: "local/test",
    });
    const arrival = {
      executionId: "execution-one",
      instructions: "Trusted Hub instructions",
      history: "Untrusted Discord history",
    };
    await daemon.client.sendMessage(agent.id, "only the request", {
      command: { name: "auto", arguments: "only the request", context: arrival },
    });
    await expect
      .poll(async () => readFile(observed, "utf8").catch(() => ""), { timeout: 30_000 })
      .not.toBe("");
    await daemon.client.waitForFinish(agent.id, 30_000);
    const result = JSON.parse(await readFile(observed, "utf8"));
    expect(result.command).toMatchObject({ command: "auto", arguments: "only the request" });
    expect(result.delivery.context).toEqual(arrival);
    expect(result.output.parts[0].metadata?.paseoArrival).toBeUndefined();
    expect(JSON.stringify(result.output)).not.toContain("paseo.commandDelivery");
    const continuationMessageID = result.delivery.continuationMessageID;
    expect(continuationMessageID).toMatch(/^msg_paseo_route_/);
    const sessionID = result.output.message.sessionID;
    const foreign = await sdk.session.create({ directory: root });
    const foreignRoute = await sdk.session.command({
      sessionID: foreign.data!.id,
      directory: root,
      messageID: continuationMessageID,
      command: "auto",
      arguments: "must not run",
      model: "local/test",
    });
    expect(foreignRoute.error).toBeDefined();
    expect(JSON.parse(await readFile(observed, "utf8")).output.message.id).toBe(
      result.output.message.id,
    );
    const validRoute = await sdk.session.command({
      sessionID,
      directory: root,
      messageID: continuationMessageID,
      command: "auto",
      arguments: "routed request",
      model: "local/test",
    });
    expect(validRoute.error).toBeDefined();
    const routed = JSON.parse(await readFile(observed, "utf8"));
    expect(routed.output.message.id).toBe(continuationMessageID);
    expect(routed.delivery.context).toEqual(arrival);
    const replay = await sdk.session.command({
      sessionID,
      directory: root,
      messageID: continuationMessageID,
      command: "auto",
      arguments: "replay",
      model: "local/test",
    });
    expect(replay.error).toBeDefined();
    expect(JSON.parse(await readFile(observed, "utf8")).command.arguments).toBe("routed request");
    await daemon.client.sendMessage(agent.id, "second request", {
      command: {
        name: "auto",
        arguments: "second request",
        context: { ...arrival, executionId: "execution-two" },
      },
    });
    await daemon.client.waitForFinish(agent.id, 30_000);
    const second = JSON.parse(await readFile(observed, "utf8"));
    const canceledMessageID = second.delivery.continuationMessageID;
    await daemon.client.cancelAgent(agent.id);
    const canceledRoute = await sdk.session.command({
      sessionID,
      directory: root,
      messageID: canceledMessageID,
      command: "auto",
      arguments: "must not run",
      model: "local/test",
    });
    expect(canceledRoute.error).toBeDefined();
    expect(JSON.parse(await readFile(observed, "utf8")).output.message.id).toBe(
      second.output.message.id,
    );
    await daemon.client.sendMessage(agent.id, "third request", {
      command: { name: "auto", arguments: "third request", context: arrival },
    });
    await daemon.client.waitForFinish(agent.id, 30_000);
    const third = JSON.parse(await readFile(observed, "utf8"));
    const claimedMessageID = third.delivery.continuationMessageID;
    const afterClaim = sdk.session.command({
      sessionID,
      directory: root,
      messageID: claimedMessageID,
      command: "auto",
      arguments: "pause after claim",
      model: "local/test",
    });
    await expect
      .poll(async () => JSON.parse(await readFile(observed, "utf8")).output.message.id)
      .toBe(claimedMessageID);
    await daemon.client.cancelAgent(agent.id);
    await writeFile(path.join(root, "release"), "yes");
    const stopped = await afterClaim;
    expect(stopped.error ?? stopped.data?.info.error).toBeDefined();
    expect(JSON.stringify(stopped)).toContain("Command arrival is no longer active");
    expect(modelRequests).toBe(0);
    await daemon.client.sendMessage(agent.id, "tool request", {
      command: { name: "auto", arguments: "tool request", context: arrival },
    });
    await daemon.client.waitForFinish(agent.id, 30_000);
    const toolInitial = JSON.parse(await readFile(observed, "utf8"));
    toolAttempt = true;
    const toolCommand = sdk.session.command({
      sessionID,
      directory: root,
      messageID: toolInitial.delivery.continuationMessageID,
      command: "auto",
      arguments: "pause model",
      model: "local/test",
    });
    await expect.poll(() => typeof releaseModel).toBe("function");
    await daemon.client.cancelAgent(agent.id);
    releaseModel!();
    const toolStopped = await toolCommand;
    expect(toolStopped.data?.info.error?.name).toBe("MessageAbortedError");
    expect(
      await readFile(path.join(root, "must-not-exist"), "utf8").then(
        () => true,
        () => false,
      ),
    ).toBe(false);
    toolAttempt = false;
    await daemon.client.sendMessage(agent.id, "/auto manual request");
    await daemon.client.waitForFinish(agent.id, 30_000);
    const manual = JSON.parse(await readFile(observed, "utf8"));
    expect(manual.output.parts[0].metadata?.paseoArrival).toBeUndefined();
    expect(manual.delivery).toBeUndefined();
    await daemon.client.sendMessage(agent.id, "restart request", {
      command: { name: "auto", arguments: "restart request", context: arrival },
    });
    await daemon.client.waitForFinish(agent.id, 30_000);
    const beforeRestart = JSON.parse(await readFile(observed, "utf8"));
    const staleMessageID = beforeRestart.delivery.continuationMessageID;
    await bridge.close();
    await bridge.start();
    await manager.shutdown();
    inspection = await manager.acquireCurrent();
    const restarted = createOpencodeClient({ baseUrl: inspection.server.url, directory: root });
    bridge.bindSession({ sessionId: sessionID, env: {} });
    const staleRoute = await restarted.session.command({
      sessionID,
      directory: root,
      messageID: staleMessageID,
      command: "auto",
      arguments: "must not run after restart",
      model: "local/test",
    });
    expect(staleRoute.error).toBeDefined();
    expect(JSON.parse(await readFile(observed, "utf8")).output.message.id).toBe(
      beforeRestart.output.message.id,
    );
    expect(modelRequests).toBe(1);
  } finally {
    await daemon?.cleanup();
    await inspection?.release();
    await manager.shutdown();
    await bridge.close();
    await providerSnapshotManager.shutdown();
    await new Promise<void>((resolve, reject) =>
      modelServer.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve();
      }),
    );
    await rm(root, { recursive: true, force: true });
  }
}, 120_000);

test("real OpenCode server persists provider permissions across creation and resume", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "paseo-opencode-permissions-"));
  const cwd = path.join(root, "repo");
  const logger = createTestLogger();
  const manager = new OpenCodeServerManager({ logger, resolveHomeDir: () => root });
  const client = new OpenCodeAgentClient(logger, undefined, { serverManager: manager });
  let session: Awaited<ReturnType<OpenCodeAgentClient["createSession"]>> | undefined;
  let inspection: Awaited<ReturnType<OpenCodeServerManager["acquireCurrent"]>> | undefined;

  try {
    await mkdir(cwd);
    session = await client.createSession({
      provider: "opencode",
      cwd,
      providerOptions: { permission: { external_directory: "allow" } },
    });
    inspection = await manager.acquireCurrent();
    const sdk = createOpencodeClient({ baseUrl: inspection.server.url, directory: cwd });
    async function readPermission(sessionId: string) {
      const response = await sdk.session.get({ sessionID: sessionId, directory: cwd });
      if (response.error) throw new Error(JSON.stringify(response.error));
      return response.data?.permission;
    }
    const handle = session.describePersistence()!;
    expect(await readPermission(handle.sessionId)).toEqual([
      { permission: "external_directory", pattern: "*", action: "allow" },
    ]);

    await session.close();
    // OpenCode appends session-update rules and the last matching rule wins, so a resume
    // with a different policy leaves both entries in order.
    session = await client.resumeSession(handle, {
      providerOptions: { permission: { external_directory: "deny" } },
    });
    expect(await readPermission(handle.sessionId)).toEqual([
      { permission: "external_directory", pattern: "*", action: "allow" },
      { permission: "external_directory", pattern: "*", action: "deny" },
    ]);
  } finally {
    await inspection?.release();
    await session?.close();
    await manager.shutdown();
    await rm(root, { recursive: true, force: true });
  }
}, 120_000);

test("real OpenCode server shares one process while shell.env stays session-scoped", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "paseo-opencode-real-"));
  const firstCwd = path.join(root, "first");
  const secondCwd = path.join(root, "second");
  const logger = createTestLogger();
  const bridge = new OpenCodeBridge({ paseoHome: root, logger });
  await bridge.start();
  const firstTools = createCallerCatalog("real-agent-one");
  const secondTools = createCallerCatalog("real-agent-two");
  bridge.setManifestCatalog(firstTools);
  const manager = new OpenCodeServerManager({
    logger,
    resolveHomeDir: () => root,
    decorateServerEnv: (env) => bridge.decorateServerEnv(env),
  });
  const client = new OpenCodeAgentClient(logger, undefined, {
    serverManager: manager,
    bridge,
  });
  let first: Awaited<ReturnType<OpenCodeAgentClient["createSession"]>> | undefined;
  let second: Awaited<ReturnType<OpenCodeAgentClient["createSession"]>> | undefined;
  let inspection: Awaited<ReturnType<OpenCodeServerManager["acquireCurrent"]>> | undefined;

  try {
    await Promise.all([
      mkdir(firstCwd, { recursive: true }),
      mkdir(secondCwd, { recursive: true }),
    ]);
    first = await client.createSession(
      {
        provider: "opencode",
        cwd: firstCwd,
        model: process.env.OPENCODE_TEST_MODEL ?? "opencode/big-pickle",
        modeId: "build",
      },
      {
        agentId: "real-agent-one",
        env: { PASEO_AGENT_ID: "real-agent-one", PASEO_AGENT_CWD: firstCwd },
        paseoTools: firstTools,
      },
      { persistSession: false },
    );
    second = await client.createSession(
      { provider: "opencode", cwd: secondCwd },
      {
        agentId: "real-agent-two",
        env: { PASEO_AGENT_ID: "real-agent-two", PASEO_AGENT_CWD: secondCwd },
        paseoTools: secondTools,
      },
      { persistSession: false },
    );

    inspection = await manager.acquireCurrent();
    const sdk = createOpencodeClient({ baseUrl: inspection.server.url, directory: root });
    const [firstShell, secondShell] = await Promise.all([
      sdk.session.shell({
        sessionID: requireSessionId(first),
        directory: firstCwd,
        agent: "build",
        command: 'printf "%s|%s" "$PASEO_AGENT_ID" "$PASEO_AGENT_CWD"',
      }),
      sdk.session.shell({
        sessionID: requireSessionId(second),
        directory: secondCwd,
        agent: "build",
        command: 'printf "%s|%s" "$PASEO_AGENT_ID" "$PASEO_AGENT_CWD"',
      }),
    ]);

    expect(firstShell.error).toBeUndefined();
    expect(secondShell.error).toBeUndefined();
    expect(JSON.stringify(firstShell.data)).toContain(`real-agent-one|${firstCwd}`);
    expect(JSON.stringify(secondShell.data)).toContain(`real-agent-two|${secondCwd}`);
    expect(inspection.server.url).toMatch(/^http:\/\/127\.0\.0\.1:/);

    // Keep the prompt assertions independent of asynchronous events from the direct shell API.
    await first.close();
    first = await client.createSession(
      {
        provider: "opencode",
        cwd: firstCwd,
        model: process.env.OPENCODE_TEST_MODEL ?? "opencode/big-pickle",
        modeId: "build",
      },
      {
        agentId: "real-agent-one",
        env: { PASEO_AGENT_ID: "real-agent-one", PASEO_AGENT_CWD: firstCwd },
        paseoTools: firstTools,
      },
      { persistSession: false },
    );
    const agentResult = await first.run(
      [
        "Use the bash tool to run: env | grep -E '^(PASEO_AGENT_ID|PASEO_AGENT_CWD)='",
        "Then report both values in your response:",
        "AGENT=real-agent-one",
        `CWD=${firstCwd}`,
      ].join("\n"),
    );
    expect(agentResult.timeline).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "tool_call", name: "bash", status: "completed" }),
      ]),
    );
    const envReport = readAssistantText(agentResult.timeline);
    expect(envReport).toContain("real-agent-one");
    expect(envReport).toContain(firstCwd);

    const callerResult = await first.run(
      [
        "Use the paseo_report_caller_agent_id tool to read your Paseo caller agent ID.",
        "Then report that ID in your response.",
      ].join("\n"),
    );
    expect(callerResult.timeline).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "tool_call",
          name: "paseo_report_caller_agent_id",
          status: "completed",
        }),
      ]),
    );
    expect(readAssistantText(callerResult.timeline)).toContain("real-agent-one");
  } finally {
    await inspection?.release();
    await first?.close();
    await second?.close();
    await manager.shutdown();
    await bridge.close();
    await rm(root, { recursive: true, force: true });
  }
}, 240_000);

test("real OpenCode server asks before running a Paseo tool with an ask rule", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "paseo-opencode-ask-"));
  const cwd = path.join(root, "repo");
  const logger = createTestLogger();
  const bridge = new OpenCodeBridge({ paseoHome: root, logger });
  await bridge.start();
  const catalog = createCallerCatalog("ask-rule-agent");
  bridge.setManifestCatalog(catalog);
  const realSettings = getRealProviderRuntimeSettings("opencode");
  const runtimeSettings = {
    ...realSettings,
    env: {
      ...realSettings.env,
      OPENCODE_CONFIG_CONTENT: JSON.stringify({ permission: { "paseo_*": "ask" } }),
    },
  };
  const manager = new OpenCodeServerManager({
    logger,
    runtimeSettings,
    resolveHomeDir: () => root,
    decorateServerEnv: (env) => bridge.decorateServerEnv(env),
  });
  const client = new OpenCodeAgentClient(logger, runtimeSettings, {
    serverManager: manager,
    bridge,
  });
  let session: Awaited<ReturnType<OpenCodeAgentClient["createSession"]>> | undefined;

  try {
    await mkdir(cwd);
    session = await client.createSession(
      { ...getRealProviderConfig("opencode"), cwd },
      { agentId: "ask-rule-agent", paseoTools: catalog },
      { persistSession: false },
    );
    const asked: string[] = [];
    const completedTools: string[] = [];
    const activeSession = session;
    session.subscribe((event) => {
      if (event.type === "permission_requested") {
        asked.push(event.request.name);
        void activeSession.respondToPermission(event.request.id, { behavior: "allow" });
      }
      if (
        event.type === "timeline" &&
        event.item.type === "tool_call" &&
        event.item.status === "completed"
      ) {
        completedTools.push(event.item.name);
      }
    });

    await session.run(
      "Call the paseo_report_caller_agent_id tool once, then reply with its exact result. Do not use any other tools.",
    );

    expect(completedTools).toContain("paseo_report_caller_agent_id");
    expect(asked).toContain("paseo_report_caller_agent_id");
  } finally {
    await session?.close();
    await manager.shutdown();
    await bridge.close();
    await rm(root, { recursive: true, force: true });
  }
}, 180_000);

function createCallerCatalog(callerAgentId: string): PaseoToolCatalog {
  const tool = {
    name: "report_caller_agent_id",
    title: "Report Paseo caller agent ID",
    description: "Returns the caller agent ID assigned by Paseo.",
    inputSchema: {},
    async handler() {
      return { content: [{ type: "text", text: callerAgentId }] };
    },
  };
  const tools = new Map([[tool.name, tool]]);
  return {
    tools,
    getTool: (name) => tools.get(name),
    async executeTool(name, input, context) {
      const definition = tools.get(name);
      if (!definition) throw new Error(`Unknown tool: ${name}`);
      return definition.handler(input, context ?? {});
    },
  };
}

test("v2 native tool bridge preserves caller identity", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "paseo-opencode-v2-"));
  const logger = createTestLogger();
  const bridge = new OpenCodeBridge({ paseoHome: root, logger });
  const catalog = createCallerCatalog("paseo-v2-caller");
  bridge.setManifestCatalog(catalog);
  await bridge.start();
  const client = new OpenCodeV2AgentClient({ logger, bridge });
  let session: Awaited<ReturnType<typeof client.createSession>> | undefined;
  try {
    session = await client.createSession(
      {
        provider: "opencode",
        cwd: root,
        model: process.env.OPENCODE_TEST_MODEL ?? "openai/gpt-6-astra",
        featureValues: { auto_accept: true },
      },
      { agentId: "paseo-v2-caller", paseoTools: catalog, env: { PASEO_TEST_SCOPE: "V2_ENV_OK" } },
      { persistSession: false },
    );
    const result = await session.run(
      "Call the paseo_report_caller_agent_id tool once, then reply with its exact result. Do not use any other tools.",
    );
    expect(readAssistantText(result.timeline)).toContain("paseo-v2-caller");
    const environment = await session.run(
      "Use the shell tool to run printf '%s' \"$PASEO_TEST_SCOPE\", then reply with its output.",
    );
    expect(environment.finalText).toContain("V2_ENV_OK");
    expect(result.timeline).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "tool_call",
          name: "paseo_report_caller_agent_id",
          status: "completed",
        }),
      ]),
    );
  } finally {
    await session?.interrupt();
    await session?.close();
    await client.shutdown();
    await bridge.close();
    await rm(root, { recursive: true, force: true });
  }
}, 120_000);

test.each([
  {
    major: 1,
    package: "opencode-ai@1.14.46",
    windowsPackage: "opencode-windows",
    expectedTimeline: [],
    expectedInitialTimeline: undefined,
    expectedNotices: undefined,
  },
  {
    major: 2,
    package: "@opencode/cli@2.0.10",
    windowsPackage: "@opencode/cli-windows",
    expectedTimeline: [
      { type: "notification", level: "info", message: "This chat uses OpenCode v2." },
    ],
    expectedInitialTimeline: [
      expect.objectContaining({
        item: { type: "notification", level: "info", message: "This chat uses OpenCode v2." },
      }),
    ],
    expectedNotices: [expect.objectContaining({ major: 2 })],
  },
])(
  "versioned runtime v$major discovers models and preserves a native session handle",
  async ({
    package: cliPackage,
    windowsPackage,
    expectedTimeline,
    expectedInitialTimeline,
    expectedNotices,
  }) => {
    const root = await mkdtemp(path.join(os.tmpdir(), "paseo-opencode-versioned-"));
    const client = new OpenCodeRuntimeClient(createTestLogger(), {
      command: {
        mode: "replace",
        argv: [
          process.platform === "win32"
            ? path.join(
                root,
                "node_modules",
                `${windowsPackage}-${process.arch}`,
                "bin",
                "opencode.exe",
              )
            : path.join(root, "node_modules", ".bin", "opencode"),
        ],
      },
      env: {
        XDG_CONFIG_HOME: path.join(root, "config"),
        XDG_DATA_HOME: path.join(root, "data"),
        XDG_CACHE_HOME: path.join(root, "cache"),
        XDG_STATE_HOME: path.join(root, "state"),
      },
    });
    let original: Awaited<ReturnType<typeof client.createSession>> | undefined;
    let resumed: Awaited<ReturnType<typeof client.resumeSession>> | undefined;
    try {
      // Windows invokes the platform binary directly. Installing its wrapper also runs an
      // unused postinstall binary probe, which has no timeout and can hold the fixture open.
      const installPackage =
        process.platform === "win32"
          ? `${windowsPackage}-${process.arch}@${cliPackage.slice(cliPackage.lastIndexOf("@") + 1)}`
          : cliPackage;
      // npm abandons a stalled registry download only after its 5 minute default, past this
      // test's budget, and the killed install keeps the fixture locked. Retry stalls sooner.
      await execCommand(
        "npm",
        [
          "install",
          "--prefix",
          root,
          "--no-audit",
          "--no-fund",
          "--fetch-timeout=30000",
          installPackage,
        ],
        {
          cwd: root,
          timeout: 180_000,
        },
      );
      const catalog = await client.fetchCatalog({ scope: "workspace", cwd: root, force: true });
      expect(catalog.models.length).toBeGreaterThan(0);
      expect(catalog.modes.map((mode) => mode.id)).toContain("build");
      original = await client.createSession(
        { provider: "opencode", cwd: root, modeId: "build" },
        undefined,
        { persistSession: false },
      );
      const handle = await original.describePersistence();
      expect(original.initialTimeline).toEqual(expectedInitialTimeline);
      expect(handle.metadata?.openCodeRuntimeNotices).toEqual(expectedNotices);
      resumed = await client.resumeSession(handle, { cwd: root });
      expect((await resumed.describePersistence()).nativeHandle).toBe(handle.nativeHandle);
      expect(
        (await drainPersistedTimeline(resumed)).map((event) =>
          event.type === "timeline" ? event.item : event,
        ),
      ).toEqual(expectedTimeline);
    } finally {
      await resumed?.close();
      await original?.close();
      await client.shutdown();
      // Windows can retain a transient executable lock after provider shutdown.
      await rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
    }
  },
  240_000,
);
