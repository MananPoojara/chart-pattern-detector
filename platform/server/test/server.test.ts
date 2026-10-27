// Runs the real server against the real Redis on 6380 under an isolated key prefix,
// simulating the Python worker in-test. Never touches live cpd:* keys.
import { afterAll, beforeAll, expect, test } from "bun:test";
import { RedisClient, type Subprocess } from "bun";

const PREFIX = `cpdtest:${process.pid}:`;
const PORT = 18000 + (process.pid % 1000);
const BASE = `http://127.0.0.1:${PORT}`;
const REDIS_URL = "redis://127.0.0.1:6380";
const r = new RedisClient(REDIS_URL);
let proc: Subprocess;

beforeAll(async () => {
  await r.set(PREFIX + "models", JSON.stringify([{ id: "m1", name: "M1", kind: "rule", url: "", how: "", intraday: true }]));
  await r.set(PREFIX + "universe", JSON.stringify({ nifty: ["RELIANCE", "M&M"] }));
  await r.set(PREFIX + "board", JSON.stringify({ asof: "2026-10-27", rows: [], universe: 2 }));
  await r.hset(PREFIX + "last:FIVE_MINUTE", "RELIANCE", JSON.stringify({ time: 1, open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 }));
  await r.lpush(PREFIX + "signals", JSON.stringify({ id: "s1" }), JSON.stringify({ id: "s2" }));
  proc = Bun.spawn(["bun", "src/index.ts"], {
    cwd: import.meta.dir + "/..",
    env: { ...process.env, PORT: String(PORT), KEY_PREFIX: PREFIX, REDIS_URL, BOARD_POLL_MS: "200" },
    stdout: "ignore",
    stderr: "inherit",
  });
  for (let i = 0; i < 50; i++) {
    try {
      if ((await fetch(BASE + "/api/health")).ok) break;
    } catch {}
    await Bun.sleep(100);
  }
});

afterAll(async () => {
  proc?.kill();
  await proc?.exited;
  const keys = (await r.send("KEYS", [PREFIX + "*"])) as string[];
  if (keys.length) await r.del(...keys);
  r.close();
});

test("health + plain reads", async () => {
  const h = await (await fetch(BASE + "/api/health")).json();
  expect(h).toMatchObject({ ok: true, redis: true, scanner_age_s: null });
  expect((await (await fetch(BASE + "/api/models")).json())[0].id).toBe("m1");
  expect((await (await fetch(BASE + "/api/universe")).json()).nifty).toContain("RELIANCE");
  expect(await (await fetch(BASE + "/api/signals?limit=1")).json()).toEqual([{ id: "s2" }]);
  const res = await fetch(BASE + "/api/board", { headers: { Origin: "http://localhost:5174" } });
  expect(res.headers.get("access-control-allow-origin")).toBe("http://localhost:5174");
});

test("validation -> 400", async () => {
  for (const q of ["model=nope&ticker=RELIANCE", "model=m1&ticker=NOPE", "model=m1&ticker=RELIANCE&interval=1H", "model=m1&ticker=RELIANCE&bars=abc", "model=m1&ticker=RELIANCE&days=0"]) {
    const res = await fetch(`${BASE}/api/run?${q}`);
    expect(res.status).toBe(400);
    expect((await res.json()).error).toBeString();
  }
  expect((await fetch(BASE + "/api/signals?limit=9999")).status).toBe(400);
});

test("/api/run: fake worker, dedupe, cache", async () => {
  const worker = new RedisClient(REDIS_URL);
  const jobs: any[] = [];
  const work = (async () => {
    const popped = (await worker.send("BLPOP", [PREFIX + "jobs", "5"])) as [string, string] | null;
    const job = JSON.parse(popped![1]);
    jobs.push(job);
    await Bun.sleep(300); // let the duplicate request join the in-flight job
    await worker.set(`${PREFIX}result:${job.id}`, JSON.stringify({ id: job.id, model: job.model, ticker: job.ticker, candles: [] }));
    await worker.publish(PREFIX + "done", job.id);
  })();
  const url = `${BASE}/api/run?model=m1&ticker=M%26M&interval=FIVE_MINUTE&bars=100`;
  const [a, b] = await Promise.all([fetch(url).then((x) => x.json()), fetch(url).then((x) => x.json())]);
  await work;
  expect(jobs).toHaveLength(1);
  expect(jobs[0]).toMatchObject({ type: "run", model: "m1", ticker: "M&M", interval: "FIVE_MINUTE", bars: 100, days: 10 });
  expect(a).toEqual(b);
  expect(a.id).toBe(jobs[0].id);
  // cached: no new job is pushed
  expect((await (await fetch(url)).json()).id).toBe(jobs[0].id);
  expect(await r.llen(PREFIX + "jobs")).toBe(0);
  // a forming bar keeps the cache; a closed bar for that ticker invalidates it -> a new job is queued
  const bar = { ticker: "M&M", interval: "FIVE_MINUTE", time: 3, open: 1, high: 1, low: 1, close: 1, volume: 1 };
  await r.publish(PREFIX + "bar:FIVE_MINUTE", JSON.stringify({ ...bar, final: false }));
  await Bun.sleep(100);
  expect((await (await fetch(url)).json()).id).toBe(jobs[0].id);
  await r.publish(PREFIX + "bar:FIVE_MINUTE", JSON.stringify({ ...bar, final: true }));
  await Bun.sleep(100);
  const ac = new AbortController();
  fetch(url, { signal: ac.signal }).catch(() => {});
  for (let i = 0; i < 20 && !(await r.llen(PREFIX + "jobs")); i++) await Bun.sleep(50);
  expect(await r.llen(PREFIX + "jobs")).toBe(1);
  ac.abort();
  await r.del(PREFIX + "jobs");
  worker.close();
});

test("websocket: cached bar on sub, live relay, signals, board", async () => {
  const ws = new WebSocket(`ws://127.0.0.1:${PORT}/ws`);
  const msgs: any[] = [];
  ws.onmessage = (e) => msgs.push(JSON.parse(e.data));
  await new Promise((res) => (ws.onopen = res));
  ws.send("not json");
  ws.send(JSON.stringify({ op: "sub", topics: ["bar:FIVE_MINUTE:RELIANCE", "bars:FIVE_MINUTE", "signals", "board", "evil topic"] }));
  const until = async (n: number) => {
    for (let i = 0; i < 50 && msgs.length < n; i++) await Bun.sleep(50);
  };
  await until(2);
  expect(msgs[0]).toEqual({ topic: "bar:FIVE_MINUTE:RELIANCE", data: { time: 1, open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 } });
  // bars:<iv> sub gets a snapshot of every ticker's last bar, tagged with ticker/interval
  expect(msgs[1]).toEqual({ topic: "bars:FIVE_MINUTE", data: { ticker: "RELIANCE", interval: "FIVE_MINUTE", final: false, time: 1, open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 } });
  msgs.length = 0;

  const bar = { ticker: "RELIANCE", interval: "FIVE_MINUTE", final: false, time: 2, open: 1, high: 1, low: 1, close: 1, volume: 1 };
  await r.publish(PREFIX + "bar:FIVE_MINUTE", JSON.stringify(bar));
  await r.publish(PREFIX + "signal", JSON.stringify({ id: "s3" }));
  await r.set(PREFIX + "board", JSON.stringify({ asof: "2026-10-28", rows: [], universe: 2 }));
  await until(4);
  const topics = msgs.map((m) => m.topic).sort();
  expect(topics).toEqual(["bar:FIVE_MINUTE:RELIANCE", "bars:FIVE_MINUTE", "board", "signals"]);
  expect(msgs.find((m) => m.topic === "bars:FIVE_MINUTE").data).toEqual(bar);
  expect(msgs.find((m) => m.topic === "board").data.asof).toBe("2026-10-28");
  ws.close();
});

test("static: 503 json when dist missing, or SPA fallback", async () => {
  const res = await fetch(BASE + "/some/route");
  expect([200, 503]).toContain(res.status);
  if (res.status === 503) expect((await res.json()).error).toBeString();
});
