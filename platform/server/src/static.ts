import { resolve, sep } from "node:path";

const DIST = resolve(import.meta.dir, "..", process.env.WEB_DIST ?? "../web/dist");

export async function serveStatic(pathname: string): Promise<Response> {
  const index = Bun.file(resolve(DIST, "index.html"));
  if (!(await index.exists()))
    return Response.json({ error: "web app not built (platform/web/dist missing)" }, { status: 503 });

  let rel = pathname;
  try {
    rel = decodeURIComponent(pathname);
  } catch {}
  const path = resolve(DIST, "." + rel);
  if (path.startsWith(DIST + sep)) {
    const f = Bun.file(path);
    if (await f.exists()) {
      const immutable = path.startsWith(resolve(DIST, "assets") + sep);
      return new Response(f, {
        headers: { "Cache-Control": immutable ? "public, max-age=31536000, immutable" : "no-cache" },
      });
    }
  }
  // SPA fallback (but a missing hashed asset is a real 404)
  if (rel.startsWith("/assets/")) return new Response("not found", { status: 404 });
  return new Response(index, { headers: { "Cache-Control": "no-cache" } });
}
