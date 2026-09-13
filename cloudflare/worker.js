import { Container } from "@cloudflare/containers";

// One long-lived container instance runs the Python pipeline server
// (map-boundary-web). Static assets are served by Worker Assets before
// this Worker runs; only /api/* (what app.js calls: health, runs, run event
// streams, reports) is proxied to the container. Everything else is answered
// here so scanner noise (/wp-admin, /.env, ...) can't wake the container or
// keep it alive.
//
// Cost note: the container is billed for every second it runs. A 20-minute
// sleepAfter plus a trickle of bot requests kept it up 24 hours a day, which
// was most of the account's Cloudflare bill; with a short idle window it now
// only runs while someone is actually using the app (an SSE run stream counts
// as activity, so an in-progress run is never cut off).
export class MapBoundaryContainer extends Container {
  defaultPort = 8765;
  sleepAfter = "2m";
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const { pathname } = url;
    if (pathname.startsWith("/api/")) {
      const instance = env.MAP_BOUNDARY.getByName("primary");
      return instance.fetch(request);
    }
    if (pathname.startsWith("/static/") && (request.method === "GET" || request.method === "HEAD")) {
      // index.html (served from assets) links /static/app.js etc.; the same
      // files live at the asset root, so serve them from the bundle.
      const assetUrl = new URL(request.url);
      assetUrl.pathname = pathname.slice("/static".length);
      const response = await env.ASSETS.fetch(new Request(assetUrl, request));
      if (response.status !== 404) return response;
    }
    return new Response("Not found", { status: 404, headers: { "cache-control": "public, max-age=300" } });
  },
};
