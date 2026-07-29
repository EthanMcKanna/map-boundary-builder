import { Container } from "@cloudflare/containers";

// One long-lived container instance runs the Python pipeline server
// (map-boundary-web). Static assets are served by Worker Assets before
// this Worker runs; everything else — /api/* including SSE event
// streams — proxies straight through to the container.
export class MapBoundaryContainer extends Container {
  defaultPort = 8765;
  sleepAfter = "20m";
}

export default {
  async fetch(request, env) {
    const instance = env.MAP_BOUNDARY.getByName("primary");
    return instance.fetch(request);
  },
};
