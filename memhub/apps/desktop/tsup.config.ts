import { createRequire } from "node:module";
import { defineConfig } from "tsup";

const require = createRequire(import.meta.url);
const { resolveBuildVersion } =
  require("../../scripts/release-version.cjs") as {
    resolveBuildVersion(environment?: NodeJS.ProcessEnv, root?: string): string;
  };
const releaseVersion = resolveBuildVersion(process.env);

export default defineConfig({
  entry: ["src/main.ts", "src/preload.ts"],
  format: ["cjs"],
  platform: "node",
  outDir: "dist",
  external: ["electron"],
  noExternal: [
    "@one-memory/host-bridge",
    "@one-memory/contracts",
    "@one-memory/runtime-control",
    "axios",
    "https-proxy-agent",
    "proxy-from-env",
  ],
  define: {
    __ONE_MEMORY_VERSION__: JSON.stringify(releaseVersion),
    __ONE_MEMORY_TRIAL_PROFILE__: JSON.stringify(
      process.env.ONE_MEMORY_BUILD_TRIAL_PROFILE ?? null,
    ),
    __ONE_MEMORY_BUILD_NUMBER__: JSON.stringify(
      process.env.ONE_MEMORY_BUILD_NUMBER ?? "unknown",
    ),
    __ONE_MEMORY_COMMIT__: JSON.stringify(
      process.env.ONE_MEMORY_COMMIT ?? "unknown",
    ),
  },
});
