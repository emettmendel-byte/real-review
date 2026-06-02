import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the workspace root to this app so Next doesn't pick up a stray
  // lockfile higher up the tree (the Python repo root has none, but the
  // user's home dir may).
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
