import type { NextConfig } from "next";

const isGitHubPages = process.env.GITHUB_ACTIONS === "true";
const repositoryBasePath = "/tpu-roofline-explorer";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath: isGitHubPages ? repositoryBasePath : undefined,
  assetPrefix: isGitHubPages ? repositoryBasePath : undefined,
  typescript: {
    tsconfigPath: "tsconfig.pages.json",
  },
};

export default nextConfig;
