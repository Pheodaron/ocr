import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  timeout: 40000,
  use: {
    baseURL: "http://127.0.0.1:8011",
    launchOptions: {
      executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
      args: ["--no-sandbox"],
    },
    screenshot: "only-on-failure",
  },
  webServer: {
    command:
      "cd .. && UV_CACHE_DIR=/tmp/train-app-uv uv run uvicorn browser_server:app --app-dir tests --host 127.0.0.1 --port 8011",
    url: "http://127.0.0.1:8011/health",
    reuseExistingServer: false,
  },
});
