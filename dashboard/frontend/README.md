# SpectatorAI frontend

React + TypeScript + Vite, with Three.js for interactive hand visualization.

From this folder:

```sh
npm ci
npm run dev
npm run build
```

The development server proxies API requests to the local backend on port 8000.
The production build is served by that backend. Use the root `run.bat` after setup for the full app.

Pages include Workspace, Library, Dataset, Room (placeholder), and Settings.
Dataset plays existing local output videos; it does not run the tracking pipeline.
See [dashboard setup](../README.md) and the [project README](../../README.md).
