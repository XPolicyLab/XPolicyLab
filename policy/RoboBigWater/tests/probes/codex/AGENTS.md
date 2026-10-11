# Robot interface (M0 probe version)

You control a robot only through the `robo` command. Observations are image files.

- `robo obs` refreshes `obs/head.png`, `obs/wrist_l.png`, `obs/wrist_r.png`, `obs/state.json`.
- `robo move <left|right> [--dx M] [--dy M] [--dz M]` moves one arm. Units are meters.
- `robo wait <sec>` lets the simulation run.
- `robo done` ends the episode.

Every command prints one JSON object. Motion commands refresh `obs/` before they return.
A command can take a few minutes to return. Look at the images with your image viewing tool.
You may write notes to `notes.md`.
