# Control panel

Local web UI for the Reactor → LeRobot → LeIsaac pipeline. Localhost only.

```bash
cd ~/reactor-augmentation
python3 webapp/server.py          # http://127.0.0.1:8080
```

## Tabs

| Tab | What it does |
|---|---|
| **Setup** | Reactor API key (written to `.env`, mode 600, never returned to the browser), GPU and LeIsaac status |
| **Datasets** | Curated HF LeRobot samples, probe any repo for episode count / video keys, fetch one episode and convert it |
| **Augment** | Scene table with the **prompt column**; runs `build_fruit_datasets.py` in ~96-frame batches |
| **Visualize** | Video player over generated episodes, source-vs-augmented preview, action trace plot, per-episode coverage / drift / prompt |
| **LeIsaac** | Teleop, GR00T policy inference, HDF5→LeRobot conversion against `~/leisaac` |
| **Jobs** | Live logs for every background job, with stop |

## Notes

- It **shells out** to the same scripts you would run by hand — nothing is
  reimplemented, so the CLI and the UI cannot drift apart.
- Jobs run detached in their own process group; stopping kills the group,
  because Isaac Sim spawns children that survive a bare terminate on the parent.
- Fetching pulls one episode (video + parquet + meta), not the whole repo.
  It auto-detects the `lerobot/` prefix some datasets use, and falls back to
  ffmpeg when OpenCV cannot decode AV1.
- The augment tab enforces nothing about prompts — the rules are in the banner,
  and the coverage floor is the safety net.
