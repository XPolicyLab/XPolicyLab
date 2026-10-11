# Astra 42-task ICL bank
12 self-success + 2 approved-open + 28 official-video fallbacks.
Existing compact content is preserved byte-for-byte. Videos supply chronological visual references only.
Usage: add this directory to sys.path, then `from loader import load_messages`; call `load_messages(task, max_images=...)` and insert the returned messages after the system prompt and before current observations.
Load only the current task. Image-budget overflow raises an error. No live worker or frozen release was modified.
Open index.html to inspect every task. validation.json records successful image/message checks for all 42 tasks.
