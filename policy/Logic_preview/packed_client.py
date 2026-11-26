"""Keep native layout outcomes on RoboDojo clients that delete their final manifest."""
from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def client_checkout(benchmark):
    """Give the legacy client a private copy of its two bookkeeping modules."""
    source = Path("src/eval_client")
    main = benchmark / source / "main.py"
    cleanup = "    _delete_resume_manifest(env)\n"
    if not main.is_file() or cleanup not in (main_text := main.read_text()):
        yield benchmark
        return

    env_text = (benchmark / source / "eval_env.py").read_text()
    # At these two points RoboDojo has identified the precise unstable envs.
    # Map them to the native layout IDs before RoboDojo advances its seed queue.
    for indent, ids in [(12, "unstable_envs"), (16, "unstable_in_batch")]:
        count = " " * indent + f"self.unstable_nums += len({ids})\n"
        update = f"self.abandoned_seeds.update(self.get_seeds_for_envs({ids}))"
        if update not in env_text:
            if env_text.count(count) != 1:
                raise RuntimeError("Unsupported RoboDojo instability bookkeeping in " + str(main.parent))
            indent_text = " " * indent
            save = indent_text + f"if {ids}:\n" if ids == "unstable_envs" else ""
            body = indent_text + ("    " if ids == "unstable_envs" else "")
            save += body + update + "\n" + body + "self.persist_resume_manifest()\n"
            env_text = env_text.replace(count, count + save)
    with tempfile.TemporaryDirectory(prefix="logic-preview-client-") as tmp:
        root = Path(tmp)
        for relative in (Path("."), Path("src"), source):
            (root / relative).mkdir(exist_ok=True)
            for entry in (benchmark / relative).iterdir():
                name = relative / entry.name
                if name in (Path("src"), source, source / "main.py", source / "eval_env.py"):
                    continue
                if entry.name != "__pycache__":
                    (root / name).symlink_to(entry)
        (root / source / "eval_env.py").write_text(env_text)
        # The scheduler archives this final native snapshot with the attempt.
        (root / source / "main.py").write_text(main_text.replace(cleanup, "    env.persist_resume_manifest()\n"))
        yield root
