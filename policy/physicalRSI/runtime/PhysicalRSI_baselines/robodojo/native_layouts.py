"""Explicit fresh-layout input for a dedicated RoboDojo simulator process.

This replaces the native SeedManager input methods, never its scoring. It is
not a sandbox for generated policies. Install before constructing any env.
"""

from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, read_json, relative_path


def bind_layouts(seed_manager_class, *, task, split, root, rows):
    """Return a SeedManager subclass which cannot fall back to Eval_Layout.

    The caller must install this class before importing simulator consumers.
    A fresh process gets exactly one task/cohort; resume filtering is rejected
    rather than silently reducing the required comparison coverage.
    """
    root = Path(root).resolve()
    if split not in {"development", "validation"} or not rows:
        raise ValueError("Expected a nonempty development or validation cohort")
    approved = []
    seen = set()
    for row in rows:
        if row["task"] != task or row["split"] != split:
            raise ValueError("Layout task/split differs from requested cohort")
        # Retain the unresolved path too: changing a symlink after binding must
        # not redirect a subsequent native reset to a different input.
        path = root / relative_path(row["file"])
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or any(
            part in {"Eval_Layout", "Train_Layout"} for part in resolved.parts
        ):
            raise ValueError("Native layout is outside the fresh-layout cohort")
        sha = digest(read_json(resolved))
        if sha != row["layout_sha256"] or sha in seen:
            raise ValueError("Changed or duplicate native layout")
        approved.append((path, resolved, sha))
        seen.add(sha)

    class FreshSeedManager(seed_manager_class):
        def init_eval(self, completed_layout_ids=None, abandoned_layout_ids=None):
            if self.task_name != task:
                raise ValueError("Native task differs from bound layout task")
            if list(completed_layout_ids or ()) or list(abandoned_layout_ids or ()):
                raise ValueError("Fresh cohort cannot skip completed/abandoned layouts")
            if not 1 <= self.num_envs <= 10:
                raise ValueError("Native cohort supports one to ten environments")
            self.eval_seed = self.config.get("seed", 0)
            self.seed_info = {
                index: {"scene_layout": str(path)}
                for index, (path, _, _) in enumerate(approved)
            }
            self.seed_list = list(range(len(approved)))
            self.st_idx, self.ed_idx, self.idx = 0, len(approved), 0
            self.type = "eval"
            self._current_batch_seeds = None

        def get_seeds(self, max_count=None):
            # Explicit size disables upstream's duplicate-last-layout padding.
            return super().get_seeds(self.num_envs if max_count is None else max_count)

        def get_seed_scene_info(self, seed):
            if type(seed) is not int or not 0 <= seed < len(approved):
                raise ValueError("Native seed is not in the approved cohort")
            path, resolved, sha = approved[seed]
            if path.resolve(strict=True) != resolved:
                raise ValueError("Native layout path changed")
            data = read_json(path)
            if digest(data) != sha:
                raise ValueError("Native layout changed before reset")
            return data

    return FreshSeedManager
