"""Capture environment-side completion without changing native reward logic."""

import math
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, file_digest


class NativeObserver:
    def __init__(self, output, count):
        self.output = Path(output)
        self.count = count
        self.outcomes = {}
        self.failures = []

    def attach(self, env):
        original = env.run_eval

        def run_eval():
            try:
                active = list(env.get_running_env_idx_list())
                result = original()
                self.capture(env, active)
                return result
            except BaseException as error:
                # Native main may catch errors and still exit zero. A sticky
                # observer failure prevents such a run from being qualified.
                self.failures.append(type(error).__name__ + ": " + str(error))
                self.persist()
                raise

        env.run_eval = run_eval
        return env

    def persist(self):
        atomic_json(
            self.output / "observed.json",
            dict(outcomes=list(self.outcomes.values()), failures=self.failures),
        )

    def capture(self, env, active):
        if env.abandoned_seeds or env.unstable_envs or env.unstable_nums:
            raise ValueError("Unstable/abandoned layouts cannot be scored failures")
        details = list(env.eval_result["details"].values())
        save_dir = Path(env.save_dir).resolve()
        if not save_dir.is_relative_to(self.output.resolve()):
            raise ValueError("Native results escaped the execution directory")
        native_result = save_dir / "_result.json"
        for index in active:
            seed = int(env.env_seeds[index])
            if seed in self.outcomes or not 0 <= seed < self.count:
                raise ValueError("Repeated or unrequested native layout")
            if not env.scene_manager.layout_manager.layout_valid[index]:
                raise ValueError("Native initialization was invalid")
            if not env.end_flag[index]:
                raise ValueError("Native policy episode did not terminate")
            matching = [row for row in details if row["layout_id"] == seed]
            if len(matching) != 1:
                raise ValueError("Missing or duplicate native score")
            row = matching[0]
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid native score")
            if type(row["success"]) is not bool or row["success"] != bool(
                env.success[index]
            ):
                raise ValueError("Native success evidence differs")
            self.outcomes[seed] = dict(
                layout_id=seed,
                success=row["success"],
                episode_score=score,
                natural_terminal=True,
                action_count=int(env.take_action_cnt[index]),
                step_limit=int(env.step_lim),
                result_file=str(native_result.relative_to(self.output.resolve())),
            )
        self.persist()

    def finish(self):
        if self.failures or set(self.outcomes) != set(range(self.count)):
            raise ValueError(
                "Native execution failed or did not cover the exact cohort"
            )
        # Result files can grow across batches; hash their final bytes only.
        for row in self.outcomes.values():
            row["result_sha256"] = file_digest(self.output / row["result_file"])
        self.persist()
        return list(self.outcomes.values())
