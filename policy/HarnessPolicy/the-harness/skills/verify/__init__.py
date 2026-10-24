"""The success verifier, asked from inside a program."""


class Verify:
    def verify(self, claim, cameras=None):
        """Ask the success verifier whether `claim` holds in the current camera views.
        Returns (holds, answer): holds is True for yes, False for no, None when the
        verifier is disabled in this configuration."""
        if self.verifier is None:
            return None, "The verifier is disabled in this configuration."
        from harness_policy.runtime.driver import VERIFIER_SYSTEM

        observation = self.get_observation(cameras=cameras)
        answer = self.verifier.ask([observation.images[c] for c in observation.images],
                                   f"Claim: {claim}", system=VERIFIER_SYSTEM).strip()
        return answer.upper().startswith("YES"), answer
