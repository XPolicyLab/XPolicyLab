"""Stable sample-to-prompt identities shared by readers and checkpoints."""
import copy
import hashlib
import json
from pathlib import Path


class EmbodimentRegistry:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.content = json.loads(Path(path).read_text())
        self.validate(self.content)
        self.num_embodiments = self.content["num_embodiments"]
        self.mapping = self.content["profile_to_embodiment_id"]
        self.assigned_ids = tuple(i for i, status in enumerate(
            self.content.get("slot_status", ["assigned"] * self.num_embodiments)) if status == "assigned")
        self.reserved_ids = tuple(i for i in range(self.num_embodiments) if i not in self.assigned_ids)
        self.sha256 = self.content_hash(self.content)

    @staticmethod
    def content_hash(content):
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    @staticmethod
    def validate(content):
        if content.get("schema") not in ("wam.embodiment_registry.v1", "wam.embodiment_registry.v2"):
            raise ValueError("Unsupported embodiment registry schema")
        count = content.get("num_embodiments")
        if type(count) is not int or count <= 0:
            raise ValueError("num_embodiments must be a positive integer")
        names = content.get("embodiments")
        if (not isinstance(names, list) or len(names) != count
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != count):
            raise ValueError("embodiments must contain one unique stable name per ID")
        mapping = content.get("profile_to_embodiment_id")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Empty profile-to-embodiment mapping")
        if any(not isinstance(key, str) or type(value) is not int or not 0 <= value < count
               for key, value in mapping.items()):
            raise ValueError("Invalid profile/embodiment ID")
        if content["schema"] == "wam.embodiment_registry.v2":
            statuses = content.get("slot_status")
            if (not isinstance(statuses, list) or len(statuses) != count
                    or any(s not in ("assigned", "reserved") for s in statuses)):
                raise ValueError("Explicit assigned/reserved status required for every slot")
            assigned = {i for i, status in enumerate(statuses) if status == "assigned"}
            if (type(content.get("assigned_count")) is not int
                    or content["assigned_count"] != len(assigned)
                    or set(mapping.values()) != assigned):
                raise ValueError("Every assigned slot must be mapped; reserved slots cannot be mapped")
            if not isinstance(content.get("version"), str) or not content["version"]:
                raise ValueError("A versioned registry namespace is required")
            contract = content.get("package_contract", {})
            for key in ("package_sha256", "norm_sha256"):
                value = contract.get(key)
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    raise ValueError("Package and normalization SHA256 are required")

    def validate_profiles(self, profiles):
        missing = set(profiles) - self.mapping.keys()
        if missing:
            raise ValueError(f"Unmapped profiles: {sorted(missing)}")

    def id_for(self, profile):
        if profile not in self.mapping:
            raise ValueError(f"Unknown embodiment profile: {profile}")
        return self.mapping[profile]

    def validate_package(self, package):
        self.validate_profiles(package.manifest["profiles"])
        if self.content["schema"] == "wam.embodiment_registry.v2":
            contract = self.content["package_contract"]
            norm_sha = hashlib.sha256((Path(package.root) / "norm.json").read_bytes()).hexdigest()
            if (package.fingerprint != contract["package_sha256"]
                    or norm_sha != contract["norm_sha256"]
                    or set(package.manifest["profiles"]) != set(self.mapping)):
                raise ValueError("Prompt registry package/normalization identity mismatch")

    def metadata(self):
        return {"registry": copy.deepcopy(self.content), "registry_sha256": self.sha256}
