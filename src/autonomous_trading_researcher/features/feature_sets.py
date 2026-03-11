"""Feature set versioning helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import json


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class FeatureSet:
    """Represents a versioned feature bundle."""

    feature_set_id: str
    feature_names: list[str]
    dataset_version: str
    creation_time: datetime

    def to_record(self) -> dict[str, object]:
        payload = asdict(self)
        payload["creation_time"] = self.creation_time.isoformat()
        return payload


class FeatureSetStore:
    """Persist and load feature set manifests."""

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_feature_set_id(feature_names: Iterable[str], dataset_version: str) -> str:
        payload = {
            "dataset_version": dataset_version,
            "feature_names": sorted(set(feature_names)),
        }
        digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        return f"fs-{stamp}-{digest}"

    def create(
        self,
        feature_names: Iterable[str],
        dataset_version: str,
    ) -> FeatureSet:
        feature_names_list = sorted(set(feature_names))
        feature_set_id = self._build_feature_set_id(feature_names_list, dataset_version)
        feature_set = FeatureSet(
            feature_set_id=feature_set_id,
            feature_names=feature_names_list,
            dataset_version=dataset_version,
            creation_time=_utc_now(),
        )
        self.save(feature_set)
        return feature_set

    def save(self, feature_set: FeatureSet) -> Path:
        target = self.base_path / f"{feature_set.feature_set_id}.json"
        target.write_text(json.dumps(feature_set.to_record(), indent=2), encoding="utf-8")
        return target

    def load(self, feature_set_id: str) -> FeatureSet | None:
        path = self.base_path / f"{feature_set_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FeatureSet(
            feature_set_id=str(payload["feature_set_id"]),
            feature_names=list(payload.get("feature_names", [])),
            dataset_version=str(payload.get("dataset_version", "")),
            creation_time=datetime.fromisoformat(payload["creation_time"]),
        )
