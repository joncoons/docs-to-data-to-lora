"""Tests for the NeMo Platform deployment pin."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nemo_platform_version_pin_matches_ngc_install_path():
    versions_path = REPO_ROOT / "deploy" / "nemo-platform" / "versions.yaml"
    versions = yaml.safe_load(versions_path.read_text())

    platform = versions["nemo_platform"]
    assert platform["docs_version"] == "26.3.1"
    assert platform["chart_org"] == "nvidian"
    assert platform["chart_ref"] == "0857255566152269/external/nemo-platform"
    assert platform["chart_version"] == "2.0.1"
    assert all(image.endswith(":26.03.1") for image in platform["task_images"])


def test_nemo_platform_values_use_existing_ngc_secrets_and_k8s_native_defaults():
    values_path = REPO_ROOT / "deploy" / "nemo-platform" / "values.yaml"
    values = yaml.safe_load(values_path.read_text())

    assert values["existingSecret"] == "ngc-api"
    assert values["existingImagePullSecret"] == "nvcrimagepullsecret"
    assert values["rbac"]["volcanoEnabled"] is True
    assert values["rbac"]["k8sNimOperatorEnabled"] is True
    assert values["k8s-nim-operator"]["enabled"] is True
    assert values["core"]["storage"]["accessModes"] == ["ReadWriteMany"]


def test_public_curator_pin_is_current_but_legacy_microservices_are_not_overstated():
    versions_path = REPO_ROOT / "deploy" / "nemo-platform" / "versions.yaml"
    public_ngc = yaml.safe_load(versions_path.read_text())["public_ngc"]

    assert public_ngc["nemo_curator_image"] == "nvcr.io/nvidia/nemo-curator:26.04"
    assert public_ngc["nim_operator_latest_tag"] == "v3.1.1"
    assert public_ngc["legacy_individual_microservices_latest"] == {
        "evaluator": "25.12",
        "guardrails": "25.12",
    }
