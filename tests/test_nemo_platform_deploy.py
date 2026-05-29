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



def _first_container(manifest_path: Path) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text())
    return manifest["spec"]["template"]["spec"]["containers"][0]


def _env_by_name(container: dict) -> dict[str, dict]:
    return {item["name"]: item for item in container.get("env", [])}


def _configmap_key(env_item: dict) -> str:
    return env_item["valueFrom"]["configMapKeyRef"]["key"]


def test_service_plane_configmap_defines_platform_aliases():
    configmap_path = REPO_ROOT / "deploy" / "nemo-platform" / "service-plane-configmap.yaml"
    configmap = yaml.safe_load(configmap_path.read_text())

    assert configmap["metadata"]["name"] == "nemo-platform-service-plane"
    data = configmap["data"]
    assert data["NMP_BASE_URL"] == "http://nemo-platform-api:8080"
    assert data["NMP_WORKSPACE"] == "default"
    for key in (
        "NMP_CUSTOMIZER_URL",
        "NMP_DATA_DESIGNER_URL",
        "NMP_EVALUATOR_URL",
        "NMP_INFERENCE_GATEWAY_URL",
        "NMP_ENTITY_STORE_URL",
        "NMP_DATASTORE_URL",
        "NMP_DATASTORE_HF_ENDPOINT",
        "NMP_DATASTORE_GIT_BASE",
    ):
        assert key in data


def test_k8s_jobs_source_platform_service_plane_configmap():
    expectations = {
        "dataset-registration/job.yaml": {
            "ENTITY_STORE_URL": "NMP_ENTITY_STORE_URL",
            "DATA_STORE_URL": "NMP_DATASTORE_URL",
            "DATA_STORE_GIT_BASE": "NMP_DATASTORE_GIT_BASE",
            "DATA_STORE_HF_ENDPOINT": "NMP_DATASTORE_HF_ENDPOINT",
        },
        "evaluator-registration/job.yaml": {
            "EVALUATOR_URL": "NMP_EVALUATOR_URL",
            "NIM_PROXY_URL": "NMP_INFERENCE_GATEWAY_URL",
        },
        "evaluation-matrix/job.yaml": {
            "EVALUATOR_URL": "NMP_EVALUATOR_URL",
        },
        "data-designer-gapfill/job.yaml": {
            "NEMO_MICROSERVICES_BASE_URL": "NMP_DATA_DESIGNER_URL",
            "NEMO_MICROSERVICES_DATASTORE_ENDPOINT": "NMP_DATASTORE_HF_ENDPOINT",
        },
        "ties-merge/job.yaml": {
            "DATA_STORE_GIT_BASE": "NMP_DATASTORE_GIT_BASE",
        },
    }

    for rel_path, env_expectations in expectations.items():
        container = _first_container(REPO_ROOT / "deploy" / rel_path)
        assert container["envFrom"] == [
            {"configMapRef": {"name": "nemo-platform-service-plane"}}
        ]
        env = _env_by_name(container)
        for env_name, key in env_expectations.items():
            assert _configmap_key(env[env_name]) == key
