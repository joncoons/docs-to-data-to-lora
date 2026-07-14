"""Tests for torch-free adapter checkpoint inspection."""

import json
import struct

from scripts.stage3.inspect_adapter_checkpoint import inspect_adapter_dir, write_json_report


def _write_fake_safetensors(path, header):
    data = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(data)) + data)


def test_inspect_adapter_dir_reports_ok_for_minimal_lora(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "/models/base",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 16,
        "target_modules": ["q_proj"] * 16,
    }))
    _write_fake_safetensors(adapter / "adapter_model.safetensors", {
        "layer.lora_A.weight": {"dtype": "BF16", "shape": [16, 8], "data_offsets": [0, 256]},
        "layer.lora_B.weight": {"dtype": "BF16", "shape": [8, 16], "data_offsets": [256, 512]},
    })

    report = inspect_adapter_dir(adapter)

    assert report["status"] == "ok"
    assert report["tensors"]["tensor_count"] == 2
    assert report["tensors"]["lora_a_count"] == 1
    assert report["tensors"]["lora_b_count"] == 1


def test_inspect_adapter_dir_warns_on_thin_config_and_non_lora_tensor(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
    }))
    _write_fake_safetensors(adapter / "adapter_model.safetensors", {
        "layer.lora_A.weight": {"dtype": "BF16", "shape": [16, 8], "data_offsets": [0, 256]},
        "layer.lora_B.weight": {"dtype": "BF16", "shape": [8, 16], "data_offsets": [256, 512]},
        "layer.extra_bias": {"dtype": "F32", "shape": [8], "data_offsets": [512, 544]},
    })

    report = inspect_adapter_dir(adapter)

    assert report["status"] == "warn"
    assert report["tensors"]["non_lora_count"] == 1
    assert any("base_model_name_or_path" in w for w in report["warnings"])


def test_write_json_report_creates_parent_directory(tmp_path):
    report = {"status": "ok", "path": "adapter"}
    out = tmp_path / "reports" / "adapter-inspection.json"

    write_json_report(report, out)

    assert json.loads(out.read_text()) == report
