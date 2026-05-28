"""
product_url_map.py — URL-prefix → (product_family, product_name) routing table.

Longest-prefix match: entries are tried in order; the first host+path prefix
that is a substring-prefix of the URL wins.

Kept in sync with nvidia_rag.utils.configuration.CRAWLER_PRODUCT_URL_MAP.
Override at runtime by setting APP_CRAWLER_PRODUCT_MAP to a JSON file path
(see config.py).
"""

from __future__ import annotations

import json
import logging
import os

from . import config

logger = logging.getLogger(__name__)

# ── Canonical URL-prefix table (longest-prefix-first within each product) ────
_DEFAULT_MAP: list[tuple[str, str, str | None]] = [
    # ── CUDA ecosystem ────────────────────────────────────────────────────────
    ("docs.nvidia.com/cuda/profiler-users-guide",   "CUDA",               "CUDA Profiler"),
    ("docs.nvidia.com/cuda/cuda-gdb",               "CUDA",               "CUDA-GDB"),
    ("docs.nvidia.com/cuda/cuda-memcheck",          "CUDA",               "CUDA-MEMCHECK"),
    ("docs.nvidia.com/cuda/thrust",                 "CUDA",               "Thrust"),
    ("docs.nvidia.com/cuda",                        "CUDA",               "CUDA Toolkit"),
    ("developer.nvidia.com/cuda",                   "CUDA",               "CUDA Toolkit"),
    # ── TensorRT ──────────────────────────────────────────────────────────────
    ("docs.nvidia.com/tensorrt",                    "Inference",          "TensorRT"),
    ("docs.nvidia.com/deeplearning/tensorrt",       "Inference",          "TensorRT"),
    # ── cuDNN ─────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/cudnn",                       "RAPIDS",             "cuDNN"),
    ("docs.nvidia.com/deeplearning/cudnn",          "RAPIDS",             "cuDNN"),
    # ── Deep Learning ─────────────────────────────────────────────────────────
    ("docs.nvidia.com/nemo/microservices",          "NeMo Microservices",  "NVIDIA NeMo Microservices"),
    ("docs.nvidia.com/nim-operator",                "NeMo Microservices",  "NVIDIA NIM Operator"),
    ("docs.nvidia.com/deeplearning/nemo",           "NeMo",               "NeMo Framework"),
    ("docs.nvidia.com/deeplearning/triton",         "Inference",          "Triton Inference Server"),
    ("docs.nvidia.com/deeplearning/performance",    "Deep Learning",      "Deep Learning Performance"),
    ("docs.nvidia.com/deeplearning/frameworks",     "Deep Learning",      "Deep Learning Frameworks"),
    ("docs.nvidia.com/deeplearning",                "Deep Learning",      None),
    ("developer.nvidia.com/deep-learning",          "Deep Learning",      None),
    # ── Triton (GitHub) ───────────────────────────────────────────────────────
    ("github.com/triton-inference-server",          "Inference",          "Triton Inference Server"),
    # ── NIM / NIMs ────────────────────────────────────────────────────────────
    ("docs.nvidia.com/nim",                         "NIM",                "NVIDIA NIM"),
    ("build.nvidia.com",                            "NIM",                "NVIDIA NIM"),
    # ── Data Center / GPUs ────────────────────────────────────────────────────
    ("docs.nvidia.com/datacenter/tesla",            "Data Center",        "Tesla GPU"),
    ("docs.nvidia.com/datacenter/cloud-native",     "Data Center",        "Cloud Native"),
    ("docs.nvidia.com/datacenter",                  "Data Center",        "Data Center GPU"),
    ("docs.nvidia.com/dgx",                         "DGX",                "NVIDIA DGX"),
    ("docs.nvidia.com/gpudirect-storage",           "Data Center",        "GPUDirect Storage"),
    # ── NVIDIA Virtual GPU (vGPU / GRID) ──────────────────────────────────────
    ("docs.nvidia.com/vgpu",                        "NVIDIA Virtual GPU", "NVIDIA vGPU Software"),
    ("docs.nvidia.com/grid",                        "NVIDIA Virtual GPU", "NVIDIA GRID"),
    ("docscontent.nvidia.com/dita",                 "NVIDIA Virtual GPU", "NVIDIA GRID/vGPU"),
    # ── CUDA deployment tools ─────────────────────────────────────────────────
    ("docs.nvidia.com/deploy",                      "CUDA",               "CUDA Deployment Tools"),
    # ── GPU architectures ─────────────────────────────────────────────────────
    ("docs.nvidia.com/blackwell",                   "Blackwell",          "Blackwell Architecture"),
    ("docs.nvidia.com/hopper",                      "Hopper",             "Hopper Architecture"),
    ("docs.nvidia.com/ampere",                      "Ampere",             "Ampere Architecture"),
    ("developer.nvidia.com/blackwell",              "Blackwell",          "Blackwell Architecture"),
    # ── Networking ────────────────────────────────────────────────────────────
    ("docs.nvidia.com/networking/mlnx_ofed",        "Networking",         "MLNX OFED"),
    ("docs.nvidia.com/networking",                  "Networking",         "NVIDIA Networking"),
    ("docs.mellanox.com",                           "Networking",         "NVIDIA Networking"),
    # ── Autonomous vehicles ───────────────────────────────────────────────────
    ("docs.nvidia.com/drive",                       "DRIVE",              "NVIDIA DRIVE"),
    # ── Embedded / Jetson ─────────────────────────────────────────────────────
    ("docs.nvidia.com/jetson",                      "Jetson",             "Jetson"),
    ("developer.nvidia.com/embedded",               "Jetson",             "Jetson"),
    ("developer.nvidia.com/jetson",                 "Jetson",             "Jetson"),
    # ── IGX (Intelligent Edge / Industrial) ───────────────────────────────────
    ("docs.nvidia.com/igx",                         "IGX",                "NVIDIA IGX"),
    # ── PVA (Programmable Vision Accelerator) ─────────────────────────────────
    ("docs.nvidia.com/pva",                         "Jetson",             "NVIDIA PVA"),
    # ── Isaac ─────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/isaac",                       "Isaac",              "Isaac"),
    ("developer.nvidia.com/isaac",                  "Isaac",              "Isaac"),
    # ── Profiling / Dev tools ─────────────────────────────────────────────────
    ("docs.nvidia.com/nsight-systems",              "CUDA",               "Nsight Systems"),
    ("docs.nvidia.com/nsight-compute",              "CUDA",               "Nsight Compute"),
    ("docs.nvidia.com/nsight-visual-studio-code",   "CUDA",               "Nsight VSCode"),
    ("docs.nvidia.com/nsight",                      "CUDA",               "Nsight"),
    # ── RAPIDS ────────────────────────────────────────────────────────────────
    ("docs.rapids.ai",                              "RAPIDS",             "RAPIDS"),
    ("developer.nvidia.com/rapids",                 "RAPIDS",             "RAPIDS"),
    ("rapids.ai",                                   "RAPIDS",             "RAPIDS"),
    ("docs.nvidia.com/rapids",                      "RAPIDS",             "RAPIDS"),
    ("docs.nvidia.com/cuvs",                        "RAPIDS",             "cuVS"),
    ("docs.nvidia.com/spark-rapids",                "RAPIDS",             "Spark RAPIDS"),
    ("docs.nvidia.com/cupynumeric",                 "RAPIDS",             "cuPyNumeric"),
    ("docs.nvidia.com/legate",                      "RAPIDS",             "Legate NumPy"),
    # ── Modulus / Physics ML ──────────────────────────────────────────────────
    ("docs.nvidia.com/modulus",                     "Physical AI",        "NVIDIA Modulus"),
    ("developer.nvidia.com/modulus",                "Physical AI",        "NVIDIA Modulus"),
    ("docs.nvidia.com/physicsnemo",                 "Physical AI",        "PhysicsNeMo"),
    # ── HPC ───────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/hpc-sdk",                     "HPC",                "NVIDIA HPC SDK"),
    ("docs.nvidia.com/nvpl",                        "HPC",                "NVIDIA Performance Libraries"),
    ("docs.nvidia.com/nvshmem",                     "HPC",                "NVSHMEM"),
    ("support.brightcomputing.com",                 "HPC",                "Bright Computing HPC"),
    # ── Holoscan / Clara ──────────────────────────────────────────────────────
    ("docs.nvidia.com/holoscan",                    "Holoscan",           "NVIDIA Holoscan"),
    ("docs.nvidia.com/clara-holoscan",              "Holoscan",           "NVIDIA Holoscan"),
    ("docs.nvidia.com/clara",                       "Holoscan",           "NVIDIA Clara"),
    # ── Aerial (CUDA-Accelerated RAN) ─────────────────────────────────────────
    ("docs.nvidia.com/aerial",                      "Aerial",             "NVIDIA Aerial"),
    # ── TAO Toolkit ───────────────────────────────────────────────────────────
    ("docs.nvidia.com/tao",                         "TAO Toolkit",        "NVIDIA TAO Toolkit"),
    # ── NeMo Framework ────────────────────────────────────────────────────────
    ("docs.nvidia.com/nemo",                        "NeMo",               "NeMo Framework"),
    ("docs.nvidia.com/nemo-framework",              "NeMo",               "NeMo Framework"),
    # ── BioNeMo ───────────────────────────────────────────────────────────────
    ("docs.nvidia.com/bionemo-framework",           "BioNeMo",            "NVIDIA BioNeMo"),
    # ── Cosmos ────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/cosmos",                      "Physical AI",        "NVIDIA Cosmos"),
    # ── Metropolis ────────────────────────────────────────────────────────────
    ("docs.nvidia.com/metropolis",                  "Metropolis",         "NVIDIA Metropolis"),
    # ── Morpheus ──────────────────────────────────────────────────────────────
    ("docs.nvidia.com/morpheus",                    "Morpheus",           "NVIDIA Morpheus"),
    # ── Maxine ────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/maxine",                      "Conversational AI",  "NVIDIA Maxine"),
    # ── DOCA (Data Center / Networking) ───────────────────────────────────────
    ("docs.nvidia.com/doca",                        "Networking",         "NVIDIA DOCA"),
    # ── cuOpt ─────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/cuopt",                       "RAPIDS",             "NVIDIA cuOpt"),
    # ── ACE (AI Character Engine) ──────────────────────────────────────────────
    ("docs.nvidia.com/ace",                         "AI Enterprise",      "NVIDIA ACE"),
    # ── NGC Catalog ───────────────────────────────────────────────────────────
    ("docs.nvidia.com/ngc",                         "NGC",                "NGC Catalog"),
    ("assets.ngc.nvidia.com",                       "NGC",                "NGC Catalog"),
    # ── Base Command Platform / LaunchPad ─────────────────────────────────────
    ("docs.nvidia.com/base-command-platform",       "Infrastructure",     "NVIDIA Base Command Platform"),
    ("docs.nvidia.com/base-command-manager",        "Infrastructure",     "NVIDIA Base Command Manager"),
    ("docs.nvidia.com/launchpad",                   "Infrastructure",     "NVIDIA LaunchPad"),
    # ── Dynamo ────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/dynamo",                      "Dynamo",             "NVIDIA Dynamo"),
    ("docs.dynamo.nvidia.com",                      "Dynamo",             "NVIDIA Dynamo"),
    ("github.com/ai-dynamo",                        "Dynamo",             "NVIDIA Dynamo"),
    # ── AI Enterprise / Fleet Command / Licensing ─────────────────────────────
    ("docs.nvidia.com/ai-enterprise",               "AI Enterprise",      "NVIDIA AI Enterprise"),
    ("docs.nvidia.com/fleet-command",               "Fleet Command",      "Fleet Command"),
    ("docs.nvidia.com/license-system",              "AI Enterprise",      "NVIDIA License System"),
    # ── CUDA tooling (compute-sanitizer, cutlass, cupti) ──────────────────────
    ("docs.nvidia.com/compute-sanitizer",           "CUDA",               "CUDA Compute Sanitizer"),
    ("docs.nvidia.com/cutlass",                     "CUDA",               "CUTLASS"),
    ("docs.nvidia.com/cupti",                       "CUDA",               "CUDA CUPTI"),
    # ── Data Center infra (HGX, multi-node NVLink, attestation) ──────────────
    ("docs.nvidia.com/hgx-platforms",               "Data Center",        "NVIDIA HGX"),
    ("docs.nvidia.com/multi-node-nvlink-systems",   "Data Center",        "Multi-Node NVLink"),
    ("docs.nvidia.com/attestation",                 "Data Center",        "NVIDIA Attestation"),
    # ── Riva (Conversational AI / ASR / TTS) ──────────────────────────────────
    ("docs.nvidia.com/riva",                        "Conversational AI",  "NVIDIA Riva"),
    # ── MONAI (Medical Imaging) ───────────────────────────────────────────────
    ("docs.nvidia.com/monai",                       "MONAI",              "NVIDIA MONAI"),
    # ── nvTrust (Confidential Computing / Attestation) ────────────────────────
    ("docs.nvidia.com/nvtrust",                     "Data Center",        "NVIDIA nvTrust"),
    # ── OpenShell ─────────────────────────────────────────────────────────────
    ("docs.nvidia.com/openshell",                   "NVIDIA",             "NVIDIA OpenShell"),
    # ── AI-GRID / GRID virtualization ─────────────────────────────────────────
    ("docs.nvidia.com/ai-grid",                     "NVIDIA Virtual GPU", "NVIDIA AI-GRID"),
    # ── DCCPU (Data Center CPU) ───────────────────────────────────────────────
    ("docs.nvidia.com/dccpu",                       "Data Center",        "NVIDIA Data Center CPU"),
    # ── VCR SDK (Video Codec Research) ───────────────────────────────────────
    ("docs.nvidia.com/vcr-sdk",                     "NVIDIA",             "NVIDIA VCR SDK"),
    # ── LLM Inference Quick-Start Recipes ─────────────────────────────────────
    ("docs.nvidia.com/llm-inference-quick-start-recipes", "NIM",          "LLM Inference Quick-Start"),
    # ── Run.ai ────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/run-ai",                      "Run.ai",             "Run.ai Platform"),
    # ── NVIDIA Blueprints (RAG, AIQ, etc.) ───────────────────────────────────
    ("docs.nvidia.com/rag",                         "NVIDIA Blueprints",  "NVIDIA RAG Blueprint"),
    ("docs.nvidia.com/aiq-blueprint",               "NVIDIA Blueprints",  "NVIDIA AIQ Blueprint"),
    # ── Cloud Functions ───────────────────────────────────────────────────────
    ("docs.nvidia.com/cloud-functions",             "NVCF",               "NVIDIA Cloud Functions"),
    # ── Video Technologies ────────────────────────────────────────────────────
    ("docs.nvidia.com/video-technologies",          "Metropolis",         "NVIDIA Video Technologies"),
    # ── Mission Control ───────────────────────────────────────────────────────
    ("docs.nvidia.com/mission-control",             "Infrastructure",     "NVIDIA Mission Control"),
    # ── NCX (NVIDIA Cluster Experience) ──────────────────────────────────────
    ("docs.nvidia.com/ncx",                         "Infrastructure",     "NVIDIA NCX"),
    # ── Certification Programs ────────────────────────────────────────────────
    ("docs.nvidia.com/certification-programs",      "Certification",      "NVIDIA Certification Programs"),
    # ── VPI (Vision Programming Interface) ───────────────────────────────────
    ("docs.nvidia.com/vpi",                         "Metropolis",         "NVIDIA VPI"),
    # ── Developer blog / general ──────────────────────────────────────────────
    ("developer.nvidia.com/blog",                   "Blog",               "NVIDIA Developer Blog"),
    ("developer.nvidia.com/technical-blog",         "Blog",               "NVIDIA Technical Blog"),
    ("developer.nvidia.com",                        "NVIDIA Developer",   None),
    ("docs.nvidia.com",                             "NVIDIA Docs",        None),
    ("www.nvidia.com/en-us/technologies",           "NVIDIA",             None),
    ("github.com/nvidia",                           "NVIDIA",             None),
    ("github.com/NVIDIA",                           "NVIDIA",             None),
]


def _load_map() -> list[tuple[str, str, str | None]]:
    """Return the product URL map, optionally overridden from a JSON file."""
    path = config.PRODUCT_MAP_PATH
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                raw = json.load(f)
            loaded = [(r[0], r[1], r[2] if len(r) > 2 else None) for r in raw]
            logger.info("product_url_map: loaded %d entries from %s", len(loaded), path)
            return loaded
        except Exception as exc:
            logger.warning("product_url_map: failed to load %s: %r — using defaults", path, exc)
    return _DEFAULT_MAP


# Module-level constant; imported by crawl.py as CRAWLER_PRODUCT_URL_MAP
CRAWLER_PRODUCT_URL_MAP: list[tuple[str, str, str | None]] = _load_map()
