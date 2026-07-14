# License and evidence plan

This is an engineering evidence checklist, not legal advice.

## Current findings

As checked on 2026-06-29:

- The upstream NeMo Curator repository identifies the project as Apache
  License 2.0 in its package metadata and source headers.
- The upstream license grants copyright and patent permissions subject to its
  conditions, including preservation of the license and applicable notices on
  redistribution and prominent notices on modified files.
- NVIDIA's current documentation identifies NeMo Curator 26.04 / v1.1.2 as the
  latest stable release and documents `DiverseQAStage` for generating diverse
  QA pairs from document text.

Primary references:

- [NeMo Curator repository](https://github.com/NVIDIA-NeMo/Curator)
- [Upstream LICENSE](https://github.com/NVIDIA-NeMo/Curator/blob/main/LICENSE)
- [26.04 synthetic data generation overview](https://docs.nvidia.com/nemo/curator/curate-text/synthetic)
- [Nemotron-CC / DiverseQA workflow](https://docs.nvidia.com/nemo/curator/curate-text/synthetic/nemotron-cc)
- [26.04 installation guidance](https://docs.nvidia.com/nemo/curator/get-started/installation)

## Important separation of rights

Curator's software license does not by itself determine:

- the license or usage terms of the NIM/container registry;
- the model checkpoint or inference-service terms;
- rights in the source documentation corpora;
- rights in generated QA data or trained adapters;
- licenses of every dependency bundled in the container.

These are separate evidence tracks. The final report must not describe all
outputs as "Apache-2.0 licensed" merely because Curator code is Apache-2.0.

## Evidence to capture before execution

1. Curator release/tag, Git commit if source is copied, container reference,
   immutable digest, retrieval date, and SHA-256 of the upstream LICENSE.
2. Every upstream NOTICE, third-party license bundle, and container SBOM.
3. Package lockfile and installed-package inventory from the actual image.
4. Any experiment-local modifications to upstream example/helper files, with
   headers and a change log identifying modified files.
5. Model identifier/checkpoint digest where available, provider, endpoint class,
   and the model/service terms effective on the run date.
6. Input corpus identifiers, hashes, acquisition/source metadata, and the
   internal authorization or license record governing their use.
7. Exact native Curator prompts/templates, common filter configs, seeds, and
   output hashes.
8. A statement of whether raw source text or generated data leaves the local
   environment. The proposed primary run uses an in-cluster endpoint.
9. Human-review rubric, reviewer provenance, and any confidentiality terms for
   review artifacts.
10. Training base-model terms and adapter artifact terms.

## Redistribution checklist

Before publishing code, images, datasets, reports, or adapters:

- include the applicable Curator license and NOTICE materials;
- mark modified upstream files clearly;
- preserve relevant copyright, patent, trademark, and attribution notices;
- audit bundled dependencies rather than assuming the top-level license covers
  them;
- confirm model-service and base-model redistribution terms;
- confirm corpus and generated-output rights;
- avoid using NVIDIA trademarks beyond reasonable identification of origin;
- obtain legal review if the evidence package will support a formal compliance
  representation.

## Required final evidence artifacts

```text
environment/
  curator-version.json
  container-digest.txt
  sbom.spdx.json
  packages.txt
  licenses/
  notices/
  model-and-service-terms.json
inputs/
  corpus-manifest.json
  corpus-rights-reference.json
manifest.json
report/
  license-and-provenance-summary.md
```

The evidence bundle should be append-only after a run is declared complete.
Corrections create a new signed or hashed manifest that references the prior
one rather than silently replacing it.
