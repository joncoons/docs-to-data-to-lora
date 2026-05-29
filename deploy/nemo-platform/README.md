# NeMo Platform Kubernetes Deployment

This directory captures the version pin and values overlay for deploying the
NeMo Platform control plane that this showcase targets.

## NGC Version Check

Checked on 2026-05-29:

- NeMo Curator public NGC container: `nvcr.io/nvidia/nemo-curator:26.04`.
- Public NGC pages for individual legacy NeMo Microservices such as Evaluator
  and Guardrails still show latest tag `25.12`.
- The newer Kubernetes platform path is the NeMo Platform private NGC chart from
  the `26.3.1` documentation: `0857255566152269/external/nemo-platform:2.0.1`.
- The same install guide lists on-demand task images tagged `26.03.1` for
  Customizer/Data Designer/Evaluator execution paths.

Do not substitute unverified `26.04` tags for the public individual microservice
containers. Use the private NeMo Platform chart when targeting the current K8s
platform deployment, and pin Curator separately to the confirmed public `26.04`
container where this repository builds a Curator handoff image.

## Install Sketch

```bash
export NAMESPACE=nemo-peft

ngc registry chart pull \
  --org nvidian \
  '0857255566152269/external/nemo-platform:2.0.1'

kubectl apply \
  -f https://raw.githubusercontent.com/volcano-sh/volcano/v1.9.0/installer/volcano-development.yaml
kubectl wait \
  --for=condition=complete \
  job/volcano-admission-init \
  -n volcano-system \
  --timeout=120s
kubectl rollout status deployment/volcano-admission -n volcano-system

helm upgrade --install nemo-platform \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  nemo-platform-2.0.1.tar.gz \
  -f deploy/nemo-platform/values.yaml

kubectl apply -f deploy/nemo-platform/service-plane-configmap.yaml
```

## Service Plane ConfigMap

`service-plane-configmap.yaml` is the compatibility layer used by the
repository-owned Jobs. It defines `NMP_BASE_URL`, `NMP_WORKSPACE`, and
service-specific aliases such as `NMP_EVALUATOR_URL` and
`NMP_INFERENCE_GATEWAY_URL`. The default `NMP_BASE_URL` assumes a Helm release
called `nemo-platform`, which normally yields `nemo-platform-api:8080`; verify
with `kubectl get svc -n nemo-peft` if the release name changes.

`NMP_DATASTORE_GIT_BASE` intentionally remains a direct Git/Data Store endpoint
for compatibility paths that still clone or push HF-style repositories. The
Customizer training path now has a Platform FileSet handoff in
`deploy/platform-filesets/`; TIES clone mode is the remaining direct Git/Data
Store consumer.

## Task Images

The NeMo Platform install guide calls out these private-registry task images for
pre-pull or mirroring when the runtime cannot pull task images on demand:

```text
nvcr.io/0857255566152269/external/customizer-automodel:26.03.1
nvcr.io/0857255566152269/external/customizer-rl:26.03.1
nvcr.io/0857255566152269/external/customizer-tasks:26.03.1
nvcr.io/0857255566152269/external/nmp-cpu-tasks:26.03.1
nvcr.io/0857255566152269/external/nmp-gpu-tasks:26.03.1
```
