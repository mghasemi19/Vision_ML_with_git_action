# Lesson 5: what remains before a real production launch?

This is a complete runnable teaching reference, not a fully operated production service. Real-world additions:

1. **Data lineage:** signed/versioned dataset manifests and snapshots; DVC or object storage with immutable IDs.
2. **Model registry:** MLflow or similar staging/production model lifecycle and an immutable approved incumbent for regression checks.
3. **Statistical rigor:** representative untouched final test; bootstrap confidence intervals, slice/fairness checks, calibrated probabilities, explicit acceptance policy.
4. **Security:** dependency and container scanning, action SHA pinning, SBOM, signed images, secrets via vault/OIDC, production IAM; model artifacts trusted from a controlled runner.
5. **Operations:** metrics, structured logs, tracing, SLOs, traffic-shift deployment, rollbacks, canary validation, alerts, ownership.
6. **Scale:** expensive training on scheduled GPU-capable infrastructure and lightweight PR checks; artifact lineage connecting code/data/model/config versions.
7. **Robustness:** photo uploads, OCR preprocessing, adversarial/corrupted image tests, versioned input contracts. The current API accepts 8x8 numeric matrices, not arbitrary images.
8. **GitHub governance:** branch protection and required status checks, protected release environments, approval policies, least-privilege tokens.

For an LLM or CV deep network, PR checks should validate shapes, loading, a miniature forward pass, deterministic preprocessing, and critical component contracts. Separate model evaluation and promotion from ordinary commits.
