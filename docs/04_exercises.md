# Lesson 4: guided failure experiments

Make changes in a **temporary branch** and restore them after each exercise.

1. **Bad pixel values:** modify `load_dataset()` to set `images[0,0,0] = 99` before validation. Run `python -m pytest -q`. Expect data validation failure.
2. **Data leakage:** try to fit the scaler on all images before splitting. The existing code structure prevents it by putting the scaler inside `Pipeline`; explain why this is essential. Add a test to guard the training API against pre-fitted scalers.
3. **Bad model:** in `new_candidate()`, replace the logistic-regression classifier with `DummyClassifier(strategy="most_frequent")`. Run `python -m vision_mlops.pipeline --output-dir models`. Expect quality-gate failure and no new model artifact in a **fresh output directory**.
4. **Quality threshold too high:** temporarily set `MIN_MACRO_F1=1.0`. See the evaluation report list which gate fails; restore the threshold.
5. **Bad request:** send `{"pixels": [[0]*8]*7}` from Python (or Swagger). Expect HTTP 422. Send a correct 8×8 matrix and expect a prediction.
6. **Missing artifact:** rename `models/model.joblib`, restart the API, request `/health/ready`. Expect HTTP 503, while `/health/live` returns 200.
7. **GitHub pipeline:** push a branch, open a PR targeting `main`; inspect `checks`, its uploaded metrics and Docker smoke job. Fix an intentionally broken unit test.
8. **Release review:** configure the `release` environment and run `publish.yml` manually from `main` if GHCR publication is desired; confirm an image with a commit-SHA tag exists. Do not deploy blindly.

## Suggested discussion

- Does a perfect training accuracy prove generalization?
- What is the difference between a schema error and data drift?
- Why is the majority dummy model not a production incumbent?
- Why should the test set not be used repeatedly to tune hyperparameters?
