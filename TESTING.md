## Tests

```sh
python -m pip install -r requirements-test.txt
python -m pytest
```

These initial tests cover selected behavior with external services mocked. They do not establish full integration coverage.

Coverage: training CLI defaults, option conversion, and invalid arguments. Model training, image preprocessing, GPU execution, and prediction accuracy are not exercised.
