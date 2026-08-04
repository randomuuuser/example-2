# pipeline.py

```python
"""
High-level pipeline for dataset preparation, transcription, feature extraction,
model evaluation, and deployment.
"""
```

### `report(title, pairs)`
```python
"""
Display a formatted summary.

Args:
    title: Report title.
    pairs: Key-value pairs to display.

Returns:
    None.
"""
```

### `load(out_dir)`
```python
"""
Load the prepared dataset.

Args:
    out_dir: Output directory.

Returns:
    Prepared records.
"""
```

### `prepare(...)`
```python
"""
Prepare the dataset for ASR transcription.

Args:
    json_path: Dataset annotation file.
    lang: Language code.
    out_dir: Output directory.
    audio_root: Root audio directory.
    min_ref_words: Minimum reference length.
    roles: Speaker roles to include.
    strip_accents: Whether to remove diacritics.
    expand_numbers: Whether to expand numbers into words.

Returns:
    Prepared records.
"""
```

### `sanity(records, model="nvidia/canary-180m-flash", n_per_lang=20)`
```python
"""
Run a sanity check on the ASR model.

Args:
    records: Prepared records.
    model: ASR model.
    n_per_lang: Number of samples per language.

Returns:
    Sample transcriptions.
"""
```

### `transcribe(records, out_dir, systems=None)`
```python
"""
Run ASR transcription.

Args:
    records: Prepared records.
    out_dir: Output directory.
    systems: ASR systems to execute.

Returns:
    Transcription results.
"""
```

### `accents(records, out_dir, min_delta=0.005)`
```python
"""
Evaluate the impact of accent normalization.

Args:
    records: Prepared records.
    out_dir: Output directory.
    min_delta: Minimum WER improvement threshold.

Returns:
    Accent normalization recommendations.
"""
```

### `merge(records, out_dir)`
```python
"""
Merge transcriptions and generate WER labels.

Args:
    records: Prepared records.
    out_dir: Output directory.

Returns:
    Merged dataset.
"""
```

### `status(out_dir)`
```python
"""
Summarize the current pipeline status.

Args:
    out_dir: Output directory.

Returns:
    Status information.
"""
```

### `to_frame(rows)`
```python
"""
Convert records into a pandas DataFrame.

Args:
    rows: Input records.

Returns:
    DataFrame.
"""
```

### `load_table(out_dir)`
```python
"""
Load the merged analysis table.

Args:
    out_dir: Output directory.

Returns:
    Merged records.
"""
```

### `ablate(rows, target="wer", n_splits=5)`
```python
"""
Run feature ablation experiments.

Args:
    rows: Input records.
    target: Prediction target.
    n_splits: Number of cross-validation folds.

Returns:
    Ablation results.
"""
```

### `evaluate(rows, target="wer", model="hgb", blocks=("proxy", "text"), n_splits=5, importance=True)`
```python
"""
Evaluate a WER prediction model.

Args:
    rows: Input records.
    target: Prediction target.
    model: Regression model.
    blocks: Feature blocks to include.
    n_splits: Number of cross-validation folds.
    importance: Whether to compute feature importance.

Returns:
    Evaluation results.
"""
```

### `compare_proxies(rows, target="wer", n_splits=5)`
```python
"""
Compare different proxy ASR configurations.

Args:
    rows: Input records.
    target: Prediction target.
    n_splits: Number of cross-validation folds.

Returns:
    Proxy comparison results.
"""
```

### `diagnose(rows, min_hyp_words=3)`
```python
"""
Run diagnostic analyses on the dataset.

Args:
    rows: Input records.
    min_hyp_words: Minimum hypothesis length.

Returns:
    None.
"""
```

### `worst(rows, n=10)`
```python
"""
Display the worst prediction examples.

Args:
    rows: Input records.
    n: Number of examples.

Returns:
    Prediction analysis.
"""
```

### `tail(rows)`
```python
"""
Analyze the distribution of WER labels.

Args:
    rows: Input records.

Returns:
    Label distribution statistics.
"""
```

### `clipped(rows, max_wer=1.0)`
```python
"""
Clip WER labels above a threshold.

Args:
    rows: Input records.
    max_wer: Maximum WER value.

Returns:
    Updated records.
"""
```

### `fit(rows, out_dir=None, target="wer", model="hgb", blocks=("proxy", "text"), notes="")`
```python
"""
Train the final WER prediction model.

Args:
    rows: Training records.
    out_dir: Output directory.
    target: Prediction target.
    model: Regression model.
    blocks: Feature blocks to include.
    notes: Additional information.

Returns:
    Trained model bundle.
"""
```

### `rank(bundle, rows)`
```python
"""
Rank calls by their estimated WER.

Args:
    bundle: Trained model bundle.
    rows: Input records.

Returns:
    Ranked calls.
"""
```

### `plot_ridge(table, n=12, path=None)`
```python
"""
Plot Ridge regression coefficients.

Args:
    table: Ridge coefficients.
    n: Number of features to display.
    path: Output file path.

Returns:
    Matplotlib figure.
"""
```

### `plot_pdp(curve, feature, path=None)`
```python
"""
Plot a partial dependence curve.

Args:
    curve: Partial dependence values.
    feature: Feature name.
    path: Output file path.

Returns:
    Matplotlib figure.
"""
```

### `plot_importance_comparison(permutation_scores, xgb_gains, n=10, path=None)`
```python
"""
Compare permutation and XGBoost feature importance.

Args:
    permutation_scores: Permutation importance scores.
    xgb_gains: XGBoost importance scores.
    n: Number of features to display.
    path: Output file path.

Returns:
    Matplotlib figure.
"""
```






# text_normalization.py

```python
"""
Utilities for text normalization and WER computation.

Provides a consistent normalization pipeline for reference and hypothesis
transcripts, along with helper functions for WER statistics.
"""
```

### `_strip_accents(text, lang=None)`
```python
"""
Normalize accented characters according to the target language.

Args:
    text: Input text.
    lang: Language code.

Returns:
    Normalized text.
"""
```

### `_parse_number(raw)`
```python
"""
Parse a numeric string into an integer or float.

Args:
    raw: Numeric string.

Returns:
    Parsed numeric value, or None if parsing fails.
"""
```

### `_expand_numbers(text, lang)`
```python
"""
Convert numeric expressions into their written form.

Args:
    text: Input text.
    lang: Language code.

Returns:
    Text with expanded numbers.
"""
```

### `normalize_text(text, lang, strip_accents=False, expand_numbers=True)`
```python
"""
Normalize a transcript before WER computation.

Args:
    text: Input transcript.
    lang: Language code.
    strip_accents: Whether to remove diacritics.
    expand_numbers: Whether to expand numbers into words.

Returns:
    Normalized transcript.
"""
```

### `normalize_all(records, lang_key="lang", text_keys=(), **norm_kwargs)`
```python
"""
Normalize multiple text fields within a collection of records.

Args:
    records: List of records.
    lang_key: Language field.
    text_keys: Text fields to normalize.
    **norm_kwargs: Additional normalization options.

Returns:
    Updated records.
"""
```

### `wer_counts(reference, hypothesis)`
```python
"""
Compute edit statistics and Word Error Rate.

Args:
    reference: Normalized reference transcript.
    hypothesis: Normalized hypothesis transcript.

Returns:
    Dictionary containing edit counts and WER.
"""
```

### `diagnose_accent_impact(pairs, lang)`
```python
"""
Compare WER with and without accent normalization.

Args:
    pairs: List of (reference, hypothesis) pairs.
    lang: Language code.

Returns:
    WER for each normalization strategy.
"""
```

---

# merge.py

```python
"""
Merge normalized transcriptions and generate the final dataset used for
feature extraction and model training.
"""
```

### `merge_transcriptions(records, results, norm_config)`
```python
"""
Merge ASR hypotheses with the reference records.

Args:
    records: Reference records.
    results: Transcription results by system.
    norm_config: Text normalization settings.

Returns:
    Tuple containing merged records and transcription coverage.
"""
```

### `add_labels(rows, target_role=TARGET_ROLE)`
```python
"""
Compute WER labels for the target ASR system.

Args:
    rows: Merged records.
    target_role: Target ASR system.

Returns:
    Labeled records.
"""
```

### `corpus_summary(rows, target_role=TARGET_ROLE)`
```python
"""
Compute corpus-level WER statistics.

Args:
    rows: Labeled records.
    target_role: Target ASR system.

Returns:
    Summary statistics for the corpus.
"""
```

### `build_table(records, results, out_dir, target_role=TARGET_ROLE)`
```python
"""
Build and save the final analysis table.

Args:
    records: Reference records.
    results: Transcription results.
    out_dir: Output directory.
    target_role: Target ASR system.

Returns:
    Analysis table and processing metadata.
"""
```

### `read_table(out_dir)`
```python
"""
Load the analysis table.

Args:
    out_dir: Output directory.

Returns:
    List of table records.
"""
```

---

# prepare_dataset.py

```python
"""
Prepare the evaluation dataset by extracting speech segments, normalizing
references, and generating the files required for ASR inference.
"""
```

### `write_norm_config(out_dir, norm_config)`
```python
"""
Save the normalization configuration.

Args:
    out_dir: Output directory.
    norm_config: Normalization settings.

Returns:
    Path to the saved configuration.
"""
```

### `read_norm_config(out_dir)`
```python
"""
Load the normalization configuration.

Args:
    out_dir: Output directory.

Returns:
    Normalization settings.
"""
```

### `probe_channels(path)`
```python
"""
Retrieve the number of audio channels.

Args:
    path: Audio file path.

Returns:
    Number of channels.
"""
```

### `load_call(path, sr=TARGET_SR)`
```python
"""
Load an audio recording.

Args:
    path: Audio file path.
    sr: Target sampling rate.

Returns:
    Audio samples.
"""
```

### `write_wav_mono(path, samples, sr=TARGET_SR)`
```python
"""
Save audio as a mono WAV file.

Args:
    path: Output file path.
    samples: Audio samples.
    sr: Sampling rate.

Returns:
    None.
"""
```

### `resolve_channel(segment, sample, n_channels)`
```python
"""
Resolve the channel associated with a speech segment.

Args:
    segment: Segment metadata.
    sample: Recording metadata.
    n_channels: Number of audio channels.

Returns:
    Channel index or None.
"""
```

### `extract_segment(call_audio, sr, start_time, end_time, channel_index, pad=0.0)`
```python
"""
Extract a speech segment from a recording.

Args:
    call_audio: Audio samples.
    sr: Sampling rate.
    start_time: Segment start time.
    end_time: Segment end time.
    channel_index: Selected channel.
    pad: Padding duration.

Returns:
    Extracted audio samples.
"""
```

### `read_segments(...)`
```python
"""
Parse the dataset annotations into segment records.

Args:
    json_path: Dataset annotation file.
    lang: Language code.
    min_ref_words: Minimum reference length.
    roles: Speaker roles to keep.
    source: Annotation source.
    norm_config: Text normalization settings.

Returns:
    Records and parsing statistics.
"""
```

### `export_segments(records, out_dir, audio_root="")`
```python
"""
Export speech segments as WAV files.

Args:
    records: Segment records.
    out_dir: Output directory.
    audio_root: Root audio directory.

Returns:
    Exported records and export failures.
"""
```

### `write_nemo_manifest(records, manifest_path, task="asr", pnc="yes")`
```python
"""
Generate a NeMo ASR manifest.

Args:
    records: Segment records.
    manifest_path: Output manifest path.
    task: Task type.
    pnc: Punctuation setting.

Returns:
    Manifest path.
"""
```

### `write_records(records, path)`
```python
"""
Save segment records.

Args:
    records: Segment records.
    path: Output file path.

Returns:
    Output file path.
"""
```

### `read_records(path)`
```python
"""
Load segment records.

Args:
    path: Input file path.

Returns:
    Segment records.
"""
```

### `prepare_corpus(...)`
```python
"""
Prepare the complete dataset for transcription.

Args:
    json_path: Dataset annotation file.
    lang: Language code.
    out_dir: Output directory.
    audio_root: Root audio directory.
    min_ref_words: Minimum reference length.
    roles: Speaker roles to keep.
    norm_config: Text normalization settings.

Returns:
    Prepared records, processing statistics, and export failures.
"""
```

---

# transcribe.py

```python
"""
Run ASR systems on the prepared speech segments and store their transcriptions
for downstream WER estimation.
"""
```

### `load_cache(path)`
```python
"""
Load a transcription cache.

Args:
    path: Cache file path.

Returns:
    Cached transcriptions.
"""
```

### `append_cache(path, rows)`
```python
"""
Append transcriptions to a cache.

Args:
    path: Cache file path.
    rows: Transcription records.

Returns:
    None.
"""
```

### `pending(records, cache)`
```python
"""
Retrieve records that have not yet been transcribed.

Args:
    records: Segment records.
    cache: Existing cache.

Returns:
    Pending records.
"""
```

### `release_model(model)`
```python
"""
Release model resources.

Args:
    model: Loaded ASR model.

Returns:
    None.
"""
```

### `load_canary(model_name="nvidia/canary-180m-flash", beam_size=1)`
```python
"""
Load a Canary ASR model.

Args:
    model_name: Model identifier.
    beam_size: Decoding beam size.

Returns:
    Loaded model.
"""
```

### `hypothesis_text(item)`
```python
"""
Extract the transcription text from a model output.

Args:
    item: Model output.

Returns:
    Transcription text.
"""
```

### `run_canary(...)`
```python
"""
Transcribe speech segments with a Canary model.

Args:
    records: Segment records.
    cache_path: Cache file path.
    model_name: Model identifier.
    batch_size: Batch size.
    chunk: Processing chunk size.

Returns:
    Updated transcription cache.
"""
```

### `run_whisper(...)`
```python
"""
Transcribe speech segments with a Whisper model.

Args:
    records: Segment records.
    cache_path: Cache file path.
    model_name: Model identifier.
    batch_size: Batch size.

Returns:
    Updated transcription cache.
"""
```

### `sanity_check_canary(...)`
```python
"""
Run a quick transcription check on a subset of segments.

Args:
    records: Segment records.
    model_name: Model identifier.
    n_per_lang: Number of segments per language.

Returns:
    Sample transcription cache.
"""
```

### `run_all(records, out_dir, systems=None)`
```python
"""
Run all configured ASR systems.

Args:
    records: Segment records.
    out_dir: Output directory.
    systems: ASR systems to execute.

Returns:
    Transcriptions grouped by system.
"""
```








# features.py

```python
"""
Extract feature vectors from normalized ASR hypotheses for WER prediction.
"""
```

### `_char_error_rate(reference, hypothesis)`
```python
"""
Compute the Character Error Rate (CER) between two normalized transcripts.

Args:
    reference: Reference transcript.
    hypothesis: Hypothesis transcript.

Returns:
    Character Error Rate.
"""
```

### `proxy_features(row, roles=PROXY_ROLES)`
```python
"""
Compute agreement-based features between the target and proxy hypotheses.

Args:
    row: Input record.
    roles: Proxy ASR systems.

Returns:
    Dictionary of proxy-based features.
"""
```

### `text_features(row)`
```python
"""
Compute text-based features from the target hypothesis.

Args:
    row: Input record.

Returns:
    Dictionary of text-based features.
"""
```

### `build_features(rows, blocks=("proxy", "text"), roles=PROXY_ROLES)`
```python
"""
Build the feature matrix and target vectors.

Args:
    rows: Input records.
    blocks: Feature blocks to include.
    roles: Proxy ASR systems.

Returns:
    Feature matrix, target vectors, groups, and feature names.
"""
```

### `block_of(name)`
```python
"""
Identify the feature block associated with a feature.

Args:
    name: Feature name.

Returns:
    Feature block name.
"""
```

---

# evaluate.py

```python
"""
Train and evaluate WER prediction models using grouped cross-validation and
feature ablation.
"""
```

### `make_model(name="hgb", seed=0)`
```python
"""
Create a regression model.

Args:
    name: Model name.
    seed: Random seed.

Returns:
    Configured regression model.
"""
```

### `select_blocks(X, names, blocks)`
```python
"""
Select features belonging to the requested blocks.

Args:
    X: Feature matrix.
    names: Feature names.
    blocks: Feature blocks to retain.

Returns:
    Filtered feature matrix and selected feature names.
"""
```

### `splitter(groups, n_splits=5)`
```python
"""
Create a grouped cross-validation splitter.

Args:
    groups: Group labels.
    n_splits: Number of folds.

Returns:
    Cross-validation splitter.
"""
```

### `cross_validate(...)`
```python
"""
Perform grouped cross-validation.

Args:
    rows: Input records.
    blocks: Feature blocks to include.
    target: Prediction target.
    model: Regression model.
    n_splits: Number of folds.
    seed: Random seed.
    roles: Proxy ASR systems.

Returns:
    Cross-validation predictions and evaluation data.
"""
```

### `segment_metrics(result)`
```python
"""
Compute segment-level evaluation metrics.

Args:
    result: Cross-validation results.

Returns:
    Dictionary of evaluation metrics.
"""
```

### `call_metrics(result)`
```python
"""
Compute call-level evaluation metrics.

Args:
    result: Cross-validation results.

Returns:
    Call-level metrics and summary statistics.
"""
```

### `denominator_bias(rows)`
```python
"""
Measure the difference between reference and hypothesis lengths.

Args:
    rows: Input records.

Returns:
    Length ratio statistics.
"""
```

### `ablation(rows, target="errors", n_splits=5, seed=0, plan=None)`
```python
"""
Evaluate different feature configurations.

Args:
    rows: Input records.
    target: Prediction target.
    n_splits: Number of folds.
    seed: Random seed.
    plan: Ablation configurations.

Returns:
    Ablation results.
"""
```

### `single_proxy_ablation(rows, target="errors", n_splits=5)`
```python
"""
Evaluate the contribution of each proxy ASR system.

Args:
    rows: Input records.
    target: Prediction target.
    n_splits: Number of folds.

Returns:
    Evaluation results for each proxy configuration.
"""
```

### `permutation_importance(...)`
```python
"""
Estimate feature importance using permutation.

Args:
    rows: Input records.
    blocks: Feature blocks to include.
    target: Prediction target.
    n_splits: Number of folds.
    n_repeats: Number of permutations.
    seed: Random seed.

Returns:
    Baseline score and feature importance values.
"""
```

### `_matrix(rows, blocks, target, roles=PROXY_ROLES)`
```python
"""
Build the feature matrix used for model analysis.

Args:
    rows: Input records.
    blocks: Feature blocks to include.
    target: Prediction target.
    roles: Proxy ASR systems.

Returns:
    Feature matrix, target vector, groups, and feature names.
"""
```

### `ridge_coefficients(rows, blocks=("proxy", "text"), target="wer", n_splits=5)`
```python
"""
Estimate Ridge regression coefficients across folds.

Args:
    rows: Input records.
    blocks: Feature blocks to include.
    target: Prediction target.
    n_splits: Number of folds.

Returns:
    Averaged feature coefficients.
"""
```

### `partial_dependence_curve(...)`
```python
"""
Compute the partial dependence of a feature.

Args:
    rows: Input records.
    feature: Feature name.
    blocks: Feature blocks to include.
    target: Prediction target.
    n_points: Number of evaluation points.
    seed: Random seed.

Returns:
    Partial dependence values.
"""
```

### `xgb_importances(rows, blocks=("proxy", "text"), target="wer", seed=0)`
```python
"""
Compute feature importance using XGBoost.

Args:
    rows: Input records.
    blocks: Feature blocks to include.
    target: Prediction target.
    seed: Random seed.

Returns:
    Feature importance scores.
"""
```

---

# deploy.py

```python
"""
Train, save, load, and deploy a WER prediction model for inference.
"""
```

### `features_for_inference(rows, blocks=("proxy", "text"), roles=PROXY_ROLES)`
```python
"""
Build inference features without reference labels.

Args:
    rows: Input records.
    blocks: Feature blocks to include.
    roles: Proxy ASR systems.

Returns:
    Feature dictionaries.
"""
```

### `entries_to_matrix(entries, feature_names)`
```python
"""
Convert feature dictionaries into a feature matrix.

Args:
    entries: Feature dictionaries.
    feature_names: Ordered feature names.

Returns:
    Feature matrix.
"""
```

### `fit_final(...)`
```python
"""
Train the final prediction model.

Args:
    rows: Training records.
    blocks: Feature blocks to include.
    target: Prediction target.
    model: Regression model.
    roles: Proxy ASR systems.
    norm_config: Normalization settings.
    seed: Random seed.
    notes: Additional information.

Returns:
    Trained model bundle.
"""
```

### `save_model(bundle, path)`
```python
"""
Save a trained model bundle.

Args:
    bundle: Model bundle.
    path: Output file path.

Returns:
    Saved file path.
"""
```

### `load_model(path)`
```python
"""
Load a saved model bundle.

Args:
    path: Model file path.

Returns:
    Loaded model bundle.
"""
```

### `predict(bundle, rows, norm_config=None)`
```python
"""
Predict the WER for individual speech segments.

Args:
    bundle: Trained model bundle.
    rows: Input records.
    norm_config: Normalization settings.

Returns:
    Estimated WER for each segment.
"""
```

### `predict_calls(bundle, rows, norm_config=None)`
```python
"""
Predict the WER at the call level.

Args:
    bundle: Trained model bundle.
    rows: Input records.
    norm_config: Normalization settings.

Returns:
    Estimated WER for each call.
"""
```
