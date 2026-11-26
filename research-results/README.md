# Cross-condition bearing fault diagnosis on HFZZ and PT700: experimental results

Numerical results accompanying *Acquisition-Record-Guided Optimal Transport Adaptation for Cross-Condition Bearing Fault Diagnosis*. The data in this directory are derived experiment outputs. Original vibration waveforms are held separately.

- **main_comparisons.csv** reproduces Tables 4 and 8 at the precision printed in the manuscript. Each transfer is written source to target. Mean rows weight all six directions equally. HFZZ neural models use three training seeds on one record split, while HFZZ SVM uses five whole-record partitions. PT700 uses three whole-record partitions. SD is the sample standard deviation in percentage points.
- **pt700_record_repeat_metrics.csv** contains per-repetition accuracy and macro-F1 for PT700. S, M and L denote increasing damage severity. The primary arm is geometric R-OT.
- **pt700_semantic_repeats_v1r1_metrics.csv** contains the separately matched PT700 comparison with and without the semantic cost.
- **hfzz_svm_repeats.json** contains the split seeds and counts of correct predictions and evaluated windows for HFZZ SVM.
- **hfzz_holdout_speed.json** gives selected weights and accuracy for each held-out HFZZ target speed.
- **hfzz_transport_sensitivity_expanded.csv** contains HFZZ transport parameter sweeps by task and seed.
