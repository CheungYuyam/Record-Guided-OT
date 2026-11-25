# Record-guided optimal transport for bearing fault diagnosis

Companion implementation for *Acquisition-Record-Guided Optimal Transport Adaptation for Cross-Condition Bearing Fault Diagnosis*. The cross-speed HFZZ experiments use R-OT+S. The PT700 cross-severity experiments use geometric R-OT. The acquisition recordings must be obtained separately.

## Environment and inputs

The neural experiments used Linux, Python 3.10, CUDA-capable PyTorch 2.5.1, NumPy 1.26, SciPy, scikit-learn 1.5.2 and threadpoolctl. Data preparation uses NumPy and SciPy. From the repository root, create a results directory and prepare record-disjoint inputs from your local copies of the original HFZZ files and PT700 archive.

```sh
mkdir -p results
python preprocessing/prepare_inputs.py hfzz --raw /path/to/HFZZ --pool data/n3 --out data/foliation_v12
python preprocessing/prepare_inputs.py pt700 --archive /path/to/Archive.zip --out data/pt700_v1
```

## Training and evaluation

```sh
python source_versions/rot_loss_weights_v1/experiment.py --index 0
python source_versions/classic_baselines_v1_r1/experiment.py --index 18
python source_versions/pt700_record_repeats_v1/experiment.py --prepare
python source_versions/pt700_record_repeats_v1/experiment.py --index 0
```

For HFZZ, indices 0 through 17 select the six tasks and three training seeds in the order in the HFZZ protocol file. The public entry point trains the source teacher from the prepared spectra in place of loading the archived teacher checkpoint. Classical HFZZ comparisons use indices 18 through 23. On PT700, prepare the three partitions once, then run indices 0 through 17 (six directions per partition). Each PT700 run includes linear SVM, CORAL+SVM and the neural models. Target test labels are used to score completed models. The paper describes the development splits and parameter selection separately.
