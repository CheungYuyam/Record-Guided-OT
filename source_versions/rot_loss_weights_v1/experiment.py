"""One prespecified auxiliary-loss sensitivity run on an existing record split."""
from pathlib import Path
import argparse
import itertools
import json
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


class Direct(torch.nn.Module):
    def __init__(self, representation):
        super().__init__()
        self.representation = representation

    def forward(self, inputs):
        return self.representation.classifier(self.representation.encode(inputs))


def verify_predictions(path, report, classes):
    with np.load(path) as saved:
        labels, records = saved["y"], saved["record_id"]
        ids, inverse = np.unique(records, return_inverse=True)
        record_labels = np.array([labels[records == item][0] for item in ids])
        for condition in report:
            if condition == "mean_accuracy":
                continue
            probabilities = saved[condition]
            assert np.isfinite(probabilities).all()
            assert np.allclose(probabilities.sum(1), 1, atol=1e-6)
            record_probabilities = np.zeros((len(ids), classes), dtype=np.float64)
            np.add.at(record_probabilities, inverse, probabilities)
            record_probabilities /= np.bincount(inverse)[:, None]
            for unit, truth, probability in [("window", labels, probabilities),
                                             ("record", record_labels, record_probabilities)]:
                predictions = probability.argmax(1)
                confusion = np.zeros((classes, classes), dtype=int)
                np.add.at(confusion, (truth, predictions), 1)
                accuracy = float((predictions == truth).mean())
                f1 = float(np.mean(2 * confusion.diagonal() /
                                  np.maximum(confusion.sum(0) + confusion.sum(1), 1)))
                assert abs(accuracy - report[condition][unit]["accuracy"]) < 1e-12
                assert abs(f1 - report[condition][unit]["macro_f1"]) < 1e-12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = read(HERE / "protocol.json")
    task = protocol["tasks"][args.index]
    dataset, pair, seed = task["dataset"], task["pair"], task["seed"]
    sys.path.insert(0, str(HERE / dataset))
    import models
    import core
    import run as base
    import refinement as ref
    from bridges import fit_temperature

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    suffix = "_check" if args.check_only else ""
    out = ROOT / f"results/rot_loss_weights_v1_{args.index:03d}{suffix}"
    if (out / "DONE.json").exists():
        assert read(out / "run.json")["task"] == task
        assert read(out / "DONE.json")["success"]
        print("Already completed", out, flush=True)
        return
    out.mkdir(exist_ok=False)
    data_root = "foliation_v12" if dataset == "hfzz" else "pt700_v1"
    data = ROOT / "data" / data_root / pair
    manifest = read(data / "manifest.json")
    splits = [set(item["record_ids"]) for item in manifest["splits"].values()]
    assert all(a.isdisjoint(b) for a, b in itertools.combinations(splits, 2))
    loaded = {}
    for name in ["source_train", "source_val", "target_unlabeled"]:
        path = data / (name + ".npz")
        assert base.sha(path) == manifest["splits"][name]["sha256"]
        loaded[name] = base.load(path, "cuda")
        assert set(loaded[name]["record_id"].tolist()) == set(manifest["splits"][name]["record_ids"])
    source, validation, adaptation = [loaded[name] for name in
                                         ["source_train", "source_val", "target_unlabeled"]]
    assert "y" not in adaptation
    raw = base.features(source["x"])
    mean = raw.mean((0, 2), keepdim=True)
    std = raw.std((0, 2), keepdim=True).clamp_min(1e-6)
    xs, xt = (raw - mean) / std, (base.features(adaptation["x"]) - mean) / std
    labels = source["y"]
    teacher = base.train_classifier("source_only", xs, labels, xt, None,
                                    argparse.Namespace(seed=seed, classifier_steps=2400),
                                    out, mean, std)
    teacher_path = out / "source_only_last.pt"
    teacher_hash = base.sha(teacher_path)
    classes = 9 if dataset == "hfzz" else 4
    assert set(labels.tolist()) == set(range(classes))
    training = argparse.Namespace(seed=seed, cycle_weight=task["cycle"], contrast_weight=task["contrast"],
                                  representation_steps=3 if args.check_only else 3000,
                                  refinement_steps=3 if args.check_only else 1500,
                                  refresh_interval=1 if args.check_only else 300)
    source_hashes = {p.relative_to(HERE).as_posix(): base.sha(p) for p in HERE.rglob("*.py")}
    base.dump(out / "run.json", {"task": task, "manifest": manifest, "args": vars(training),
              "check_only": args.check_only, "protocol_sha256": base.sha(HERE / "protocol.json"),
              "source_hashes": source_hashes, "teacher_path": str(teacher_path),
              "teacher_sha256": teacher_hash, "target_labels_used_in_training": False,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    started = time.perf_counter()
    initial = models.train_representation(xs, labels, xt, training, out)
    with torch.no_grad():
        latent = torch.cat([models.encode(initial, xs), models.encode(initial, xt)])
        lm, ls = latent.mean(0), latent.std(0).clamp_min(.05)
    torch.save({"state_dict": initial.state_dict(), "step": training.representation_steps,
                "latent_mean": lm, "latent_std": ls}, out / "representation_last.pt")
    vs = manifest["source_rpm"] if dataset == "hfzz" else "SML".index(manifest["source_domain"])
    validation_inputs = base.arrays(validation, mean, std, 12345 + vs)
    with torch.no_grad():
        temperature, _ = fit_temperature(torch.cat([teacher(x) for x in validation_inputs["clean"].split(256)]), validation["y"])
        teacher_probabilities = (torch.cat([teacher(x) for x in xt.split(256)]) / temperature).softmax(-1)
    solver = ref.sinkhorn
    ref.sinkhorn = lambda cost: solver(cost, epsilon=.1, max_steps=20000)
    dummy = torch.full((len(xs), len(xt)), 1 / (len(xs) * len(xt)), device="cuda")
    final, _ = ref.refine(task["arm"], initial, dummy, 1., xs, labels, xt,
                         validation_inputs["clean"], validation["y"], teacher_probabilities,
                         lm, ls, training, out, source["record_id"], adaptation["record_id"])
    torch.cuda.synchronize()
    base.dump(out / "TRAINING_COMPLETE.json", {"target_test_read": False,
              "seconds": time.perf_counter() - started, "initial_updates": training.representation_steps,
              "refinement_updates": training.refinement_steps})
    reports = {}
    for name, model in [("initial", initial), ("final", final)]:
        path = out / f"{name}_source_val_predictions.npz"
        report = base.evaluate(Direct(model), validation_inputs, validation, path)
        verify_predictions(path, report, classes)
        reports[name] = {"source_validation": report}
    if not args.check_only:
        path = data / "target_test.npz"
        assert base.sha(path) == manifest["splits"]["target_test"]["sha256"]
        target = base.load(path, "cuda")
        assert set(target["record_id"].tolist()) == set(manifest["splits"]["target_test"]["record_ids"])
        ts = manifest["target_rpm"] if dataset == "hfzz" else "SML".index(manifest["target_domain"])
        target_inputs = base.arrays(target, mean, std, 12345 + ts)
        for name, filename in [("initial", "representation_last.pt"),
                               ("final", task["arm"] + "_representation_last.pt")]:
            model = models.Representation().cuda()
            model.load_state_dict(torch.load(out / filename, map_location="cuda", weights_only=False)["state_dict"])
            model.eval()
            prediction_path = out / f"{name}_target_predictions.npz"
            report = base.evaluate(Direct(model), target_inputs, target, prediction_path)
            verify_predictions(prediction_path, report, classes)
            reports[name]["target"] = report
    assert base.sha(teacher_path) == teacher_hash
    assert all(base.sha(HERE / path) == digest for path, digest in source_hashes.items())
    base.dump(out / "summary.json", reports)
    base.dump(out / "DONE.json", {"success": True, "check_only": args.check_only,
              "task": task, "target_evaluated": not args.check_only,
              "predictions_independently_verified": True, "final_model_reloaded": True})
    print("DONE", str(out), flush=True)


if __name__ == "__main__":
    main()
