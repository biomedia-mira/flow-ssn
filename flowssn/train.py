from typing import Optional, Dict

import os
import time
import wandb
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from flowssn.utils import *
from flowssn.factory import *
from flowssn.eval.metrics import *
from flowssn.models.continuous.model import ContinuousFlowSSN
from flowssn.models.autoregressive.model import AutoregressiveFlowSSN


@torch.no_grad()
def eval_batch(
    batch: Dict[str, torch.Tensor], probs: torch.Tensor
) -> Dict[str, torch.Tensor]:
    if batch["y"].ndim == 3:
        batch["y"] = batch["y"].unsqueeze(0)
    batch_size, _, _, num_classes = batch["y"].shape
    metrics = {
        name: torch.zeros(batch_size, device=probs.device)
        for name in ["hmiou", "dice", "energy_distance", "diversity"]
    }
    # (b, h, w, k)
    preds_oh = nn.functional.one_hot(probs.mean(0).argmax(dim=-1), num_classes).float()
    # (m, b, h, w, k)
    mc_preds_oh = nn.functional.one_hot(probs.argmax(dim=-1), num_classes).float()

    if num_classes == 2:  # LIDC/REFUGE
        assert "y_all" in batch.keys()
        # (n=num_raters, b, h, w)
        modes = batch["y_all"].permute(3, 0, 1, 2)
        # (n, b, h, w, k)
        modes_oh = nn.functional.one_hot(modes.long(), num_classes).float()
        # (b, h, w, k)
        fused = torch.sum(modes_oh, dim=0)
        fused_oh = nn.functional.one_hot(fused.argmax(dim=-1), num_classes).float()
        idx = [1]  # class id to eval, '1' is foreground
    else:
        raise NotImplementedError

    ged, div = energy_distance(mc_preds_oh, modes_oh, filter_bg=idx)
    metrics["energy_distance"], metrics["diversity"] = ged, div
    metrics["hmiou"] = hungarian_matched_iou(mc_preds_oh, modes_oh, filter_bg=idx)
    metrics["dice"] = dice_score(preds_oh, fused_oh, filter_bg=idx)
    return metrics


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    mininterval = float(os.environ["TQDM_MININTERVAL"])  # set in launch script
    loader = tqdm(dataloader, total=len(dataloader), mininterval=mininterval)
    keys = ["loss", "hmiou", "dice", "energy_distance", "diversity"]
    metrics = {k: 0.0 for k in keys}  # accumulated metrics
    counts = {k: 0.0 for k in keys}  # non-nan value counters

    for batch in loader:
        bs = batch["x"].shape[0]
        batch = model.preprocess_fn({"batch": batch})
        model.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            out = model(batch, mc_samples=model.mc_samples)
            loss = out["loss"]

        if training:
            loss.backward()
            stats = {}
            stats["grad_norm"] = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                stats["lr"] = scheduler.get_last_lr()[0]
            else:
                stats["lr"] = optimizer.param_groups[0]["lr"]
            stats["mc_std"] = out["std"]
            wandb.log(stats)
            if ema is not None:
                ema.update()
        else:
            with torch.no_grad():
                # uses micro batches to save memory when using many MC samples for eval
                micro_bs = min((4 * bs) // model.eval_samples, bs)
                micro_bs = 1 if micro_bs == 0 else micro_bs

                for i in range(0, bs, micro_bs):
                    micro_batch = {k: v[i : i + micro_bs] for k, v in batch.items()}
                    out = model({"x": micro_batch["x"]}, model.eval_samples)
                    probs = (
                        out["probs"]
                        if "probs" in out.keys()
                        else out["logits"].softmax(dim=-1)
                    )
                    batch_metrics = eval_batch(micro_batch, probs)  # type: ignore

                    for k, v in batch_metrics.items():
                        valid_vals = v[~torch.isnan(v)]
                        metrics[k] += valid_vals.sum().item()
                        counts[k] += valid_vals.numel()

        counts["loss"] += bs
        metrics["loss"] += loss.detach() * bs
        metrics_desc = ""
        for k, v in metrics.items():
            if counts[k] > 0:
                m = v / counts[k]
                metrics_desc += f", {k}: {m:.4f}" if m != 0 else ""
        loader.set_description(
            f"{'train' if training else 'valid'}"
            + metrics_desc
            + (f", std: {out['std']:.1e}" if training and out["std"] > 0 else "")
            + (f", gnorm: {stats['grad_norm']:.2f}" if training else "")
            + (f", lr: {stats['lr']:.1e}" if training else ""),
            refresh=False,
        )

    return {
        k: v / counts[k]
        for k, v in metrics.items()
        if (counts[k] > 0 and v / counts[k] != 0)  # filters out metrics not updated
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="")
    # DATA:
    parser.add_argument("--dataset", type=str, default="lidc")
    parser.add_argument(
        "--data_dir", type=str, default="./datasets/lidc/data_lidc.hdf5"
    )
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--cache", action="store_true", default=False)
    # TRAIN:
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup", type=int, default=1000)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--determ", action="store_true", default=False)
    # EVAL:
    parser.add_argument("--eval_freq", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=32)
    # MODEL:
    parser.add_argument(
        "--model", type=str, choices=["c-flowssn", "ar-flowssn"], default="c-flowssn"
    )
    nn_ch = ["transformer", "unet"]
    parser.add_argument("--net", type=str, choices=nn_ch, default="unet")
    parser.add_argument("--base_net", type=str, choices=nn_ch + [""], default="")
    args = parser.parse_known_args()[0]

    seed_all(args.seed, args.determ)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    resuming = os.path.isfile(args.resume)
    if resuming:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume)
        new_exp_name = args.exp_name
        args = argparse.Namespace(**ckpt["args"])
        args.exp_name = new_exp_name

    if args.dataset == "lidc":
        from flowssn.data.lidc import get_lidc as get_dataset
        from flowssn.data.lidc import preprocess_lidc_fn as preprocess_fn
    elif args.dataset == "refuge":
        from flowssn.data.refuge import get_refuge as get_dataset
        from flowssn.data.refuge import preprocess_refuge_fn as preprocess_fn
    else:
        raise NotImplementedError

    datasets = get_dataset(args)
    dataloaders = {
        k: DataLoader(
            datasets[k],
            batch_size=args.bs,
            shuffle=(k == "train"),
            drop_last=(k == "train"),
            num_workers=4,
            pin_memory=True,
        )
        for k in ["train", "valid", "test"]
    }

    if not resuming:
        args = parse_ssn_args(args.model, parser)
        args = parse_nn_args(args.net, parser)

    base_net = None
    if args.base_net != "" and args.cond_base:
        if resuming:
            base_net, _ = build_nn(args.base_net, args=args, prefix="base_")
        else:
            base_net, args = build_nn(args.base_net, parser, prefix="base_")

    if args.model == "c-flowssn":
        # NOTE: recommended model
        model = ContinuousFlowSSN(
            flow_net=build_nn(args.net, args=args)[0],
            base_net=base_net,
            num_classes=args.num_classes,
            cond_base=args.cond_base,
            cond_flow=args.cond_flow,
            base_std=args.base_std,
        )
        model.eval_T = args.eval_T  # set ODE solver steps for eval
        _flow, _base = model.flow_net, model.base_net

    elif args.model == "ar-flowssn":
        iaf_nets = nn.ModuleList()
        for _ in range(args.num_flows):
            iaf_nets.append(build_nn(args.net, args=args)[0])
        flow_nets = {"iaf": iaf_nets}

        model = AutoregressiveFlowSSN(
            flow_type=args.flow_type,
            flow_nets=flow_nets,
            base_nets=(None if base_net is None else nn.ModuleDict({"iaf": base_net})),
            cond_flow=args.cond_flow,
            cond_base=args.cond_base,
            base_std=args.base_std,
            num_classes=args.num_classes,
        )
        _flow, _base = model.flows, model.base_nets
    else:
        raise NotImplementedError

    model.to(device)
    model.preprocess_fn = LambdaModule(
        lambda kwargs: preprocess_fn(**kwargs, device=device)
    )
    model.mc_samples = args.mc_samples
    model.eval_samples = args.eval_samples

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / args.lr_warmup, total_iters=args.lr_warmup
    )

    if resuming:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt.keys():
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            scheduler = None

    ema = EMA(model.parameters(), rate=args.ema_rate) if args.ema_rate > 0 else None
    wandb.init(project="flow-ssn", name=args.exp_name, config=vars(args))

    print(model)
    for k, v in vars(args).items():
        print(f"--{k}={v}")
    print(f"#params: {count_params(model):,}")
    print(
        f"#flow params: {count_params(_flow):,} | #base params: {0 if _base is None else count_params(_base):,}"
    )

    save_path = f"../checkpoints/{args.exp_name}/"
    best_ged, best_metric = 1.0, 0.0
    track_metric = "dice"  # or, e.g., "hmiou"
    plots(model, dataloaders["valid"], save_path + f"0")

    start_t = time.time()

    for i in range(args.epochs):
        train_metrics = run_epoch(
            model, dataloaders["train"], ema, optimizer, scheduler
        )
        wandb.log({"train_" + k: v for k, v in train_metrics.items()})

        if i > 0 and (i % args.eval_freq) == 0:
            model.eval()
            if ema is not None:
                ema.apply()  # apply ema model weights
            valid_metrics = run_epoch(model, dataloaders["valid"])
            wandb.log({"valid_" + k: v for k, v in valid_metrics.items()})

            step = (i + 1) * len(dataloaders["train"])
            t = time.strftime("[%H:%M:%S, %d/%m/%Y]")
            elapsed_t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_t))
            print(
                f"{t} epoch: {i + 1}, step: {step}, elapsed: {elapsed_t}"
                f"\n{t} train {', '.join(f'{k}: {v:.5f}' for k, v in train_metrics.items())}"
                f"\n{t} valid {', '.join(f'{k}: {v:.5f}' for k, v in valid_metrics.items())}"
            )
            plots(model, dataloaders["valid"], save_path + f"{step}")

            save_dict = {
                "args": vars(args),
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scheduler is not None:
                save_dict["scheduler_state_dict"] = scheduler.state_dict()

            if valid_metrics[track_metric] > best_metric:
                best_metric = valid_metrics[track_metric]
                path = save_path + f"checkpoint_{track_metric}.pt"
                torch.save(save_dict | {f"{track_metric}": best_metric}, path)
                print(f"{t} model saved: {path}")

            if valid_metrics["energy_distance"] < best_ged:
                best_ged = valid_metrics["energy_distance"]
                path = save_path + "checkpoint_ged.pt"
                torch.save(save_dict | {"ged": best_ged}, path)
                print(f"{t} model saved: {path}")

            if ema is not None:
                ema.restore()  # restore model weights
    # run test
    t = time.strftime("[%H:%M:%S, %d/%m/%Y]")
    elapsed_t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_t))

    for ckpt_metric in [track_metric, "ged"]:
        load_path = save_path + f"checkpoint_{ckpt_metric}.pt"
        print(f"\n{t} Loading checkpoint: {load_path}")
        torch.cuda.empty_cache()
        ckpt = torch.load(load_path)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.eval_samples = 100  # type: ignore
        model.eval_T = 50  # type: ignore
        print(f"{t} total_steps: {ckpt['step']}, training_time: {elapsed_t}")
        test_metrics = run_epoch(model, dataloaders["test"])
        print(f"{t} test {', '.join(f'{k}: {v:.5f}' for k, v in test_metrics.items())}")
