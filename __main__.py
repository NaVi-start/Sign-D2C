import argparse
import os

from stage1_models.stage1_training import train_stage1, test_stage1
from stage2_models.stage2_training import train_stage2, test_stage2

def main():

    ap = argparse.ArgumentParser("Sign-D2C")
    ap.add_argument("pattern", choices=["train","test"], help="train or test a model")
    ap.add_argument("stage", choices=["stage1","stage2"], help="Choose stage1 or stage2")
    ap.add_argument("config_path", default="./Configs/config.yaml", type=str, help="path to YAML config file")
    ap.add_argument("--ckpt_VAE", type=str, help="path to VQ-vae checkpoint")
    ap.add_argument("--ckpt_Diffusion", type=str, help="path to Diffusion checkpoint")
    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.pattern == "train":
        if args.stage == "stage1":
            train_stage1(cfg_file=args.config_path, ckpt=args.ckpt_VAE)
        elif args.stage == "stage2":
            train_stage2(cfg_file=args.config_path, ckpt=args.ckpt_Diffusion)
        else:
            raise ValueError("Unknown model")
            
    elif args.pattern == "test":
        if args.stage == "stage1":
            test_stage1(cfg_file=args.config_path, ckpt=args.ckpt_VAE)
        elif args.stage == "stage2":
            test_stage2(cfg_file=args.config_path, ckpt=args.ckpt_Diffusion)
        else:
            raise ValueError("Unknown model")

if __name__ == "__main__":
    main()
