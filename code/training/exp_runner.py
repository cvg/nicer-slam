import sys

sys.path.append("../code")
import torch
import random
import argparse
import numpy as np

from training.volsdf_train import SLAMRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/replica/runconf_replica_2.conf")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument(
        "--is_continue", default=False, action="store_true", help="If set, indicates continuing from a previous run."
    )
    parser.add_argument(
        "--new_expfolder",
        default=False,
        action="store_true",
        help="If set, create new expfolder when continuing from a previous run.",
    )
    parser.add_argument(
        "--timestamp",
        default="latest",
        type=str,
        help="The timestamp of the run to be used in case of continuing from a previous run.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        type=str,
        help="The checkpoint of the run to be used in case of continuing from a previous run.",
    )
    parser.add_argument(
        "--scan_id", type=int, default=-1, help="If set, taken to be the scan id. Overrides the conf file."
    )

    opt = parser.parse_args()

    trainrunner = SLAMRunner(
        conf=opt.conf,
        expname=opt.expname,
        exps_folder_name=opt.exps_folder,
        is_continue=opt.is_continue,
        timestamp=opt.timestamp,
        new_expfolder=opt.new_expfolder,
        checkpoint=opt.checkpoint,
        scan_id=opt.scan_id,
    )

    trainrunner.run()
