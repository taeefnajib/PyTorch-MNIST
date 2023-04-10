import torch
import os
import typing
from flytekit import workflow
from project.wf_39_284.main import Hyperparameters
from project.wf_39_284.main import run_wf

_wf_outputs=typing.NamedTuple("WfOutputs",run_wf_0=torch.nn.modules.module.Module)
@workflow
def wf_39(_wf_args:Hyperparameters)->_wf_outputs:
	run_wf_o0_=run_wf(hp=_wf_args)
	return _wf_outputs(run_wf_o0_)