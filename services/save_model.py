import bentoml
import torch

with open("/userRepoData/taeefnajib/pima-diabetes/sidetrek/models/df84234d42f872a7feb4ba11779b3dbb.pt", "rb") as f:
    model = torch.load(f)
    saved_model = bentoml.pytorch.save(model, "saved_model", signatures={"__call__": {"batch_dim": 0, "batchable": False}})
    print(saved_model) # This is required!