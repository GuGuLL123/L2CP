from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.jit
import random
from lib.losses.loss import *
import cv2
class FusionTTA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer_fusion(self.model, self.optimizer)

    def forward(self, x,gt):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt_fusion(x, gt, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer_fusion(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy_fusion(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_fusion(x, gt, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    #fusion
    # source_vessel_namepool_train = ["training/1st_manual/{}_manual1.gif".format(i) for i in range(21, 41)]
    # source_vessel_namepool_test = ["test/1st_manual/{:02d}_manual1.gif".format(i) for i in range(1, 21)]
    # source_vessel_namepool = source_vessel_namepool_train + source_vessel_namepool_test
    #
    #
    # kernel = np.ones((13, 13), np.uint8)
    # backgrounds = np.array(x.detach().cpu())
    # for index in range(x.shape[0]):
    #
    #     random_number = random.randint(0, 40)
    #     source_vessel_name = source_vessel_namepool[random_number]
    #     source_vessel_path = '/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/DRIVE/' + source_vessel_name
    #     source_vessel = cv2.imread(source_vessel_path, cv2.IMREAD_GRAYSCALE)
    #
    #     background = backgrounds[index,0, :, :]
    #     closing = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel)
    # forward
    dice_loss = DiceLoss(2)
    criterion = CrossEntropyLoss2d()
    outputs = model(x)
    outputs_soft = torch.softmax(outputs, dim=1)
    loss1 = criterion(outputs, gt)
    loss2 = dice_loss(outputs_soft, gt.unsqueeze(1))
    loss = loss1 + loss2
    # loss = loss2
    loss.backward()
    # adapt
    # loss = softmax_entropy_fusion(outputs).mean()
    # loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params_fusion(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer_fusion(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer_fusion(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model_fusion(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(True)
    # configure norm for tent updates: enable grad + force batch statisics
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.requires_grad_(True)
    #         # force use of batch stats in train and eval modes
    #         m.track_running_stats = False
    #         m.running_mean = None
    #         m.running_var = None
    return model


def check_model_fusion(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
