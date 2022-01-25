from u2net_model import U2NET, U2NETP
import tracer
from tracer.TRACER import TRACER
from unet import Unet
from torchinfo import summary
import torch

tracer_cfg = tracer.getConfig()
net = TRACER(tracer_cfg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.rand(1, 3, 128, 128)
net = net.to(device)
a = torch.tensor(a, device=device, dtype=torch.float32)
net(a)
# summary(tracer, input_size=(1, 3, 320, 320), col_names=("input_size", "output_size"))