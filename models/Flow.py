{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.utils.data\n",
    "import torch.optim as optim\n",
    "import numpy as np\n",
    "import argparse\n",
    "import time\n",
    "import math\n",
    "import random\n",
    "import json\n",
    "import pathlib\n",
    "from torchinfo import summary\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from torch.autograd import Variable\n",
    "from PIL import Image\n",
    "import math\n",
    "\n",
    "class Planar(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Planar, self).__init__()\n",
    "        self.h = nn.Tanh()\n",
    "\n",
    "    def forward(self, z, u, w, b):\n",
    "        z = z.unsqueeze(2)\n",
    "        prod = torch.bmm(w, z) + b\n",
    "        f_z = z + u * self.h(prod) \n",
    "        f_z = f_z.squeeze(2) \n",
    "        psi = w * (1 - self.h(prod) ** 2) \n",
    "        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u)))\n",
    "        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)\n",
    "\n",
    "\n",
    "        return f_z, log_det_jacobian"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
