import copy
import torch
import numpy as np
import time

from client.clientbase import Client


class ClientAvg(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
