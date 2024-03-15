from dataclasses import dataclass, asdict


@dataclass
class AverageMeter:
    """Computes and stores the average and current value"""

    name: str
    avg: float = 0

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
