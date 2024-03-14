from dataclasses import dataclass, asdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


@dataclass
class Metrics:
    f1: AverageMeter = AverageMeter()
    acc: AverageMeter = AverageMeter()
    sen: AverageMeter = AverageMeter()
    spec: AverageMeter = AverageMeter()
    loss: AverageMeter = AverageMeter()
    auc: AverageMeter = AverageMeter()
    prauc: AverageMeter = AverageMeter()

    def update(self, n: int, metrics: dict):
        for key, value in metrics.items():
            if hasattr(self, key):
                meter = getattr(self, key)
                meter.update(value, n)

            else:
                raise AttributeError(f"Attribute '{key}' not found in Metrics.")

    def to_dict(self, prefix=str()):
        return {
            prefix + attr: round(meter.avg, 5) for attr, meter in asdict(self).items()
        }
