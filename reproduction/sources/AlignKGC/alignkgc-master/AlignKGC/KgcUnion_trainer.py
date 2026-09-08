from AlignKGC.Jaccard_trainer import Jaccard


class KgcUnion(Jaccard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raloss_coeff = 0.0
