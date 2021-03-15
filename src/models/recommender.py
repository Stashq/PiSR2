from abc import ABC


class RecommenderSystem(ABC):
    def __init__(self):
        super(RecommenderSystem, self).__init__()
        self.MAX_RATING = 5
        self.MIN_RATING = 0
