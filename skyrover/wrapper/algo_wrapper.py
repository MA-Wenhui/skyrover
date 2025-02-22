
class AlgorithmWrapperBase:
    def reset(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")

    def step(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
