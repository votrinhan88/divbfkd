from abc import ABC, abstractmethod, abstractproperty

class Metric(ABC):
    @abstractmethod
    def update(self):
        """Update metric from given inputs."""
        pass

    # @abstractmethod
    def update_ddp(self):
        """Update metric from given inputs with Distributed Data Parallel logic.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset tracked parameters, typically used when at the beginning of an 
        epoch or the beginning of validation."""
        pass
    
    @abstractproperty
    def value(self):
        """Return the latest metric value."""
        return None

    @property
    def py_value(self):
        """Return the latest metric value in a pythonic format."""
        return self.value.item()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(py_value={self.py_value:.4g})"