from abc import ABC, abstractmethod
 
class Model(ABC):
 
    def __init__(self, path):
        self.path = path
        super().__init__()
    
    @abstractmethod
    def generate(self):
        pass