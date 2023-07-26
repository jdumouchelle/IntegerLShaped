from abc import ABC, abstractmethod


class InstanceGenerator(ABC):
    @abstractmethod
    def generate_instance(self):
        """ Generates instance and stores as python dictionary. """
        pass