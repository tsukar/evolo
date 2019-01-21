class SimpleLayer:
    def __init__(self, section):
        self.section = section

class ConvLayer(SimpleLayer):
    def get_output_size(self):
        pass

class MaxPoolLayer(SimpleLayer):
    def get_output_size(self):
        pass
