from src.section import Section

class SimpleLayer:
    def __init__(self, section):
        self.section = section

class ConvLayer(SimpleLayer):
    def get_output_size(self):
        pass

    @classmethod
    def create(cls):
        conv_section = Section('[convolutional]')
        conv_section.params['batch_normalize'] = '1'
        conv_section.params['size'] = '3'
        conv_section.params['stride'] = '1'
        conv_section.params['pad'] = '1'
        conv_section.params['filters'] = '1024'
        conv_section.params['activation'] = 'leaky'
        return ConvLayer(conv_section)

class MaxPoolLayer(SimpleLayer):
    def get_output_size(self):
        pass
