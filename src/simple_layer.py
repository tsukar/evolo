from src.section import Section

class SimpleLayer:
    def __init__(self, section):
        self.section = section

    def get_name(self):
        return self.__class__.__name__

class ConvLayer(SimpleLayer):
    def get_output_size(self, in_h, in_w, in_c):
        size = int(self.section.params['size'])
        stride = int(self.section.params['stride'])
        pad = int(self.section.params['pad']) if size > 1 else 0

        out_h = int((in_h + pad * 2 - size) / stride + 1)
        out_w = int((in_w + pad * 2 - size) / stride + 1)
        out_c = int(self.section.params['filters'])
        return out_h, out_w, out_c

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
    def get_output_size(self, in_h, in_w, in_c):
        size = int(self.section.params['size'])
        stride = int(self.section.params['stride'])

        out_h = int((in_h - size) / stride + 1)
        out_w = int((in_w - size) / stride + 1)
        out_c = in_c
        return out_h, out_w, out_c

    @classmethod
    def create(cls):
        maxpool_section = Section('[maxpool]')
        maxpool_section.params['size'] = '2'
        maxpool_section.params['stride'] = '2'
        return MaxPoolLayer(maxpool_section)

class DropoutLayer(SimpleLayer):
    def get_output_size(self, in_h, in_w, in_c):
        return in_h, in_w, in_c

    @classmethod
    def create(cls):
        dropout_section = Section('[dropout]')
        dropout_section.params['probability'] = '.5'
        return DropoutLayer(dropout_section)
