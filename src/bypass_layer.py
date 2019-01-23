import random
from src.section import Section

class BypassLayer:
    def __init__(self, sections, source_index):
        self.sections = sections
        self.source_index = source_index

class ConcatLayer(BypassLayer):
    def get_output_size(self, in_h, in_w, in_c):
        source_h, source_w, _ = self.get_source_size(self.source_index)
        source_c = int(self.sections[1].params['filters'])
        stride = int(self.sections[2].params['stride'])

        out_h = int(source_h / stride)
        out_w = int(source_w / stride)
        out_c = int(source_c * (stride ** 2) + in_c)
        return out_h, out_w, out_c

    def get_source_size(self, source_index):
        source_sizes = [
            (416, 416, 32),
            (208, 208, 32),
            (208, 208, 64),
            (104, 104, 64),
            (104, 104, 128),
            (104, 104, 64),
            (104, 104, 128),
            (52, 52, 128),
            (52, 52, 256),
            (52, 52, 128),
            (52, 52, 256),
            (26, 26, 256),
            (26, 26, 512),
            (26, 26, 256),
            (26, 26, 512),
            (26, 26, 256),
            (26, 26, 512),
            (13, 13, 512),
            (13, 13, 1024),
            (13, 13, 512),
            (13, 13, 1024),
            (13, 13, 512),
            (13, 13, 1024)
        ]
        return source_sizes[source_index]

    @classmethod
    def create(cls):
        concat_sections = [
            Section('[route]'),
            Section('[convolutional]'),
            Section('[reorg]'),
            Section('[route]'),
        ]

        concat_sections[0].params['layers'] = None
        concat_sections[1].params['batch_normalize'] = '1'
        concat_sections[1].params['size'] = '1'
        concat_sections[1].params['stride'] = '1'
        concat_sections[1].params['pad'] = '1'
        concat_sections[1].params['filters'] = '64'
        concat_sections[1].params['activation'] = 'leaky'
        concat_sections[2].params['stride'] = '2'
        concat_sections[3].params['layers'] = '-1,-4'

        source_index = random.randrange(23)
        return ConcatLayer(concat_sections, source_index)
