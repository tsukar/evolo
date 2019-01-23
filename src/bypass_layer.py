import random
from src.section import Section

class BypassLayer:
    def __init__(self, sections, source_index):
        self.sections = sections
        self.source_index = source_index

    def get_name(self):
        return self.__class__.__name__

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

class ConcatLayer(BypassLayer):
    def get_output_size(self, in_h, in_w, in_c):
        source_h, source_w, _ = self.get_source_size(self.source_index)
        source_c = int(self.sections[1].params['filters'])
        if source_h % in_h == 0:
            stride = int(source_h / in_h)
            out_h = int(source_h / stride)
            out_w = int(source_w / stride)
            out_c = int(source_c * (stride ** 2) + in_c)
        elif in_h % source_h == 0:
            stride = int(in_h / source_h)
            out_h = int(source_h * stride)
            out_w = int(source_w * stride)
            out_c = int(source_c / (stride ** 2) + in_c)
            self.sections[2].params['reverse'] = '1'
        else:
            return 0, 0, 0

        self.sections[2].params['stride'] = str(stride)
        return out_h, out_w, out_c

    @classmethod
    def create(cls):
        concat_sections = [
            Section('[route]'),
            Section('[convolutional]'),
            Section('[reorg]'),
            Section('[route]')
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

class SkipLayer(BypassLayer):
    def get_output_size(self, in_h, in_w, in_c):
        source_h, source_w, _ = self.get_source_size(self.source_index)
        self.sections[2].params['filters'] = str(in_c)
        out_c = in_c
        if source_h % in_h == 0:
            stride = int(source_h / in_h)
            out_h = int(source_h / stride)
            out_w = int(source_w / stride)
        elif in_h % source_h == 0:
            stride = int(in_h / source_h)
            out_h = int(source_h * stride)
            out_w = int(source_w * stride)
            self.sections[1].params['reverse'] = '1'
        else:
            return 0, 0, 0

        self.sections[1].params['stride'] = str(stride)
        return out_h, out_w, out_c

    @classmethod
    def create(cls):
        skip_sections = [
            Section('[route]'),
            Section('[reorg]'),
            Section('[convolutional]'),
            Section('[shortcut]')
        ]

        skip_sections[0].params['layers'] = None
        skip_sections[1].params['stride'] = '2'
        skip_sections[2].params['batch_normalize'] = '1'
        skip_sections[2].params['size'] = '1'
        skip_sections[2].params['stride'] = '1'
        skip_sections[2].params['pad'] = '1'
        skip_sections[2].params['filters'] = '64'
        skip_sections[2].params['activation'] = 'leaky'
        skip_sections[3].params['layers'] = '-1,-4'

        source_index = random.randrange(23)
        return SkipLayer(skip_sections, source_index)
