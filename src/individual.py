import random
from src.section import Section
from src.simple_layer import ConvLayer, MaxPoolLayer
from src.bypass_layer import ConcatLayer

class Individual:
    def __init__(self, sections):
        self.sections = sections
        self.layers = self.import_layers()

    def save(self):
        self.sections = self.export_layers()
        with open(self.get_filename(), mode='w') as f:
            for section in self.sections:
                f.write(section.dump())

    def import_layers(self):
        layers = []
        target_sections = self.sections[24:-2]

        i = 0
        while i < len(target_sections):
            section = target_sections[i]
            section_name = section.name
            if section_name == '[convolutional]':
                layers.append(ConvLayer(section))
            elif section_name == '[maxpool]':
                layers.append(MaxPoolLayer(section))
            elif section_name == '[route]':
                layers.append(ConcatLayer(target_sections[i:(i + 4)], i))
                i += 3
            i += 1

        return layers

    def export_layers(self):
        sections = self.sections[:24]
        for layer in self.layers:
            if hasattr(layer, 'section'):
                sections.append(layer.section)
            elif hasattr(layer, 'sections'):
                sections.extend(layer.sections)
        sections.extend(self.sections[-2:])
        return sections

    def get_filename(self):
        return 'model.cfg'

    def is_valid(self):
        h, w, c = 13, 13, 1024
        for layer in self.layers:
            h, w, c = layer.get_output_size(h, w, c)
            if h == 0 or w == 0 or c == 0:
                return False
        return True

    def add_convolution(self):
        conv_layer = ConvLayer.create()
        position = random.randrange(len(self.layers) + 1)
        self.layers.insert(position, conv_layer)

    def remove_convolution(self):
        pass

    def alter_channel_number(self):
        pass

    def alter_filter_size(self):
        pass

    def alter_stride(self):
        pass

    def add_dropout(self):
        pass

    def remove_dropout(self):
        pass

    def add_pooling(self):
        maxpool_layer = MaxPoolLayer.create()
        position = random.randrange(len(self.layers) + 1)
        self.layers.insert(position, maxpool_layer)

    def remove_pooling(self):
        pass

    def add_skip(self):
        pass

    def remove_skip(self):
        pass

    def add_concatenate(self):
        pass

    def remove_concatenate(self):
        pass

    def add_fully_connected(self):
        pass

    def remove_fully_connected(self):
        pass

    @classmethod
    def load(cls, path_to_cfg_file):
        sections = []
        with open(path_to_cfg_file) as f:
            for line in f:
                line = line[:-1]
                if line == '' or line[0] == '#':
                    next
                elif line[0] == '[':
                    sections.append(Section(line))
                else:
                    sections[-1].add_entry(line)
        return Individual(sections)
