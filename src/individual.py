import random
import copy
import re
import subprocess
from src.section import Section
from src.simple_layer import ConvLayer, MaxPoolLayer, DropoutLayer, ConnectedLayer
from src.bypass_layer import ConcatLayer, SkipLayer

class Individual:
    def __init__(self, sections, gen, id):
        self.sections = sections
        self.sections[0].params['max_batches '] = str(self.get_adaptive_max_batches(gen))
        self.layers = self.import_layers()
        self.gen = gen
        self.id = id
        self.score = 0.0

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
                source_index = 23 + i + int(target_sections[i].params['layers'])
                layers.append(ConcatLayer(target_sections[i:(i + 4)], source_index))
                i += 3
            i += 1

        return layers

    def export_layers(self):
        sections = self.sections[:24]
        for layer in self.layers:
            if hasattr(layer, 'section'):
                sections.append(layer.section)
            elif hasattr(layer, 'sections'):
                layer.sections[0].params['layers'] = str(layer.source_index - len(sections) + 1)
                sections.extend(layer.sections)
        sections.extend(self.sections[-2:])
        return sections

    def get_filename(self):
        padded_gen = str(self.gen).zfill(2)
        padded_id = str(self.id).zfill(2)
        return f'individuals/{padded_gen}-{padded_id}.cfg'

    def get_adaptive_max_batches(self, gen):
        if gen == 0:
            return 10000
        elif gen == 1:
            return 15000
        elif gen == 2:
            return 20000
        elif gen == 3:
            return 25000

    def is_valid(self):
        h, w, c = 13, 13, 1024
        for layer in self.layers:
            h, w, c = layer.get_output_size(h, w, c)
            if h == 0 or w == 0 or c == 0:
                return False
        return True

    def sample_layer_by_name(self, name):
        candidates = [l for l in self.layers if l.get_name == name]
        if candidates:
            chosen = random.choice(candidates)
        else:
            chosen = None
        return chosen

    def remove_layer_by_name(self, name):
        chosen = self.sample_layer_by_name(name)
        if not chosen:
            return False
        else:
            chosen_index = self.layers.index(chosen)
            self.layers.pop(chosen_index)
            return True

    def add_convolution(self):
        conv_layer = ConvLayer.create()
        position = random.randrange(len(self.layers) + 1)
        self.layers.insert(position, conv_layer)
        return True

    def remove_convolution(self):
        return self.remove_layer_by_name('ConvLayer')

    def alter_channel_number(self):
        chosen = self.sample_layer_by_name('ConvLayer')
        if not chosen:
            return False
        else:
            chosen.section.params['filters'] = random.choice([512, 1024, 2048])
            return True

    def alter_filter_size(self):
        chosen = self.sample_layer_by_name('ConvLayer')
        if not chosen:
            return False
        else:
            chosen.section.params['size'] = random.choice([1, 3, 5])
            return True

    def alter_stride(self):
        chosen = self.sample_layer_by_name('ConvLayer')
        if not chosen:
            return False
        else:
            chosen.section.params['stride'] = random.choice([1, 2])
            return True

    def add_dropout(self):
        dropout_layer = DropoutLayer.create()
        position = random.randrange(len(self.layers) + 1)
        self.layers.insert(position, dropout_layer)
        return True

    def remove_dropout(self):
        return self.remove_layer_by_name('DropoutLayer')

    def add_pooling(self):
        maxpool_layer = MaxPoolLayer.create()
        position = random.randrange(len(self.layers) + 1)
        self.layers.insert(position, maxpool_layer)
        return True

    def remove_pooling(self):
        return self.remove_layer_by_name('MaxPoolLayer')

    def add_skip(self):
        skip_layer = SkipLayer.create()
        position = random.randrange(len(self.layers) + 1)
        self.layers.insert(position, skip_layer)
        return True

    def remove_skip(self):
        return self.remove_layer_by_name('SkipLayer')

    def add_concatenate(self):
        concat_layer = ConcatLayer.create()
        position = random.randrange(len(self.layers) + 1)
        self.layers.insert(position, concat_layer)
        return True

    def remove_concatenate(self):
        return self.remove_layer_by_name('ConcatLayer')

    def add_fully_connected(self):
        connected_layer = ConnectedLayer.create()
        self.layers.append(connected_layer)
        return True

    def remove_fully_connected(self):
        return self.remove_layer_by_name('ConnectedLayer')

    def mutate(self):
        operations = [
            self.add_convolution,
            self.remove_convolution,
            self.alter_channel_number,
            self.alter_filter_size,
            self.alter_stride,
            # self.add_dropout,
            # self.remove_dropout,
            self.add_pooling,
            self.remove_pooling,
            # self.add_skip,
            # self.remove_skip,
            self.add_concatenate,
            self.remove_concatenate,
            # self.add_fully_connected,
            # self.remove_fully_connected
        ]
        operation_weights = [
            2, 1,
            2, 2, 2,
            1, 1,
            2, 1
        ]

        is_success = False
        layers_backup = copy.deepcopy(self.layers)
        while is_success == False:
            self.layers = copy.deepcopy(layers_backup)
            selected_operation = random.choices(operations, k=1, weights=operation_weights)[0]
            if selected_operation():
                is_success = self.is_valid()
        return self

    def evaluate(self):
        filename = self.get_filename()
        weights_filename = filename.replace('individuals', 'backup').replace('.cfg', '') + '_final.weights'
        proc = subprocess.run([
            './darknet',
            'detector',
            'map',
            'cfg/x-ray.data',
            filename,
            weights_filename
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        score = float(re.search(r'mean average precision \(mAP\) = (0\.\d+), ', proc.stdout.decode('utf8')).group(1))
        self.score = score
        return score

    def populate(self):
        for i in range(4):
            ind = Individual.load(self.get_filename(), self.gen + 1, i)
            if i > 0:
                ind.mutate()
            ind.save()

    @classmethod
    def load(cls, path_to_cfg_file, gen, id):
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
        return Individual(sections, gen, id)
