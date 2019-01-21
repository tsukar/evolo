class BypassLayer:
    def __init__(self, sections, current_index):
        self.sections = sections
        self.from_index = current_index + int(sections[0].params['layers'])

class ConcatLayer(BypassLayer):
    def get_output_size(self):
        pass
