class Section(dict):
    def __init__(self, name):
        self.name = name
        self.params = {}

    def add_entry(self, line):
        (key, val) = line.split('=', 1)
        self.params[key] = val

    def dump(self):
        section_text = self.name + '\n'
        for key, val in self.params.items():
            section_text += '='.join([key, val]) + '\n'
        return section_text
