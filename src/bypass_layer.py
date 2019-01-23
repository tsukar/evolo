class BypassLayer:
    def __init__(self, sections, current_index):
        self.sections = sections
        self.from_index = current_index + int(sections[0].params['layers'])

class ConcatLayer(BypassLayer):
    def get_output_size(self, in_h, in_w, in_c):
        from_h, from_w = 26, 26
        from_c = int(self.sections[1].params['filters'])
        stride = int(self.sections[2].params['stride'])

        out_h = int(from_h / stride)
        out_w = int(from_w / stride)
        out_c = int(from_c * (stride ** 2) + in_c)
        return out_h, out_w, out_c
