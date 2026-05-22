class Mention(object):

    def __init__(self, term, mention_type, chunk: str, pos, data_source):
        self.term = term
        self.type = mention_type
        self.chunk = chunk
        self.pos = pos
        self.data_source = data_source


    def __str__(self):
        return f'{self.type} {self.term}'

    def __repr__(self):
        return f'{self.type} {self.term}'

    def __eq__(self, other):
        if not isinstance(other, Mention):
            return False
        # Compare based on pos and data_source
        return self.pos == other.pos and self.data_source == other.data_source

    def __hash__(self):
        # Combine pos and data_source into a hash value
        return hash((self.pos, self.data_source))

