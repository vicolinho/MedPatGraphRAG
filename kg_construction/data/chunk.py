from kg_construction.data.mention import Mention


class Chunk:

    def __init__(self, text, pos, data_source, mentions: list[Mention]):
        self.text = text
        self.mentions = mentions
        self.pos = pos
        self.data_source = data_source

    def __str__(self):
        return (f'{self.text}:\n'
                f' {self.mentions}')

    def __repr__(self):
        return (f'{self.text}:\n'
                f' {self.mentions}')




