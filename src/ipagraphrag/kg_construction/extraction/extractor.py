from ipagraphrag.kg_construction.data.mention import Mention


class Extractor(object):

    def __init__(self):
        pass

    def extract(self, file) -> list[Mention]:
        pass

    def extract_from_text(self, text) -> list[Mention]:
        pass