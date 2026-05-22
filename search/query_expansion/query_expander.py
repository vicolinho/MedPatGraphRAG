import json
import os

from llm import llm_agent
from search.query_expansion import prompts
class QueryExpander(object):

    def __init__(self, api_url, api_key, model):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def expand_query(self, query, context_information):
        query = prompts.QUERY_TEMPLATE.format(query, context_information)
        response = llm_agent.llm_query(self.api_url, self.api_key, query, self.model)
        data = json.loads(response.text)
        print(data['choices'][0]['message']['content'])
        data['choices'][0]['message']['content'] = data['choices'][0]['message']['content'].replace('```', '')
        content_json = json.loads(data['choices'][0]['message']['content'])
        print(content_json)
        # for item in content_json['medical_mentions']:
        #     chunk_pos = text.find(item["term"])
        #     pos = chunk_pos
        #     m = Mention(item["term"], item["type"], text, pos, "")
        #     chunk_mentions.append(m)
        # return chunk_mentions
