import asyncio
import json
from typing import Tuple, List

import requests
from langchain_community.document_loaders import TextLoader
from openai import OpenAI
from pydantic import BaseModel
from pypdf import PdfReader

# Load environment variables from .env file

import util.plot_generation
from kg_construction.data.chunk import Chunk
from kg_construction.data.mention import Mention
from kg_construction.extraction.extractor import Extractor
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kg_construction.extraction.llm import prompt
from llm import llm_agent


class Prompt(BaseModel):
    prompt: str


class LLMExtractor(Extractor):

    def __init__(self, api_url, api_key, model):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def extract(self, file, **kwargs) -> tuple[list[Mention], list[Chunk]]:
        content = ""
        if file.endswith(".pdf"):
            reader = PdfReader(file)
            # Iterate through the pages and extract text

            for page in reader.pages:
                text = page.extract_text()  # Extract text from the current page
                content += text
        elif file.endswith(".txt"):
            loader = TextLoader(file, encoding="utf-8")
            content = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],  # Paragraph → Line → Sentence → Word → Character
            chunk_size=300,
            chunk_overlap=0
        )
        texts = text_splitter.split_documents(content)
        chunks = []
        for text in texts:
            chunks.append(text.page_content)
        mentions = []
        chunk_length = []
        chunk_list = []
        current_chunk_pos = 0
        for chunk in chunks:
            chunk_mentions = []
            prompt = kwargs['prompt'] + ": {}".format(chunk)
            chunk_length.append(len(chunk))
            response = llm_agent.llm_query(self.api_url, self.api_key, prompt, self.model)
            data = json.loads(response.text)
            content_json = json.loads(data['choices'][0]['message']['content'])
            print("found {} mentions".format(len(content_json['medical_mentions'])))
            for item in content_json['medical_mentions']:
                chunk_pos = chunk.find(item["term"])
                pos = current_chunk_pos + chunk_pos
                m = Mention(item["term"], item["type"], chunk, pos, file)
                chunk_mentions.append(m)
            mentions.extend(chunk_mentions)
            if len(chunk_mentions) > 0:
                c = Chunk(chunk, current_chunk_pos, file, chunk_mentions)
                chunk_list.append(c)
            current_chunk_pos += len(chunk)
            mentions.extend(chunk_mentions)
        print(chunk_length)
        return mentions, chunk_list


    def extract_from_text(self, text, **kwargs) -> list[Mention]:
        chunk_mentions = []
        prompt = kwargs['prompt'] + ": {}".format(text)
        response = llm_agent.llm_query(self.api_url, self.api_key, prompt, self.model)
        data = json.loads(response.text)
        print(data['choices'][0]['message']['content'])
        data['choices'][0]['message']['content'] = data['choices'][0]['message']['content'].replace('```','')
        content_json = json.loads(data['choices'][0]['message']['content'])
        print("found {} mentions".format(len(content_json['medical_mentions'])))
        for item in content_json['medical_mentions']:
            chunk_pos = text.find(item["term"])
            pos = chunk_pos
            m = Mention(item["term"], item["type"], text, pos, "")
            chunk_mentions.append(m)
        return chunk_mentions


