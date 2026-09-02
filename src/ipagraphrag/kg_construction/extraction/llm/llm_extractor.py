import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import TextLoader
from pydantic import BaseModel
from pypdf import PdfReader
from loguru import logger

from ipagraphrag.kg_construction.data.chunk import Chunk
from ipagraphrag.kg_construction.data.mention import Mention
from ipagraphrag.kg_construction.extraction.extractor import Extractor
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ipagraphrag.llm import llm_agent

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

# setup logger
os.makedirs("logs", exist_ok=True)
logger.remove()  # remove default stderr
logger.add(sys.stdout, format=LOG_FORMAT, level="WARNING")
logger.add("logs/ipaGraphRAG.log", format=LOG_FORMAT, level="INFO")

class Prompt(BaseModel):
    prompt: str


class LLMExtractor(Extractor):

    def __init__(self, api_url, api_key, model):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def extract(self, path, **kwargs) -> tuple[list[Mention], list[Chunk]]:
        all_mentions, all_chunks = [], []

        def process_file(file_path):
            """Helper function to process a single file."""
            patient_id = os.path.splitext(os.path.basename(file_path))[0]
            logger.info(f"Extracting for {patient_id}")
            mentions, chunks = self.extract_from_file(file_path, patient_id, **kwargs)
            return mentions, chunks

        if os.path.isdir(path):
            file_names = os.listdir(path)
            file_paths = [os.path.join(path, f_name) for f_name in file_names]
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
                for future in as_completed(futures):

                    try:
                        mentions, chunks = future.result()
                        logger.info(f"finished {chunks[0].data_source}")
                        all_mentions.extend(mentions)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Error processing file {futures[future]}: {e}")
            # for f_name in file_names:
            #     patient_id = os.path.splitext(os.path.basename(os.path.join(path, f_name)))[0]
            #     logger.info(f"extract for {patient_id}")
            #     mentions, chunks = self.extract_from_file(os.path.join(path, f_name), patient_id, **kwargs)
            #     all_mentions.extend(mentions)
            #     all_chunks.extend(chunks)
        else:
            patient_id = os.path.splitext(os.path.basename(path))[0]
            mentions, chunks = self.extract_from_file(path, patient_id, **kwargs)
            all_mentions.extend(mentions)
            all_chunks.extend(chunks)
        logger.info(f"Extracted {len(all_mentions)} mentions in {len(all_chunks)} chunks")
        return all_mentions, all_chunks

    def extract_from_file(self, path, patient_name, **kwargs):
        content = ""
        if path.endswith(".pdf"):
            reader = PdfReader(path)
            # Iterate through the pages and extract text
            for page in reader.pages:
                text = page.extract_text()  # Extract text from the current page
                content += text
        elif path.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
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
            for item in content_json['medical_mentions']:
                chunk_pos = chunk.find(item["term"])
                pos = current_chunk_pos + chunk_pos
                m = Mention(item["term"], item["type"], chunk, pos, patient_name)
                chunk_mentions.append(m)
            mentions.extend(chunk_mentions)
            if len(chunk_mentions) > 0:
                c = Chunk(chunk, current_chunk_pos, patient_name, chunk_mentions)
                chunk_list.append(c)
            current_chunk_pos += len(chunk)
            mentions.extend(chunk_mentions)
        return mentions, chunk_list

    def extract_from_text(self, text, **kwargs) -> list[Mention]:
        chunk_mentions = []
        prompt = kwargs['prompt'] + ": {}".format(text)
        print()
        response = llm_agent.llm_query(self.api_url, self.api_key, prompt, self.model)
        data = json.loads(response.text)
        logger.debug(data['choices'][0]['message']['content'])
        data['choices'][0]['message']['content'] = data['choices'][0]['message']['content'].replace('```','')
        content_json = json.loads(data['choices'][0]['message']['content'])
        logger.info("found {} mentions".format(len(content_json['medical_mentions'])))
        for item in content_json['medical_mentions']:
            chunk_pos = text.find(item["term"])
            pos = chunk_pos
            m = Mention(item["term"], item["type"], text, pos, "")
            chunk_mentions.append(m)
        return chunk_mentions


