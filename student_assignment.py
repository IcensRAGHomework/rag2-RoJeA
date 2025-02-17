from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)
import unittest

q1_pdf = "OpenSourceLicenses.pdf"
q2_pdf = "勞動基準法.pdf"


def hw02_1(q1_pdf):
    loader_q1 = PyPDFLoader(q1_pdf)
    document_q1 = loader_q1.load()
    splitter = CharacterTextSplitter(chunk_overlap=0)
    chunks = splitter.split_documents(document_q1)
    last_chunck = chunks[-1]
    return last_chunck

def hw02_2(q2_pdf):
    loader_q2 = PyPDFLoader(q2_pdf)
    document_q2 = loader_q2.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20, chunk_overlap=0, 
        separators=[r"第\s+.*\s+章\s+", r"第\s+.*\s+條\s+"],
        is_separator_regex = True)
    chunks = splitter.split_text(" ".join([doc.page_content for doc in document_q2]))
    return len(chunks)

class TestTextSplitter(unittest.TestCase):
    def test_q1(self):
        result = hw02_1(q1_pdf)
        self.assertEqual(result.metadata['source'], 'OpenSourceLicenses.pdf')
        self.assertIsNotNone(result.page_content)
        print(f"檔名: {result.metadata['source']}")
        print(f"頁數: {result.metadata['page']}")
        print(f"頁面內容: {result.page_content[:10]}...")
    
    def test_q2(self):
        result = hw02_2(q2_pdf)
        self.assertEqual(result, 111) # 111 chunks
        print(f"總共切割出的 chunks 數量: {result}")

if __name__ == "__main__":
    unittest.main()