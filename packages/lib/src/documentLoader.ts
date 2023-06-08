import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { JSONLoader, JSONLinesLoader } from 'langchain/document_loaders/fs/json';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CSVLoader } from 'langchain/document_loaders/fs/csv';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

export const dirToDocs = async (dir: string) => {
  const loader = new DirectoryLoader(dir, {
    '.json': path => new JSONLoader(path, '/texts'),
    '.jsonl': path => new JSONLinesLoader(path, '/html'),
    '.txt': path => new TextLoader(path),
    '.csv': path => new CSVLoader(path, 'text'),
    '.pdf': path => new PDFLoader(path),
  });
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1500,
    chunkOverlap: 250,
  });
  const docs = await loader.loadAndSplit(splitter);
  return docs;
};
