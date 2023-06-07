import { ChatOpenAI } from 'langchain/chat_models/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { chatWithHistory } from './prompts';
import { HumanChatMessage, SystemChatMessage } from 'langchain/schema';
import { Document } from 'langchain/document';
import { TimeWeightedVectorStoreRetriever } from 'langchain/retrievers/time_weighted';

import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { JSONLoader, JSONLinesLoader } from 'langchain/document_loaders/fs/json';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CSVLoader } from 'langchain/document_loaders/fs/csv';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

interface ChatConfig {
  temperature: number;
  openAIApiKey: string | undefined;
  modelName: 'gpt-3.5-turbo' | 'gpt-4';
}
const defaultConfig: ChatConfig = {
  temperature: 0.9,
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo',
};

const dirToDocs = async (dir: string) => {
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

export const conversation = async (config = defaultConfig) => {
  const chat = new ChatOpenAI(config);
  const convVectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
  const kbVectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
  const kbDocs = await dirToDocs('data');
  console.log(`kbDocs: ${JSON.stringify(kbDocs.length)}`);
  kbVectorStore.addDocuments(kbDocs);

  const convRetriever = new TimeWeightedVectorStoreRetriever({
    vectorStore: convVectorStore,
    memoryStream: [],
    k: 3,
    searchKwargs: 5,
  });

  const searchForRelevant = async (input: string) => {
    const searchConv = await convRetriever.getRelevantDocuments(input); //await vectorStore.similaritySearch(input, 2);
    console.log(`doc count: ${searchConv.length}`);
    const relevantHistory = searchConv.reduce((acc, doc) => `${acc}\n${doc.pageContent}`, '');

    const searchKb = await kbVectorStore.similaritySearch(input, 4);
    const relevantKb = searchKb.reduce((acc, doc) => `${acc}\n${doc.pageContent}`, '');

    return { relevantHistory, relevantKb };
  };

  const converse = async (input: string) => {
    const { relevantHistory, relevantKb } = await searchForRelevant(input);
    const history = chatWithHistory(relevantHistory, relevantKb);
    console.log(`history: ${history}`);

    const response = await chat.call([new SystemChatMessage(history), new HumanChatMessage(input)]);
    const doc = new Document({ pageContent: `human: ${input}\nai: ${response.text}\n` });
    convRetriever.addDocuments([doc]);
    return { response: response.text };
  };

  return { converse };
};
