import { ChatOpenAI } from 'langchain/chat_models/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { chatWithHistory } from './prompts';
import { HumanChatMessage, SystemChatMessage } from 'langchain/schema';
import { Document } from 'langchain/document';
import { TimeWeightedVectorStoreRetriever } from 'langchain/retrievers/time_weighted';
import { dirToDocs } from './documentLoader';

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

export const createChat = async (config = defaultConfig) => {
  const chat = new ChatOpenAI(config);
  const convVectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
  const kbVectorStore = new MemoryVectorStore(new OpenAIEmbeddings());

  const kbDocs = await dirToDocs('data');
  kbVectorStore.addDocuments(kbDocs);

  const convRetriever = new TimeWeightedVectorStoreRetriever({
    vectorStore: convVectorStore,
    memoryStream: [],
    k: 3,
    searchKwargs: 5,
  });

  const searchHistory = async (input: string) => {
    const searchConv = await convRetriever.getRelevantDocuments(input); 
    return searchConv.reduce((acc, doc) => `${acc}\n${doc.pageContent}`, '');
  };

  const searchKb = async (input: string) => {
    const searchKb = await kbVectorStore.similaritySearch(input, 4);
    return searchKb.reduce((acc, doc) => `${acc}\n${doc.pageContent}`, '');
  };

  const searchForRelevant = async (input: string) => {
    const relevantHistory = await searchHistory(input);
    const relevantKb = await searchKb(input);
    return { relevantHistory, relevantKb };
  };

  const converse = async (input: string) => {
    const { relevantHistory, relevantKb } = await searchForRelevant(input);
    const history = chatWithHistory(relevantHistory, relevantKb);

    const response = await chat.call([new SystemChatMessage(history), new HumanChatMessage(input)]);
    const doc = new Document({ pageContent: `human: ${input}\nai: ${response.text}\n` });
    convRetriever.addDocuments([doc]);
    return { response: response.text };
  };

  return { converse };
};
