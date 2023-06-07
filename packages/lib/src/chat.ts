import { ChatOpenAI } from 'langchain/chat_models/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { chatWithHistory } from './prompts';
import { HumanChatMessage, SystemChatMessage } from 'langchain/schema';
import { Document } from 'langchain/document';
import { TimeWeightedVectorStoreRetriever } from 'langchain/retrievers/time_weighted';

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

export const conversation = (config = defaultConfig) => {
  const chat = new ChatOpenAI(config);
  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings());

  const retriever = new TimeWeightedVectorStoreRetriever({
    vectorStore,
    memoryStream: [],
    k: 3,
    searchKwargs: 5,
  });

  const searchForRelevant = async (input: string) => {
    const search = await retriever.getRelevantDocuments(input); //await vectorStore.similaritySearch(input, 2);
    console.log(`doc count: ${search.length}`);
    const relevantHistory = search.reduce((acc, doc) => `${acc}\n${doc.pageContent}`, '');
    return relevantHistory;
  };

  const converse = async (input: string) => {
    const relevantHistory = await searchForRelevant(input);
    const history = chatWithHistory(relevantHistory, input);
    console.log(`history: ${history}`);

    const response = await chat.call([new SystemChatMessage(history), new HumanChatMessage(input)]);
    const doc = new Document({ pageContent: `human: ${input}\nai: ${response.text}\n` });
    retriever.addDocuments([doc]);
    return { response: response.text };
  };

  return { converse };
};
