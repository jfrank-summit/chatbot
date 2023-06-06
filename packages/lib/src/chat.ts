import { ChatOpenAI, OpenAIChatInput } from 'langchain/chat_models/openai'
import * as dotenv from 'dotenv'
import { PromptTemplate } from 'langchain/prompts'
import { ConversationChain } from 'langchain/chains'
import { VectorStoreRetrieverMemory } from 'langchain/memory'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { chatPromptTemplate } from './prompts'

dotenv.config()

interface ChatConfig {
  temperature: number
  openAIApiKey: string | undefined
  modelName: 'gpt-3.5-turbo' | 'gpt-4'
}
const defaultConfig: ChatConfig = {
  temperature: 0.9,
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo'
}

export const conversationChain = async (config = defaultConfig) => {
  const chat = new ChatOpenAI(config)
  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings())

  const memory = new VectorStoreRetrieverMemory({
    vectorStoreRetriever: vectorStore.asRetriever(3),
    memoryKey: 'history'
  })
  const chatPrompt = PromptTemplate.fromTemplate(chatPromptTemplate)

  const chain = new ConversationChain({
    memory,
    prompt: chatPrompt,
    llm: chat,
    verbose: true
  })
  return chain
}
