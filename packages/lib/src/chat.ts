import { ChatOpenAI } from 'langchain/chat_models/openai'
import * as dotenv from 'dotenv'
import { PromptTemplate } from 'langchain/prompts'
import { ConversationChain } from 'langchain/chains'
import { VectorStoreRetrieverMemory } from 'langchain/memory'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { chatPromptTemplate } from './prompts'
import { BaseChatMessage, HumanChatMessage, SystemChatMessage } from 'langchain/schema'
import { Document } from 'langchain/document'

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

export const conversation = () => {
  const chat = new ChatOpenAI(defaultConfig)
  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings())

  const converse = async (input: string) => {
    const search = await vectorStore.similaritySearch(input, 2)
    const relevantHistory = search.reduce((acc, doc) => `${acc}\n${doc.pageContent}`, '')
    const systemMessage = `The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Relevant pieces of previous conversation:
        ${relevantHistory}

        (You do not need to use these pieces of information if not relevant)`
    console.log(systemMessage)
    const response: BaseChatMessage = await chat.call([
      new SystemChatMessage(systemMessage),
      new HumanChatMessage(input)
    ])
    const doc = new Document({ pageContent: `human: ${input}\nai: ${response.text}\n` })
    vectorStore.addDocuments([doc])
    return { response: response.text }
  }

  return { converse }
}
