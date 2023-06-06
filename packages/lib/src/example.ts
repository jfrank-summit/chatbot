import * as dotenv from 'dotenv'
import * as readline from 'readline'
import { conversationChain } from './chat'

dotenv.config()

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

const question = (prompt: string): Promise<string> => {
  return new Promise(resolve => {
    rl.question(prompt, input => {
      resolve(input)
    })
  })
}

const main = async () => {
  const chain = await conversationChain()

  while (true) {
    const input = await question('user: ')
    const action = await chain.call({ input })
    console.log(`response: ${action.response}`)
  }
}
main().catch(console.error)
