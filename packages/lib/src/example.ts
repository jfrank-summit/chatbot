import * as dotenv from 'dotenv'
import * as readline from 'readline'
import { conversation } from './chat'

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
  const conv = conversation()

  while (true) {
    const input = await question('user: ')
    const action = await conv.converse(input)
    console.log(`response: ${JSON.stringify(action.response)}`)
  }
}
main().catch(console.error)
