import { type Config } from "./lib/utils";
import { createOpenAI } from "@ai-sdk/openai";
import { z } from "zod";
import { anthropic, type AnthropicProviderOptions } from "@ai-sdk/anthropic";


const vllm = createOpenAI({
  apiKey: "",
  baseURL: "http://localhost:8015/v1",
});

async function beans001RewardFunc(
  prompt: string,
  completion: string,
  problem: z.infer<typeof ChessGameSampleSchema>
) {
  const response = await fetch("http://localhost:8001/calculate-reward", {
    method: "POST",
    body: JSON.stringify({
      llm_response: completion,
      target: problem.target,
      question_type: problem.question_type,
      fen: problem.fen,
    }),
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to calculate reward: ${errorText}`);
  }

  return z.number().parse(await response.json());
}

function parseCompletion(text: string | undefined) {
  if (!text) return "";
  const lastAnswerIndex = text?.lastIndexOf("<answer>");
  if (lastAnswerIndex === -1 || !lastAnswerIndex) return text || "";
  const think = text?.slice(0, lastAnswerIndex)?.trim();
  if (!think) return text || "";
  const answer = text
    ?.slice(lastAnswerIndex + 8)
    ?.split("</answer>")[0]
    ?.trim();
  return `<think>\n${think}\n</think>\n\n<answer>${answer === "" ? "\n" : `\n${answer}\n`
    }</answer>`;
}

const ChessGameSampleSchema = z.object({
  id: z.string().describe("The unique identifier for the sample"),
  fen: z.string().describe("The FEN representation of the board"),
  target: z.string().describe("The expected answer (from get_target)"),
  question_type: z.string().describe("The original question type"),
  prompt: z.string().describe("The formatted prompt (from get_prompt)"),
});

export const chess_query_config: Config<typeof ChessGameSampleSchema> = {
  name: "sft_gen",
  llm: {
    model: anthropic("claude-3-7-sonnet-20250219"),
    maxRunning: 2,
    providerOptions: {
      anthropic: {
        thinking: { type: 'enabled', budgetTokens: 1024 },
      } satisfies AnthropicProviderOptions,
    },
    params: {
      temperature: 0.6,
      topP: 0.95,
      maxTokens: 1024,
    },
    stream: true,
    getCompletion: (reasoning, text) => parseCompletion(text)
  },
  problems: {
    src: "../beans001/datasets/beans000_train.json",
    problemSchema: ChessGameSampleSchema,
    rewards: [beans001RewardFunc],
    getPrompt: async ({ prompt }) => prompt,
    formatSaveData(prompt, completion) {
      return {
        prompt: prompt.replace(/\n\n<rules>.*<\/rules>/s, ""),
        completion: completion,
      };
    },
  },
};

// deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export const testConfig: Config<typeof ChessGameSampleSchema> = {
  name: "local_test",
  llm: {
    model: vllm("/mnt/nas/lambda/sshkeda/ml/LLaMA-Factory/models/beans001s-1.5B"),
    stream: true,
    maxRunning: 100,
    params: {
      temperature: 1,
      topP: 0.95,
      maxTokens: 1600,
    },
    getCompletion: (_, text) => `<think>\n${text}`
  },
  problems: {
    src: "../beans001/datasets/beans001_test.json",
    problemSchema: ChessGameSampleSchema,
    rewards: [beans001RewardFunc],
    getPrompt: async (problem) => {
      return problem.prompt.replace(/\n\n<rules>.*<\/rules>/s, "");
    },
  },
};
