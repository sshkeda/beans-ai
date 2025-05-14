import type { JSONValue, LanguageModelV1 } from "ai";
import { existsSync } from "node:fs";
import { z } from "zod";


export type Config<T extends z.ZodTypeAny> = {
  name: string;
  llm: {
    model: LanguageModelV1;
    providerOptions?: Record<string, Record<string, JSONValue>>
    params: {
      temperature: number;
      topP: number;
      maxTokens: number;
    };
    stream: boolean;
    maxRunning?: number;
    maxRetries?: number;
    getCompletion: (reasoning?: string, text?: string) => string;
  };
  problems: {
    src: string;
    count?: number;
    shuffle?: boolean;
    problemSchema: T;
    getPrompt: (
      ctx: z.infer<T>
    ) => Promise<string>;
    rewards: ((
      prompt: string,
      completion: string,
      problem: z.infer<T>
    ) => Promise<number> | number)[];
    formatSaveData?: (
      prompt: string,
      completion: string
    ) => {
      prompt: string;
      completion: string;
    };
  };
};

export async function getOutputDir<T extends z.ZodTypeAny>(config: Config<T>) {
  const name = process.argv[3];
  if (typeof name !== "string") throw new Error("Name argument is required");
  const outputDir = `output/${config.name}/${name}/`;
  if (existsSync(outputDir))
    throw new Error(`Output directory ${name} already exists`);
  return outputDir;
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 5,
  retryDelay = 30000
) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      console.error(error);
      if (i === maxRetries - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
    }
  }

  throw new Error("Failed to execute function");
}

export function getChosenConfig<T extends z.ZodTypeAny>(configs: Config<T>[]) {
  const chosenConfig = configs.find(
    (config) => config.name === process.argv[2]
  );
  if (!chosenConfig) throw new Error("Invalid config");
  return chosenConfig;
}
