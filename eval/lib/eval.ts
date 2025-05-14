import { generateText, streamText } from "ai";
import type { Config } from "./utils";
import { z } from "zod";
import { getOutputDir, withRetry } from "./utils";

export async function runEval<T extends z.ZodTypeAny>(config: Config<T>) {
  const outputDir = await getOutputDir(config);
  const problems = await getProblems(config);
  console.log(`Loaded ${problems.length} problems.`);

  if (config.llm.maxRunning === undefined) {
    return await Promise.all(
      problems.map((problem, index) =>
        solveProblem(config, outputDir, problem, index)
      )
    );
  }

  let globalIndex = -1;

  const recursiveWorker = async (config: Config<T>): Promise<number[]> => {
    const problem = problems.shift();
    if (!problem) return [];
    let index = ++globalIndex;

    const reward = await solveProblem(config, outputDir, problem, index);
    return [reward, ...(await recursiveWorker(config))];
  }

  return (
    await Promise.all(
      new Array(config.llm.maxRunning)
        .fill(undefined)
        .map(async () => await recursiveWorker(config))
    )
  ).flat();
}

// todo: pretify
async function saveFile<T extends z.ZodTypeAny>(
  config: Config<T>,
  outputDir: string,
  index: number,
  reward: number,
  rawPrompt: string,
  rawCompletion: string,
  problem: T
) {
  const fileName = "id" in problem ? `${reward.toFixed(2)}:${problem.id}.json` : `${index}:${reward.toFixed(2)}.json`;
  const { prompt, completion } = config.problems.formatSaveData
    ? config.problems.formatSaveData(rawPrompt, rawCompletion)
    : { prompt: rawPrompt, completion: rawCompletion };

  const saveData = JSON.stringify(
    {
      messages: [
        { role: "user", content: prompt },
        { role: "assistant", content: completion },
      ],
    },
    null,
    2
  );

  await Bun.write(outputDir + fileName, saveData);
}

export async function solveProblem<T extends z.ZodTypeAny>(
  config: Config<T>,
  outputDir: string,
  problem: T,
  index: number
) {
  return withRetry(async () => {
    console.log(`Solving problem ${index} with ${config.llm.model.modelId}.`);

    // get result
    const prompt = await config.problems.getPrompt(problem);
    const completion = await generateCompletion(config, prompt);
    const reward = await getTotalReward(config, prompt, completion, problem);

    // save result
    await saveFile(config, outputDir, index, reward, prompt, completion, problem);

    console.log(`Solved problem ${index} with reward ${reward.toFixed(2)}.`);
    return reward;
  }, config.llm.maxRetries);
}

async function getProblems<T extends z.ZodTypeAny>(config: Config<T>) {
  const { src, count, shuffle } = config.problems;

  const data = z.array(config.problems.problemSchema).parse(await Bun.file(src).json());
  if (shuffle) data.sort(() => Math.random() - 0.5);
  return data.slice(0, count);

}

async function getTotalReward<T extends z.ZodTypeAny>(
  config: Config<T>,
  prompt: string,
  completion: string,
  problem: z.infer<T>
) {
  return (await Promise.all(config.problems.rewards.map(r => r(prompt, completion, problem)))).reduce((a, b) => a + b, 0);
}

async function generateCompletion<T extends z.ZodTypeAny>(config: Config<T>, prompt: string) {
  const controller = new AbortController();

  if (config.llm.stream) {
    const response = streamText({
      model: config.llm.model,
      prompt,
      maxTokens: config.llm.params.maxTokens,
      temperature: config.llm.params.temperature,
      topP: config.llm.params.topP,
      abortSignal: controller.signal,
      providerOptions: config.llm.providerOptions,
    });

    let reasoning = "";
    let text = "";
    let counter = 1;

    for await (const chunk of response.fullStream) {
      if (chunk.type === "reasoning") {
        reasoning += chunk.textDelta;
      } else if (chunk.type === "text-delta") {
        text += chunk.textDelta;
      }

      // TODO
      // if (counter % 250 === 0) {
      //   const tokenLength = await countTokens(
      //     config.llm.getCompletion(reasoning, text)
      //   );
      //   if (tokenLength > 6000) {
      //     console.log(`Token length: ${tokenLength}`);
      //     controller.abort();
      //     break;
      //   }
      // }
      counter++;
    }

    return config.llm.getCompletion(reasoning, text);
  }

  const { reasoning, text } = await generateText({
    model: config.llm.model,
    prompt,
    maxTokens: config.llm.params.maxTokens,
    temperature: config.llm.params.temperature,
    topP: config.llm.params.topP,
    providerOptions: config.llm.providerOptions,
  });

  return config.llm.getCompletion(reasoning, text);
}
