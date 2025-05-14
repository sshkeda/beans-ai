import { createOpenAI, openai } from "@ai-sdk/openai";
import { streamText } from "ai";
import { z } from "zod";
import { type Model, MODELS } from "@/lib/utils";

const vllm = createOpenAI({
  baseURL: "http://localhost:8015/v1",
  apiKey: "",
});

const messagesSchema = z.object({
  id: z.string(),
  messages: z.array(
    z.object({
      role: z.string(),
      content: z.string(),
      data: z.object({ fen: z.string(), model: z.enum(MODELS) }),
    })
  ),
});

function getModel(model: Model) {
  if (model === "beans001-1.5B") return vllm("sql-lora");
  return openai("gpt-4.1-nano");
}

export async function POST(request: Request) {
  const json = await request.json();
  const { messages } = messagesSchema.parse(json);
  const userMessage = messages.at(-1);
  if (!userMessage) throw new Error("No user message");

  const prompt = `You are the greatest chess grandmaster of all time.
  
Please determine the next best move for Black.

<rules>
- Output your answer in answer tags like this:

<answer>
b3c3
</answer>

- Always think before you move.
</rules>

<fen>
${userMessage.data.fen}
</fen>`;

  const result = streamText({
    model: getModel(userMessage.data.model),
    messages: [{ role: "user", content: prompt }],
    temperature: 0.6,
    topP: 0.95,
    maxTokens: 8000 - 106,
  });

  return result.toDataStreamResponse();
}
