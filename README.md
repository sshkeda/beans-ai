Today, chess engines like Stockfish are incredibly powerful.

They can beat every human on earth with little to no effort.

And while they are so strong, there's one thing that sets them apart from the best grandmasters.

They can’t explain their thinking process.

But imagine if they could.

Could we:

- Offer free world-class chess tutoring to everyone?
- Discover new, intuitive strategies?
- Even beat Stockfish at its own game?

To explore these possibilities, I want to introduce:

# Beans AI

The goal: create the world's best chess tutor.

How? Reinforcement learning and fine-tuning LLMs.

## beans-0-1.5B

Before becoming a grandmaster, one must first learn to make legal moves.

If you give a [1.5B parameter](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) LLM a chess puzzle, it only outputs legal moves 4% of the time.

I wanted to see if I could improve this.

So, I prompted DeepSeek R1 to solve several thousand chess puzzles. Then, I collected the responses that contained legal moves and created a [dataset](https://huggingface.co/datasets/sshkeda/beans-0-dataset.json) of 2,159 reasoning examples.

Next, I fine-tuned DeepSeek-R1-Distill-Qwen-1.5B on the dataset using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

The result: [beans-0-1.5B](https://huggingface.co/sshkeda/beans-0-1.5B).

## Evals

To test how often beans-0-1.5B outputs legal moves, I randomly selected 100 chess puzzles from the [EleutherAI/lichess-puzzles dataset](https://huggingface.co/datasets/EleutherAI/lichess-puzzles) and prompted beans to solve each puzzle.

Answers were scored as follows:

- **1:** Generated move is legal and exists on the board.
- **0:** Generated move is illegal or doesn't exist on the board.
- **-1:** Invalid format (missing <answer></answer> tags).

## Results

| Model                         | Parsing Failures (-1) | Invalid Moves (0) | Legal Moves (1) | Accuracy | Expected value |
| ----------------------------- | --------------------- | ----------------- | --------------- | -------- | -------------- |
| **beans-0-1.5B**              | **62**                | **16**            | **22**          | **0.22** | **-0.40**      |
| DeepSeek-R1-Distill-Qwen-1.5B | 75                    | 21                | 4               | 0.04     | -0.71          |

beans-0-1.5B showed a clear improvement over the baseline DeepSeek-R1-Distill-Qwen-1.5B in generating legal chess moves.

## Acknowledgements

Special thanks to:

- Startup Shell for providing access to two NVIDIA 3090 GPUs.
- Bobby George for setting up SSH access to these GPUs.
- DeepSeek for their open-source AI models.
