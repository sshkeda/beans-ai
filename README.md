# beans-ai (alpha)

If LLMs are getting so good at math and coding, why haven't they become chess grandmasters yet?

Introducing [beans-0-1.5B](https://huggingface.co/sshkeda/beans-0-1.5B)—an AI trained to reason about chess. 

## beans-0-1.5B 

The ultimate goal of beans-ai is to master chess, but before doing so, it must learn to make legal chess moves consistently.

Currently, DeepSeek-R1-Distill-Qwen-1.5B generates legal moves 4% of the time. But I wanted to see if I could improve that.

So, I prompted DeepSeek R1 to generate the next best move for several thousand chess puzzles and used the generations that produced a legal next move to create a [dataset (2,159 samples)](https://huggingface.co/datasets/sshkeda/beans-0-dataset.json) to fine-tune DeepSeek-R1-Distill-Qwen-1.5B. And thus, beans-0-1.5B was born.

## Evals

To assess how well beans-0-1.5B generates legal moves, I randomly sampled 100 chess puzzles from the [EleutherAI/lichess-puzzles dataset](https://huggingface.co/datasets/EleutherAI/lichess-puzzles) and prompted beans to generate the next best move.

Answers were evaluated as follows:

- **1:** The move exists on the board.
- **0:** The move doesn't exist on the board.
- **-1:** Invalid format (no \<answer>\</answer>).

## Results

| Model                               | Parsing Failures (-1) | Invalid Moves (0) | Valid Moves (1) | Accuracy | Expected value |
|-------------------------------------|-----------------------|-------------------|-----------------|----------|----------------|
| **beans-0-1.5B**                    | **62**                | **16**            | **22**          | **0.22** | **-0.40**      |
| DeepSeek-R1-Distill-Qwen-1.5B       | 75                    | 21                | 4               | 0.40     | -0.71          |

beans-0-1.5B demonstrated significant improvement over its baseline DeepSeek-R1-Distill-Qwen-1.5B at generating valid moves!

## Future:

- Build a reward function to evaluate chess move quality.
- Implement GRPO for RL training.
- Scale training data with >> 2,159 samples.
- Fine-tune using only good moves.
- Explore self-improvement via synthetic data generation.
