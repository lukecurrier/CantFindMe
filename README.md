# Can't Find Me - Obscuring LLM Traces
Created as a final project for CS6973 Trustworthy Generative AI. Paper can be found <a href="https://lukecurrier.github.io/assets/files/cantfindme.pdf" target=None>here</a>.

The Hide and Seek approach uses a two-model iterative learning process to create prompts that identify the model family of the LLM they're querying from and subsequently analyze them against other generations. These two models are known as the 'Auditor' and the 'Detective'.

The experiment is conducted over T trials:
1. The Auditor generates an initial set of prompts.
2. These prompts are presented to N different LLMs (including two from the same source).
3. The Detective analyzes the outputs and attempts to identify the two similar models.
4. The Results block is provided to the Auditor.
5. Steps 2-4 are repeated for T trials.

To account for the Auditor’s learning curve, there is a warm-up period of W trials. The Auditor’s performance can be analyzed once it has had the opportunity to refine its prompt generation strategy based on feedback.

## Quickstart
source trust/bin/activate

pip install -r utils/requirements.txt

export TOGETHER_API_KEY="insert_key_here"

python -m algo_helpers.adversarial_helpers --save_response --num_trials 10 --models_file utils/models.yaml --output_path ./results

