from llmlingua import PromptCompressor
import torch


prompt = "Imagine you are observing a sunset on a serene beach at dusk. The sky is a tapestry of orange, pink, and purple hues blending into the horizon. The waves gently caress the shore, their melody accompanying the symphony of seabirds returning to roost. Capture this moment in a poem that evokes a range of emotions - tranquility, nostalgia, and a sense of infinite possibility. The poem should resonate with the essence of human experience, evoking memories and provoking introspection."

print(prompt) 

llm_lingua = PromptCompressor("microsoft/phi-2", device_map="mps")
compressed_prompt = llm_lingua.compress_prompt(prompt, rate=0.8)

print(compressed_prompt)
