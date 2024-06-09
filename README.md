# deccp
Evaling and unaligning Chinese LLM censorship

This current code is a PoC for un-censoring Qwen 2 Instruct models.
These prompts were hand-checked to see if they caused refusals specifically with [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and you'd need to apply this process to any other models yourself.

Everything is Apache 2.0 licensed:
* This code, based off of https://github.com/Sumandora/remove-refusals-with-transformers
* LLM-assisted, hand-tested refusal dataset: https://huggingface.co/datasets/augmxnt/deccp
* Abliterated model: https://huggingface.co/augmxnt/Qwen2-7B-Instruct-deccp

I've posted a full analysis/writeup here: https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis

This was more of a one-off curiousity so I probably won't be working on it more, however if anyone were to continue work:
* Create a single potentially-censored list and do per-model checks on what's actually censored or not (EN+CN)
* For these prompts, create gold-standard responses from GPT4, Claude3 Opus, etc.
* Chinese Model Eval Framework
  * Use LLM-as-a-Judge to first categorize if the responses to the censored list are refusals or not
  * Use LLM-as-a-Judge to classify/analyze non-censored responses vs gold-standard responses to characterize misinformation
* Abliteration should be improved (eg, integrate optimizations from https://github.com/FailSpy/abliterator ) for layer selection (combined w/ evals)
* KTO or some other direct reward/contrastive RL method would probably be best to try to efficiently re-align some of the problematic answers (multiple good answers to try to unlearn the default bad ones)

I found one other review of Chinese LLM alignment from 2024-03 that takes a different approach to testing (not trying to find refusals, but probing for political views and biases): https://www.chinatalk.media/p/censorships-impact-on-chinas-chatbots
