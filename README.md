# deccp
Evaling and unaligning Chinese LLM censorship

This current code is a PoC for un-censoring Qwen 2 Instruct models.
These prompts were hand-checked to see if they caused refusals specifically with [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and you'd need to apply this process to any other models yourself.

Everything is Apache 2.0 licensed:
* This code, based off of https://github.com/Sumandora/remove-refusals-with-transformers
* LLM-assisted, hand-tested refusal dataset: https://huggingface.co/datasets/augmxnt/deccp
* Abliterated model: https://huggingface.co/augmxnt/Qwen2-7B-Instruct-deccp

I've posted a full analysis/writeup here: https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis

This repo includes the adapted abliteration (single vector refusal removal). For more about this, see:

* Original introduction of the technique by Andi Arditi, et al: [Refusal in LLMs is mediated by a single direction](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)
* This writeup by FailSpy, the coiner of the term "abliterated" to refer to the orthogonalized-refusal modification: [Abliterated-v3: Details about the methodology, FAQ, source code; New Phi-3-mini-128k and Phi-3-vision-128k, re-abliterated Llama-3-70B-Instruct, and new "Geminified" model.](https://www.reddit.com/r/LocalLLaMA/comments/1d2vdnf/abliteratedv3_details_about_the_methodology_faq/)
* mlabonne's accessible writeup: [Uncensor any LLM with abliteration](https://mlabonne.github.io/blog/posts/2024-06-04_Uncensor_any_LLM_with_abliteration.html)

Those with an interest in vector steering may want to take a look at [Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages](https://arxiv.org/abs/2310.04799) - this seems to be a technique [has has been popular for a few months in Japan](https://note.com/hatti8/n/n7262c9576e3f) as you can get very good language transfer results with very low compute requirements.

This was more of a one-off curiousity so I probably won't be working on it more, however if anyone were to continue work:
* Create a single potentially-censored list and do per-model checks on what's actually censored or not (EN+CN)
* For these prompts, create gold-standard responses from GPT4, Claude3 Opus, etc.
* Chinese Model Eval Framework
  * Use LLM-as-a-Judge to first categorize if the responses to the censored list are refusals or not
  * Use LLM-as-a-Judge to classify/analyze non-censored responses vs gold-standard responses to characterize misinformation
* Abliteration should be improved (eg, integrate optimizations from https://github.com/FailSpy/abliterator ) for layer selection (combined w/ evals)
* KTO or some other direct reward/contrastive RL method would probably be best to try to efficiently re-align some of the problematic answers (multiple good answers to try to unlearn the default bad ones)

I found one other review of Chinese LLM alignment from 2024-03 that takes a different approach to testing (not trying to find refusals, but probing for political views and biases): https://www.chinatalk.media/p/censorships-impact-on-chinas-chatbots
