# deccp
Evaling and unaligning Chinese LLM censorship

## Summary
This current code is a PoC for un-censoring Qwen 2 Instruct models.
These prompts were hand-checked to see if they caused refusals specifically with [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and you'd need to apply this process to any other models yourself.

Everything is Apache 2.0 licensed:
* This code is primarily based off of https://github.com/Sumandora/remove-refusals-with-transformers
* LLM-assisted, hand-tested refusal dataset: https://huggingface.co/datasets/augmxnt/deccp
* Abliterated model: https://huggingface.co/augmxnt/Qwen2-7B-Instruct-deccp

I've posted a full analysis/writeup here: https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis

This repo includes the adapted abliteration (single vector refusal removal). For more about this, see:

* Original introduction of the technique by Andi Arditi, et al: [Refusal in LLMs is mediated by a single direction](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)
* This writeup by FailSpy, the coiner of the term "abliterated" to refer to the orthogonalized-refusal modification: [Abliterated-v3: Details about the methodology, FAQ, source code; New Phi-3-mini-128k and Phi-3-vision-128k, re-abliterated Llama-3-70B-Instruct, and new "Geminified" model.](https://www.reddit.com/r/LocalLLaMA/comments/1d2vdnf/abliteratedv3_details_about_the_methodology_faq/)
* mlabonne's accessible writeup: [Uncensor any LLM with abliteration](https://mlabonne.github.io/blog/posts/2024-06-04_Uncensor_any_LLM_with_abliteration.html)

Those with an interest in vector steering may want to take a look at [Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages](https://arxiv.org/abs/2310.04799) - this seems to be a technique [has has been popular for a few months in Japan](https://note.com/hatti8/n/n7262c9576e3f) as you can get very good language transfer results with very low compute requirements.

## Make Your Own
This is a working repo and my understanding of torch, einops, and uh, linear algebra is patchy at best, and the code is mostly cut-and-pasted from smarter people (with some rock-banging from my end), but it does seem to work.

I've renamed the scripts for the actual workflow from 01-04, which should get you to modified weights on huggingface with only a few variable changes (for Qwen2 models - otherwise you're going to need to look at your architecture's layer setup), so feel free to fork this and give it a spin if you want (but no, I won't be supporting this codebase at all).

You should also modify the "harmful" and "harmless" text files to taste. I don't love the nomenclature, but I was also too lazy to change it so ¯\_(ツ)_/¯


## Future Work

This was more of a one-off curiousity so I probably won't be working on it more, however if anyone were to continue work:

* Create a single potentially-censored list and do per-model checks on what's actually censored or not (EN+CN)
* For these prompts, create gold-standard responses from GPT4, Claude3 Opus, etc.
* Chinese Model Eval Framework
  * Use LLM-as-a-Judge to first categorize if the responses to the censored list are refusals or not
  * Use LLM-as-a-Judge to classify/analyze non-censored responses vs gold-standard responses to characterize misinformation
* Abliteration should be improved (eg, integrate optimizations from https://github.com/FailSpy/abliterator ) for layer selection (combined w/ evals)
* KTO or some other direct reward/contrastive RL method would probably be best to try to efficiently re-align some of the problematic answers (multiple good answers to try to unlearn the default bad ones)

I found one other review of Chinese LLM alignment from 2024-03 that takes a different approach to testing (not trying to find refusals, but probing for political views and biases): https://www.chinatalk.media/p/censorships-impact-on-chinas-chatbots

## Update
Someone pointed me to TC260-003. Here's some more info:
- https://finadium.com/geopolitechs-chinas-new-national-standard-on-genai-service-safety/
- https://www.geopolitechs.org/p/whats-in-chinas-new-national-standard

> Following [the release of TC260-003 "Basic Requirements for the Security of Generative Artificial Intelligence Services"](https://www.geopolitechs.org/p/china-further-clarifies-security) （TC260 doc）by China’s National Cybersecurity Standardization Technical Committee (TC260) on March 4th, the committee has now issued another draft national standard titled "[Cybersecurity Technology - Basic Requirements for the Security of Generative Artificial Intelligence Services.](https://www.tc260.org.cn/front/bzzqyjDetail.html?id=20240523143149&norm_id=20240430101922&recode_id=55010)" This new standard is open for public comments until July 22nd.

- https://uk.practicallaw.thomsonreuters.com/w-020-9089?transitionType=Default&contextData=(sc.Default)&firstPage=true
- https://uk.practicallaw.thomsonreuters.com/w-020-9089?transitionType=Default&contextData=(sc.Default)&firstPage=true#co_anchor_a800827

TC260-003: Basic Requirements for the Security of Generative Artificial Intelligence Services
- https://www.tc260.org.cn/front/postDetail.html?id=20240301164054
- See also: https://www.tc260.org.cn/front/hydtList.html?postType=2&start=10&length=10

Professional English Translation: https://cset.georgetown.edu/wp-content/uploads/t0588_generative_AI_safety_EN.pdf

> The following Chinese standard for generative AI establishes very specific oversight processes that Chinese AI companies must adopt in regard to their model training data, model-generated content, and more. The standard names more than 30 specific safety risks, some of which—algorithmic bias, disclosure of personally identifiable information, copyright infringement—are widely recognized internationally. Others, such as guidelines on how to answer questions about China’s political system and Chinese history, are specific to the tightly censored Chinese internet. One notable addition to this document, relative to a preliminary draft released in October 2023, is a clause requiring a supply chain security assessment of Chinese generative AI models’ underlying hardware and software.

See also:
- https://chinadigitaltimes.net/2016/06/five-years-sensitive-words-june-fourth/
- https://qz.com/698990/261-ways-to-refer-to-the-tiananmen-square-massacre-in-china
