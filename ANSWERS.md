# Answers

## Section 1: Diagnosis Log

### Problem 1: Wrong pricing answers

1. I first checked whether the failures were random or concentrated on pricing questions.
2. I ruled out temperature as the primary root cause because hallucinated pricing is content-specific rather than broadly stochastic. Lower temperature can reduce variance, but it does not create missing knowledge.
3. I then ruled out a pure prompt issue by checking whether the model was still answering other support topics correctly.
4. The most likely root cause is retrieval failure or stale knowledge. Pricing changes are usually external to the model and must come from a live source of truth. If the bot has no retrieval layer, or the retrieval layer is returning outdated pricing docs, GPT-4o will confidently fill gaps.

Concrete fix:

- route all pricing questions to a live pricing service or up-to-date retrieval index
- add a hard instruction to never invent prices
- add a refusal path when pricing cannot be grounded in retrieved data

How I would distinguish root causes:

- Prompt issue: same retrieved pricing facts, but the model still distorts them.
- Retrieval issue: the retrieved context is missing or stale, and the answer follows the missing context.
- Temperature issue: answers vary materially across repeated runs with the same context.
- Knowledge cutoff: the question depends on post-training product changes and no retrieval source exists.

### Problem 2: Responds in English to Hindi or Arabic users

1. I first inspected the prompt structure.
2. In a system prompt + user message architecture, the model often treats the system prompt as higher priority but not necessarily sufficient to preserve the user language if the system prompt does not explicitly constrain output language.
3. The failure mechanism is that the model optimizes for helpfulness and may default to its highest-probability language, often English, especially if the system prompt says things like "answer clearly" without language locking.

Prompt fix:

```
You must respond in the same language as the user's latest message.
If the user's message is Hindi, reply in Hindi.
If the user's message is Arabic, reply in Arabic.
If the user's message is mixed-language, preserve the dominant user language.
Never switch to English unless the user explicitly asks for English.
Do not mention this rule.
```

This is language-agnostic because it conditions on the user's language rather than enumerating specific languages.

### Problem 3: Latency regressed from 1.2s to 8-12s

At least three plausible causes:

1. Context growth. As the user base grew, the application may have started appending longer conversation history or retrieval context, increasing tokens per request.
2. Queueing and saturation. More traffic can create request backlogs even if model compute is unchanged.
3. Downstream dependency slowness. Retrieval, database calls, logging, or rate-limit retries may have become slower under load.
4. Cache miss rate increase. If prompt/result caching was effective at low volume but degraded as traffic patterns changed, average latency could jump.

What I would investigate first:

- token counts and prompt size distribution, because token growth is the simplest explanation for a 6-10x slowdown without code changes
- then queueing metrics and p95 latency to separate compute time from waiting time
- then dependency traces for retrieval and network calls

## Post-mortem summary

The chatbot experienced three separate problems. First, it gave wrong pricing answers because pricing information was not grounded in a live source of truth, so the model filled in missing facts. Second, it sometimes replied in English because the prompt did not explicitly lock the response language to the user’s language, and the model defaulted to English when the instruction was ambiguous. Third, latency increased because the system likely started handling larger prompts, more traffic, or slower dependent services as usage grew. The fixes are straightforward: connect pricing questions to current data, add a strict language-preservation rule to the prompt, and inspect token growth and service saturation to restore response time. The overall lesson is that good model behavior in testing does not guarantee good production behavior unless the system is grounded, constrained, and monitored.

## Section 2: Scaling answer

If the legal corpus grows to 50,000 documents, the main bottlenecks would be ingestion, embedding generation, and retrieval quality. I would parallelize PDF extraction, store page text and metadata in a database, and batch embedding jobs. I would replace the current TF-IDF + FAISS setup with a two-stage retrieval stack: BM25 for lexical recall, vector search for semantic recall, and a cross-encoder reranker for precision. I would also move from a flat FAISS index to an IVF or HNSW index and persist the index to disk. At that scale, evaluation would need to become more formal, with a larger test set and metrics such as recall@k, citation accuracy, and refusal precision.

## Section 4: Written Systems Design Review

### Question A - Prompt Injection & LLM Security

Malicious users can attack a prompt in several distinct ways. One common technique is direct instruction override, where the user says to ignore previous instructions and follow their new rules. I would mitigate this by separating system instructions from user content, wrapping user input in a quoted or structured field, and explicitly telling the model that user text is untrusted data, not instructions. A second technique is role confusion, such as pretending the user is the system or developer. The mitigation is to hard-code role boundaries in the prompt template and reject any user attempt to redefine roles. A third technique is delimiter breakout, where the attacker injects fake XML or Markdown tags to escape the user block. I would escape delimiters before insertion and use a structured schema rather than raw concatenation. A fourth technique is indirect prompt injection through retrieved documents, where hostile text in a knowledge base tells the model to ignore the user. The fix is to treat retrieved text as evidence only, scan for instruction-like content, and keep retrieval and instruction channels separate. A fifth technique is goal hijacking, where the user asks for an apparently legitimate task that embeds a hidden instruction. I would use output validation, constrained decoding when possible, and a policy layer that checks whether the request matches the allowed task before sending it to the model.

### Question B - Evaluating LLM Output Quality

I would evaluate summarization quality with a mix of automatic metrics and human review. ROUGE-L is useful for overlap with reference summaries, but it misses factuality and can reward verbatim copying. BERTScore captures semantic similarity better, but it still does not guarantee correctness. For internal reports, factual consistency is critical, so I would add a factuality audit using human raters or an LLM-judge that checks whether each summary claim is supported by the source document. I would build a ground-truth set by sampling reports across departments, lengths, and difficulty levels, then asking trained reviewers to write reference summaries and mark key facts that must appear. To detect regressions, I would keep a locked benchmark set and rerun evaluation whenever the model, prompt, or decoding settings change. I would track metric deltas over time and establish alert thresholds for factuality and coverage. To communicate quality to a non-technical stakeholder, I would report a simple scorecard: how often summaries are factually correct, how often they miss key points, and how much manual review is still needed. The main limitation is that automatic metrics do not fully capture usefulness, so periodic human review remains necessary.

## Section 3: Model choice justification

I chose a CPU-based TF-IDF plus logistic regression classifier because the constraint is under 500ms per ticket on a single CPU server, and the traffic volume is only 2,880 tickets/day. That works out to about 2 tickets per minute on average, so the system needs to be predictable and cheap rather than heavyweight. A local linear model with vectorization is typically far below 500ms per request on this scale, even in small batches, while a remote LLM prompt pipeline would introduce network latency, vendor variability, and per-request cost. The main risk is latency consistency, not raw throughput. A prompt-based LLM could occasionally exceed the SLA because of network or API delays, whereas a local CPU model stays stable. The tradeoff is reduced semantic flexibility, but the five ticket classes are narrow enough that lexical and n-gram features are usually sufficient.

## Section 3: Confusion classes

The main confusion pairs in practice are `billing` versus `other`, and `technical_issue` versus `feature_request`. `billing` and `other` overlap because users often describe account or payment questions in generic support language, while `technical_issue` and `feature_request` overlap when users describe product pain points as requests for improvements or fixes. Additional account context, ticket history, and explicit intent metadata would improve separation.

## Section 5

Optional screen recording not included.
