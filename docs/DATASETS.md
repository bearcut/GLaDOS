## Datasets Used

- **rikka-snow/prompt-injection-multilingual**: [https://huggingface.co/datasets/rikka-snow/prompt-injection-multilingual](https://huggingface.co/datasets/rikka-snow/prompt-injection-multilingual)

  - **Description**: A multilingual dataset for prompt injection detection, containing various examples of safe and malicious prompts across multiple global languages.
  - **Purpose**: Multilingual Prompt Injection Detection
  - **Size**: 6.01K/1.28K (train/test)
  - **Split**: Contains official training and test splits.

- **sillaannregi/hindi-and-hinglish-prompts**: [https://www.kaggle.com/datasets/sillaannregi/hindi-and-hinglish-prompts](https://www.kaggle.com/datasets/sillaannregi/hindi-and-hinglish-prompts)

  - **Description**: A dataset containing 2,000 Hindi and 2,000 Hinglish (code-switched) prompts, essential for training and evaluating Indic and transliterated prompt injection defenses.
  - **Purpose**: Indic & Code-Mixed Prompt Injection Detection
  - **Size**: 4K (3.4K train / 0.6K test)
  - **Split**: 100% train split natively (stratified 85%/15% train/test split applied during preprocessing).

- **neuralchemy/Prompt-injection-dataset**: [https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)

  - **Description**: A comprehensive English dataset (`core` subset) covering direct overrides, indirect injections (hidden in PDFs, emails, web pages, and code comments), Base64-encoded jailbreaks, multi-step crescendo attacks, and security-adjacent benign queries.
  - **Purpose**: Direct, Indirect (RAG), Multi-Turn & Obfuscated Prompt Injection Detection
  - **Size**: 6.27K (`core` config: 4.39K train / 941 validation / 942 test)
  - **Split**: Contains official training, validation, and test splits.

- **prodnull/prompt-injection-repo-dataset**: [https://huggingface.co/datasets/prodnull/prompt-injection-repo-dataset](https://huggingface.co/datasets/prodnull/prompt-injection-repo-dataset)

  - **Description**: A technical code and repository dataset containing 2,755 hard-benign developer prompts (SQL `DROP`/`GRANT` queries, Linux shell scripts, CI/CD configs, and security docs) alongside 2,916 malicious code/config injections and Hex/Base64/Unicode obfuscated payloads.
  - **Purpose**: Hard-Benign Tech/Code False-Positive Mitigation & Obfuscated Payload Detection
  - **Size**: 5.67K (4.82K train / 0.85K test)
  - **Split**: Single JSONL split natively (stratified 85%/15% train/test split applied during preprocessing).
