<p align="left">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/63d8d8879dfcfa941d4d7cd9/GsQKdaTyn2FFx_cZvVHk3.png" alt="Logo">
</p>

# EraX: Reimagine LLaMA 3.1 with DeepSeek's Innovation – and We Need Your Help!

At EraX, curiosity drives us. We've taken the groundbreaking LLaMA 3.1 8B model and engineered a revolutionary transformation, selectively integrating DeepSeek R1's cutting-edge Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE) layers.

We're excited to share the code and raw model – refined with insights from Claude Sonnet 3.7 – enabling you to:

*   **Transform LLaMA 3.1:** Seamlessly convert specific layers of LLaMA 3.1-8B to DeepSeek R1's advanced MLA and MoE architecture.
  *   For **MLA**, we picked layers [8, 10, 12, 14, 16, 18, 20, 22, 24. This alternate Layers approach converting every other layer offers several advantages:
        - Gradient stability: Alternating patterns help maintain more stable gradient flow
        - Architectural diversity: Provides a mix of processing mechanisms that can be complementary
        - Lower implementation risk: More conservative approach that preserves some of the original architecture
        - Easier performance isolation: Simplifies attribution of performance changes to specific modifications
        
  * For **MoE** we picked [11, 15, 19, 23] with the rationale:
        - Early enough to influence mid-network processing
        - Located at a critical middle point in the reasoning chain
        - Can specialize in different reasoning approaches after significant attention refinement
        - Late enough to influence final output generation
        - Provides expert routing for response formatting after all MLA processing is complete

*   **Experience the Future:** Reload and rigorously test the newly architected model, unlocking its potential.
  
*   **Unlock New Frontiers:** Leverage our continual pretraining code, powered by FSDP (or DDP for BitAndBytes 8-bit optimization), to push the boundaries of model performance.

## The Challenge: Unleashing the Power

While we've built the foundation, we need the resources to truly unleash this hybrid model's capabilities. Specifically, we're seeking funding – GPU compute – to embark on the crucial stages of pretraining and fine-tuning.

## The Vision: Open-Source AI for All

We believe in democratizing AI. If you're in a position to contribute compute resources and help us train this converted model, we implore you to do so. More importantly, we pledge to open-source the fully trained model on Hugging Face so the entire community can benefit from its innovation.

Join us in shaping the future of AI. Let's transform LLaMA 3.1 together!

**You can access the raw converted model at: [EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw]([https://github.com/EraX-JS-Company/LLaMA3.1-8B-DeepSeekR1-MLA-MoE](https://huggingface.co/erax-ai/EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw))**

Good luck, and we look forward to seeing what you create!

The EraX Team.
