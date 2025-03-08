<p align="left">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/63d8d8879dfcfa941d4d7cd9/GsQKdaTyn2FFx_cZvVHk3.png" alt="Logo">
</p>

# EraX: Reimagine LLaMA 3.1 with DeepSeek's Innovation!

At EraX, curiosity drives us. We've taken the groundbreaking LLaMA 3.1 8B model and engineered a revolutionary transformation, selectively integrating DeepSeek R1's cutting-edge Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE) layers.

We're excited to share the code and raw model – refined with insights from Claude Sonnet 3.7 – enabling you to:

*   **Transform LLaMA 3.1:** Seamlessly convert specific layers of LLaMA 3.1-8B to DeepSeek R1's advanced MLA and MoE architecture.

    * For **MLA**, we picked layers 8, 10, 12, 14, 16, 18, 20, 22, 24. This alternate Layers approach converting every other layer offers several advantages:
          - Gradient stability: Alternating patterns help maintain more stable gradient flow
          - Architectural diversity: Provides a mix of processing mechanisms that can be complementary
          - Lower implementation risk: More conservative approach that preserves some of the original architecture
          - Easier performance isolation: Simplifies attribution of performance changes to specific modifications
        
    * For **MoE** we picked layers 11, 15, 19, 23 with the rationale:
          - Early enough to influence mid-network processing
          - Located at a critical middle point in the reasoning chain
          - Can specialize in different reasoning approaches after significant attention refinement
          - Late enough to influence final output generation
          - Provides expert routing for response formatting after all MLA processing is complete

*   **Experience the Future:** Reload and rigorously test the newly architected model, unlocking its potential. Use standard 🤗 Transformers without any PR required.
  
*   **Unlock New Frontiers:** Leverage our continual pretraining code, powered by FSDP (or DDP for BitAndBytes 8-bit optimization), to push the boundaries of model performance.  You will need to continual pretrain the new model with 25G - 40G multi-lingual multi-domain corpus and some 100k finetuning (or distiling from DeepSeek R1), plus some serious GRPO to make use the full power of this new model and retain most of LLaMA-3.1 8B's world knowledge.

*   **Strategy of cloning weights from original LLaMA-3.1 *B layers:**: 
    * **MLA Weight Transfer Strategy**: When transferring weights from standard LLaMA attention to MLA layers (_transfer_attention_to_mla function):   
      - Standard Attention Projections: The original query, key, value, and output projections are copied directly. This preserves the original attention mechanism's learned representations.
      - Latent Query Projection: Initialized with small random values. This projection is unique to MLA with no direct correspondence in LLaMA, so random initialization allows learning from scratch.
      - Latent Key Projection also initialized with small random values. The small scale (0.02) prevents disruption of initial forward passes while allowing gradual specialization.
      - Latent Value Projection. Attempts to reuse part of the original value projection when dimensions allow, Rreusing a subset of the original value projection leverages learned representational capabilities while adapting to the latent structure. Fallback to random initialization if dimensions don't match.
      - Latent Output Projection: Similar to value projection, attempts to reuse part of original output projection. This approach maintains the model's ability to project back to the original hidden state space.

  * **MoE Weight Transfer Strategy**:  When transferring weights from standard LLaMA MLP to MoE layers (_transfer_mlp_to_moe function):
        - Router Initialization: Initialized with very small random values (smaller than MLA initializations). The smaller scale (0.01) ensures initially more uniform routing to experts during early training.
        - First Expert Initialization: The first expert receives weights from the original MLP. This means Expert 0 starts with the same behavior as the original LLaMA MLP, maintaining baseline performance.
        - Other Experts Initialization: Remaining experts start as slightly perturbed versions of Expert 0. This approach ensures all experts start with similar capabilities but can specialize during training. The small noise factor (0.01) provides unique starting points for specialization.

  * **Dimension Safety Checks:** All transfers include dimension checks to handle potential mismatches:

## The Challenge: Unleashing the Power

While we've built the foundation, we need the resources to truly unleash this hybrid model's capabilities. Specifically, we're seeking funding – GPU compute – to embark on the crucial stages of pretraining and fine-tuning.

## The Vision: Open-Source AI for All

We believe in democratizing AI. If you're in a position to contribute compute resources and help us train this converted model, we implore you to do so. More importantly, we pledge to open-source the fully trained model on Hugging Face so the entire community can benefit from its innovation.

Join us in shaping the future of AI. Let's transform LLaMA 3.1 together!

**You can access the raw converted model at: [EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw](https://huggingface.co/erax-ai/EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw)**

Good luck, and we look forward to seeing what you create!

The EraX Team.

## Citation 📝
<!-- title={EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw: Reimagine LLaMA 3.1 with DeepSeek's Innovation!},
  author={Nguyễn Anh Nguyên},
  organization={EraX},
  year={2025},
  url={https://huggingface.co/erax-ai/EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw},
  github={https://github.com/EraX-JS-Company/LLaMA3.1-8B-DeepSeekR1-MLA-MoE/tree/main} -->
  
If you find our project useful, we would appreciate it if you could star our repository and cite our work as follows:
```
@article{EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw,
  title={EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw: Reimagine LLaMA 3.1 with DeepSeek's Innovation!},
  author={Nguyễn Anh Nguyên - nguyen@erax.ai},
  organization={EraX},
  year={2025},
  url={https://huggingface.co/erax-ai/EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw},
  github={https://github.com/EraX-JS-Company/LLaMA3.1-8B-DeepSeekR1-MLA-MoE}
}
```
