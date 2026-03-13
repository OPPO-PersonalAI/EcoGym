<h1 align="center">
<br>
EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies
</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Preprint-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/HuggingFace-Dataset-ff9800?style=flat&logo=huggingface" alt="Hugging Face Dataset"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10-3776ab?style=flat&logo=python&logoColor=white" alt="Python 3.10"></a>
</p>

<p align="center">
  <strong>A Generalizable Benchmark for Continuous Plan-and-Execute Decision Making in Interactive Economies</strong>
</p>

## 🌟 Overview

Long-horizon planning is widely recognized as a core capability of autonomous LLM-based agents; however, current evaluation frameworks suffer from being largely episodic, domain-specific, or insufficiently grounded in persistent economic dynamics. We introduce **EcoGym**, a generalizable benchmark for continuous plan-and-execute decision making in interactive economies. 

EcoGym comprises three diverse environments: **Vending**, **Freelance**, and **Operation**, implemented in a unified decision-making process with standardized interfaces, and budgeted actions over an effectively unbounded horizon (1000+ steps if 365 day-loops for evaluation). The evaluation is based on business-relevant outcomes (e.g., net worth, income, and DAU), targeting long-term strategic coherence and robustness under partial observability and stochasticity. 

<div align="center">
  <img src="assets/overview.png" width="95%" alt="EcoGym Overview">
  <p><i>EcoGym's design principles and three economic environments: Vending, Freelance, and Operation.</i></p>
</div>

Experiments across eleven leading LLMs expose a systematic tension: no single model dominates across all three scenarios. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient action executions. EcoGym is released as an open, extensible testbed for transparent long-horizon agent evaluation and for studying controllability–utility trade-offs in realistic economic settings.

## 📊 Experimental Results

Our empirical evaluation on EcoGym reveals a significant performance gap in current LLMs: no single model consistently achieves superior performance across all scenarios, highlighting the inherent difficulty of long-horizon economic decision-making. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient actions executions. Furthermore, we conduct a comprehensive suite of 8 diagnostic experiments or case studies, encompassing factors such as context window length, agent behavior patterns, additional memory modules, and human baselines.

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="assets/day_vending.png" width="100%" alt="Vending Results"></td>
      <td align="center"><img src="assets/day_freelance.png" width="100%" alt="Freelance Results"></td>
      <td align="center"><img src="assets/day_operation.png" width="100%" alt="Operation Results"></td>
    </tr>
  </table>
  <img src="assets/day_legend.png" width="80%" alt="Legend">
  <p><i>Performance comparison across eleven leading LLMs in the three EcoGym environments.</i></p>
</div>

## ⚙️ Setup

```bash
# Create environment
conda create -n ecogym python=3.10
conda activate ecogym

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Configuration

Configure your API key and model pricing:
- Create `.env` file with your API keys
- Set pricing in `config/model_pricing.yaml`

## 🚀 Usage

```bash
python main.py --type {vending, freelance, operation}
```

Results will be saved to `logs/sessions/{session_id}/`

## 🙏 Acknowledgements

This project is adapted from [Agno](https://github.com/agno-agi/agno), a framework for building multi-agent systems that learn and improve with every interaction.
