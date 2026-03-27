# 🧪 Auto-ChemInstruct v0.1

> Automated pipeline that generates, validates, and annotates LLM-generated
> chemical synthesis routes — producing preference pair datasets for
> fine-tuning chemistry-aware language models.

---

## What problem does this solve?

Large language models (LLMs) like GPT-4o can describe chemical synthesis
routes in fluent, convincing text — but the underlying chemistry is often
physically impossible. Atoms disappear, reagents clash, and SMILES strings
are invalid. This means chemists cannot trust AI-generated routes without
manual verification.

**Auto-ChemInstruct automates that verification** and converts the results
into structured training data so future models can learn to avoid these mistakes.

---

## How it works

```

Target molecule
│
▼
[route_generator.py]  ─── asks LLM to propose N routes (JSON format)
│
▼
[validator.py]        ─── checks each route with RDKit:
-  SMILES syntax valid?
-  Atom balance correct?
-  Reagent incompatibilities?
│
├── valid route ─────────────────────────────────────┐
│                                                     │
└── invalid route                                     │
│                                           │
▼                                           │
[reflection_agent.py]  ── asks LLM why it failed     │
│                                           │
▼                                           ▼
[dataset_builder.py]  ── builds preference pair (chosen / rejected)
│
▼
outputs/preference_pairs.json  ← DPO-ready training data

```

---

## Project structure

```

auto-cheminstruct/
│
├── main.py                ← run the full pipeline
├── config.py              ← API keys, paths, model name
├── molecule_loader.py     ← list of target molecules (name + SMILES)
├── sanitizer.py           ← maps reagent names to valid SMILES
├── route_generator.py     ← LLM call to generate routes
├── validator.py           ← RDKit-based chemistry checker
├── reflection_agent.py    ← LLM call to explain failures
├── dataset_builder.py     ← assembles preference pairs
├── visualizer.py          ← generates charts and molecule grids
│
├── outputs/
│   ├── preference_pairs.json   ← final DPO dataset
│   ├── failure_modes.png        ← bar chart of error types
│   └── target_molecules.png     ← molecule grid
│
├── requirements.txt
├── .env.example
└── README.md

```

---

## Quickstart

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/auto-cheminstruct.git
cd auto-cheminstruct
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set up your API key**

```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

**4. Run the pipeline**

```bash
python main.py
```

Results will be saved to the `outputs/` folder.

---

## Early results (v0.1)

| Metric | Value |
| :-- | :-- |
| Target molecules | 20 |
| Routes generated | 80 |
| Routes passing strict validation | ~4% |
| Preference pairs created | 9 |
| Distinct failure modes captured | 4 |
| Total pipeline runtime | ~4.5 minutes |

**Failure modes identified:**

- `atom_imbalance` — atoms appear or disappear across the reaction
- `functional_group_clash` — incompatible groups in the same step (e.g. Grignard + water)
- `reagent_incompatibility` — reagent and solvent cannot coexist
- `steric_hindrance` — overcrowded reaction center unlikely to react

---

## What makes this different?

| Feature | This project | Typical chemistry datasets |
| :-- | :-- | :-- |
| Automated generation | ✅ | ❌ (hand-curated) |
| Physics-verified routes | ✅ RDKit | Partial |
| Causal failure explanations | ✅ | ❌ |
| DPO-ready format | ✅ | ❌ |
| Open source \& reproducible | ✅ | Often not |


---

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o access)
- RDKit (install via `pip install rdkit`)

---

## Roadmap

- [ ] Scale to 1000+ molecules
- [ ] Add multi-step retrosynthesis
- [ ] Fine-tune a small chemistry LLM on this dataset
- [ ] Expand validator to check reaction conditions (temperature, pH)
- [ ] Release public dataset on HuggingFace Hub

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this project in your research, please cite:

```
@misc{auto-cheminstruct-2026,
  title   = {Auto-ChemInstruct: Automated LLM Chemistry Route Validation and DPO Dataset Generation},
  author  = {Your Name},
  year    = {2026},
  url     = {https://github.com/YOUR_USERNAME/auto-cheminstruct}
}
```