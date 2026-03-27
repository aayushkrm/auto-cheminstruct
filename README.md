# 🧪 Auto-ChemInstruct v0.1

> An automated pipeline that generates, validates, and annotates LLM-generated
> chemical synthesis routes — producing preference pair datasets for
> fine-tuning chemistry-aware language models.

---

## The Problem

Large language models like GPT-4o can describe chemical synthesis routes in
fluent, convincing text — but the underlying chemistry is often physically
impossible. Atoms disappear, reagents clash, and SMILES strings are invalid.
Chemists cannot trust AI-generated routes without manual verification.

**Auto-ChemInstruct automates that verification** and converts the results into
structured training data so future models learn to avoid these mistakes.

---

## How It Works

```text
Target Molecule
       │
       ▼
┌─────────────────────┐
│  route_generator.py │  ← asks LLM to propose N synthesis routes (JSON)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│    validator.py     │  ← checks each route with RDKit:
│                     │    -  SMILES syntax valid?
│                     │    -  Atom balance correct?
│                     │    -  Reagent incompatibilities?
└─────────────────────┘
       │
       ├──── valid route ──────────────────────────────────┐
       │                                                    │
       └──── invalid route                                  │
                  │                                         │
                  ▼                                         │
   ┌───────────────────────────┐                            │
   │   reflection_agent.py     │  ← LLM explains the error │
   └───────────────────────────┘                            │
                  │                                         │
                  ▼                                         ▼
         ┌─────────────────────────────────────────────────────┐
         │               dataset_builder.py                     │
         │   builds preference pair: chosen ✅  vs rejected ❌  │
         └─────────────────────────────────────────────────────┘
                  │
                  ▼
         outputs/preference_pairs.json  ← DPO-ready training data
```

---

## Project Structure

auto-cheminstruct/
│
├── auto_cheminstruct/
│ ├── main.py ← runs the full pipeline
│ ├── config.py ← model name, paths, settings
│ ├── molecule_loader.py ← target molecule list (name + SMILES)
│ ├── sanitizer.py ← maps reagent names to valid SMILES
│ ├── route_generator.py ← LLM call to generate routes
│ ├── validator.py ← RDKit-based chemistry checker
│ ├── reflection_agent.py ← LLM call to explain failures
│ ├── dataset_builder.py ← assembles preference pairs
│ └── visualizer.py ← generates charts and molecule grids
│
├── outputs/ ← auto-generated (gitignored)
│ ├── preference_pairs.json ← final DPO dataset
│ ├── failure_modes.png ← bar chart of error types
│ └── target_molecules.png ← molecule grid
│
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md

text

---

## Quickstart

**1. Clone the repo**
```bash
git clone https://github.com/aayushkrm/auto-cheminstruct.git
cd auto-cheminstruct
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your API key**
```bash
cp .env.example .env
# Open .env and paste your OpenAI API key
```

**4. Run the pipeline**
```bash
python auto_cheminstruct/main.py
```

Results are saved to the `outputs/` folder automatically.

---

## Early Results (v0.1)

| Metric | Value |
|---|---|
| Target molecules | 20 |
| Routes generated | 80 |
| Routes passing strict validation | ~4% |
| Preference pairs created | 9 |
| Distinct failure modes captured | 4 |
| Pipeline runtime | ~4.5 minutes |

**Failure modes identified:**

| Failure Mode | Description |
|---|---|
| `atom_imbalance` | Atoms appear or disappear across the reaction |
| `functional_group_clash` | Incompatible groups in the same step (e.g. Grignard + water) |
| `reagent_incompatibility` | Reagent and solvent cannot coexist |
| `steric_hindrance` | Overcrowded reaction center unlikely to react |

---

## What Makes This Different

| Feature | Auto-ChemInstruct | Typical Chemistry Datasets |
|---|---|---|
| Automated generation | ✅ | ❌ Hand-curated |
| Physics-verified routes | ✅ RDKit | Partial |
| Causal failure explanations | ✅ | ❌ |
| DPO-ready format | ✅ | ❌ |
| Open source & reproducible | ✅ | Often not |

---

## Requirements

- Python 3.10+
- OpenAI API key with GPT-4o access
- RDKit (`pip install rdkit`)

---

## Roadmap

- [ ] Scale to 1,000+ molecules
- [ ] Add multi-step retrosynthesis support
- [ ] Fine-tune a small chemistry LLM on this dataset
- [ ] Expand validator to check reaction conditions (temperature, pH)
- [ ] Release public dataset on Hugging Face Hub

---

## License

MIT License — see [LICENSE](LICENSE) for details.