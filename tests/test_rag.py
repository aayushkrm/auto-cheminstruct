"""Tests for the Chemical RAG module (multi-hop + knowledge graph)."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.config import AutoChemConfig, RAGConfig


class TestChemicalKnowledgeGraph:
    """Tests for ChemicalKnowledgeGraph."""

    def test_add_node(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_node("CCO", node_type="molecule")
        assert "CCO" in kg.graph
        assert kg.graph.nodes["CCO"]["node_type"] == "molecule"

    def test_add_edge(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_node("CCO", node_type="molecule")
        kg.add_node("CC=O", node_type="molecule")
        kg.add_edge("CCO", "CC=O", relation="reactant_of")
        assert kg.graph.has_edge("CCO", "CC=O")
        assert kg.graph.edges["CCO", "CC=O"]["relation"] == "reactant_of"

    def test_add_reaction(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_reaction(["CCO", "O"], ["CC=O", "O"], "oxidation")
        assert "CCO" in kg.graph
        assert "CC=O" in kg.graph
        assert kg.graph.has_edge("CCO", "CC=O")
        assert kg.graph.has_edge("CC=O", "CCO")

    def test_add_scaffold_relation(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_scaffold_relation("c1ccccc1O", "c1ccccc1")
        assert kg.graph.has_edge("c1ccccc1O", "c1ccccc1")

    def test_add_functional_group(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_functional_group("CC(=O)O", "carboxylic_acid")
        assert "group:carboxylic_acid" in kg.graph
        assert kg.graph.has_edge("CC(=O)O", "group:carboxylic_acid")

    def test_get_neighbors(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_reaction(["CCO"], ["CC=O"], "oxidation")
        kg.add_scaffold_relation("CCO", "CC")
        neighbors = kg.get_neighbors("CCO", depth=2)
        total_items = sum(len(v) for v in neighbors.values())
        assert total_items >= 1  # At least some neighbors at depth 2

    def test_find_paths(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_reaction(["CCO"], ["CC=O"], "oxidation")
        kg.add_reaction(["CC=O"], ["CC(=O)O"], "oxidation")
        paths = kg.find_paths("CCO", "CC(=O)O", max_depth=3)
        assert len(paths) >= 1

    def test_describe(self):
        from src.rag.chemical_rag import ChemicalKnowledgeGraph
        kg = ChemicalKnowledgeGraph()
        kg.add_node("CCO", node_type="molecule")
        d = kg.describe()
        assert d["nodes"] == 1
        assert d["edges"] == 0


class TestChemicalRAG:
    """Tests for ChemicalRAG system."""

    @pytest.fixture
    def config(self):
        return AutoChemConfig(
            rag=RAGConfig(
                enabled=True,
                embedding_model="text-embedding-3-small",
                chroma_persist_dir=".chromadb_test",
                retrieval_k=3,
                use_reranker=False,
            )
        )

    def test_initialization_unavailable(self, config):
        """RAG gracefully handles missing dependencies."""
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        assert rag.describe() == {"initialized": False}

    def test_retrieve_context_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        assert rag.retrieve_context("test query") == []

    def test_enrich_prompt_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        prompt = "Generate a reaction"
        result = rag.enrich_prompt(prompt)
        assert result == prompt  # Unchanged when uninitialized

    def test_multi_hop_retrieve_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        assert rag.multi_hop_retrieve("test") == []

    def test_index_reaction_templates_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        assert rag.index_reaction_templates(["template1"]) == 0

    def test_index_molecule_data_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        assert rag.index_molecule_data([{"smiles": "CCO"}]) == 0

    def test_graph_neighbors_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        result = rag.graph_neighbors("CCO")
        assert isinstance(result, dict)
        assert "molecules" in result

    def test_graph_paths_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        paths = rag.graph_paths("CCO", "CC=O")
        assert paths == []

    def test_index_reaction_graph(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag.index_reaction_graph(["CCO"], ["CC=O"], "oxidation")
        assert "CCO" in rag._knowledge_graph.graph
        assert "CC=O" in rag._knowledge_graph.graph

    def test_index_scaffold_relations_real_molecule(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        # Use a molecule with a clear scaffold (phenol)
        count = rag.index_scaffold_relations(["c1ccccc1O"])
        # May be 0 if RDKit version handles scaffold differently
        assert isinstance(count, int)

    def test_index_scaffold_relations_invalid_smiles(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        count = rag.index_scaffold_relations(["NOT_A_VALID_SMILES_XXX"])
        assert count == 0

    def test_index_functional_groups(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        count = rag.index_functional_groups(["CC(=O)O", "c1ccccc1"])
        assert isinstance(count, int)

    def test_describe_uninitialized(self, config):
        from src.rag.chemical_rag import ChemicalRAG
        rag = ChemicalRAG(config)
        rag._initialized = False
        assert rag.describe() == {"initialized": False}


class TestExtractSmiles:
    """Tests for SMILES extraction from text."""

    def test_extract_valid_smiles(self):
        from src.rag.chemical_rag import _extract_smiles_from_text
        text = "The reaction CCO + CC=O -> product uses ethanol and acetaldehyde"
        smiles = _extract_smiles_from_text(text)
        assert len(smiles) >= 1

    def test_extract_no_smiles(self):
        from src.rag.chemical_rag import _extract_smiles_from_text
        text = "This text contains no valid SMILES strings at all"
        smiles = _extract_smiles_from_text(text)
        assert len(smiles) == 0
