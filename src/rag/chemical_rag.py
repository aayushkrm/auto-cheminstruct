"""Chemical RAG Module — retrieval-augmented generation for chemical knowledge.

Provides context-enhanced LLM prompts by retrieving relevant chemical data from:
1. ChromaDB vector store (chemical literature embeddings)
2. Multi-hop chemical knowledge graph traversal (reaction → molecule → scaffold → property chains)
3. Direct PubChem lookups
4. Cached reaction templates
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from loguru import logger

from src.config import AutoChemConfig


class ChemicalKnowledgeGraph:
    """Lightweight in-memory chemical knowledge graph for multi-hop traversal.

    Nodes are chemical entities (molecules, functional groups, scaffolds).
    Edges represent relationships (reactant_of, product_of, contains_group,
    shares_scaffold, similar_to).
    """

    def __init__(self):
        self._graph = None
        self._node_count = 0
        self._edge_count = 0

    @property
    def graph(self):
        if self._graph is None:
            import networkx as nx
            self._graph = nx.DiGraph()
        return self._graph

    def add_node(self, node_id: str, **attrs) -> None:
        """Add a chemical entity node."""
        self.graph.add_node(node_id, **attrs)
        self._node_count += 1

    def add_edge(self, source: str, target: str, relation: str, **attrs) -> None:
        """Add a typed relationship edge."""
        self.graph.add_edge(source, target, relation=relation, **attrs)
        self._edge_count += 1

    def add_reaction(self, reactants: list[str], products: list[str], reaction_type: str) -> None:
        """Add a reaction to the graph linking reactants → products."""
        for r in reactants:
            if r not in self.graph:
                self.add_node(r, node_type="molecule", role="reactant")
            for p in products:
                if p not in self.graph:
                    self.add_node(p, node_type="molecule", role="product")
                self.add_edge(r, p, relation="reactant_of", reaction_type=reaction_type)
                self.add_edge(p, r, relation="product_of", reaction_type=reaction_type)

    def add_scaffold_relation(self, smiles: str, scaffold: str) -> None:
        """Link a molecule to its Murcko scaffold."""
        if smiles not in self.graph:
            self.add_node(smiles, node_type="molecule")
        if scaffold not in self.graph:
            self.add_node(scaffold, node_type="scaffold")
        self.add_edge(smiles, scaffold, relation="has_scaffold")

    def add_functional_group(self, smiles: str, group: str) -> None:
        """Link a molecule to a functional group it contains."""
        if smiles not in self.graph:
            self.add_node(smiles, node_type="molecule")
        group_id = f"group:{group}"
        if group_id not in self.graph:
            self.add_node(group_id, node_type="functional_group", name=group)
        self.add_edge(smiles, group_id, relation="contains_group")

    def get_neighbors(self, node_id: str, depth: int = 1) -> dict[str, list]:
        """Get neighbors up to a specified depth.

        Returns:
            Dict with 'molecules', 'scaffolds', 'reactions', 'groups'.
        """
        import networkx as nx

        result = {"molecules": [], "scaffolds": [], "reactions": [], "groups": []}

        if node_id not in self.graph:
            return result

        visited = {node_id}
        frontier = deque([(node_id, 0)])

        while frontier:
            current, d = frontier.popleft()
            if d >= depth:
                continue

            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                edge_data = self.graph.get_edge_data(current, neighbor) or {}
                node_data = self.graph.nodes.get(neighbor, {})
                node_type = node_data.get("node_type", "molecule")
                relation = edge_data.get("relation", "unknown")

                entry = {"id": neighbor, "relation": relation}
                entry.update(node_data)
                entry.update(edge_data)

                if node_type == "scaffold":
                    result["scaffolds"].append(entry)
                elif node_type == "functional_group":
                    result["groups"].append(entry)
                elif "reaction_type" in edge_data:
                    result["reactions"].append(entry)
                else:
                    result["molecules"].append(entry)

                if d + 1 < depth:
                    frontier.append((neighbor, d + 1))

        return result

    def find_paths(self, source: str, target: str, max_depth: int = 3) -> list[list[str]]:
        """Find all paths between two chemical entities."""
        import networkx as nx
        try:
            paths = list(nx.all_simple_paths(
                self.graph.to_undirected(), source, target, cutoff=max_depth
            ))
            return paths[:10]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def describe(self) -> dict:
        return {
            "nodes": self._node_count,
            "edges": self._edge_count,
            "initialized": self._graph is not None,
        }


class ChemicalRAG:
    """Retrieval-augmented generation system for chemical knowledge.

    Enriches agent prompts with relevant chemical data to improve
    hypothesis quality and reduce hallucination.
    """

    def __init__(self, config: AutoChemConfig):
        self.config = config
        self._vector_store = None
        self._embedding_model = None
        self._initialized = False
        self._knowledge_graph = ChemicalKnowledgeGraph()

    def initialize(self) -> bool:
        """Initialize the RAG system (vector store + embedding model).

        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        try:
            from src.utils.llm_factory import create_embedding_model

            self._embedding_model = create_embedding_model(self.config)

            import chromadb
            from chromadb.config import Settings

            persist_dir = self.config.rag.chroma_persist_dir

            self._chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._vector_store = self._chroma_client.get_or_create_collection(
                name="chemical_reactions",
                metadata={"hnsw:space": "cosine"},
            )

            self._initialized = True
            logger.info("Chemical RAG initialized: persist_dir={}", persist_dir)
            return True

        except ImportError as e:
            logger.warning("RAG dependencies not available: {}", e)
            return False
        except Exception as e:
            logger.error("Failed to initialize RAG system: {}", e)
            return False

    @property
    def knowledge_graph(self) -> ChemicalKnowledgeGraph:
        return self._knowledge_graph

    def retrieve_context(
        self,
        query: str,
        k: int | None = None,
    ) -> list[str]:
        """Retrieve relevant chemical context for a query.

        Args:
            query: Natural language query about a reaction or molecule.
            k: Number of results to retrieve (default from config).

        Returns:
            List of relevant context strings (reaction descriptions, properties, etc.).
        """
        if not self._initialized:
            return []

        k = k or self.config.rag.retrieval_k

        try:
            results = self._vector_store.query(
                query_texts=[query],
                n_results=k,
            )

            documents = []
            if results and "documents" in results and results["documents"]:
                for doc_list in results["documents"]:
                    if doc_list:
                        documents.extend(doc_list)

            logger.debug("Retrieved {} documents for query: {}", len(documents), query[:80])
            return documents[:k]

        except Exception as e:
            logger.warning("RAG retrieval failed: {}", e)
            return []

    def multi_hop_retrieve(
        self,
        query: str,
        max_hops: int = 3,
        k_per_hop: int = 3,
    ) -> list[dict]:
        """Perform multi-hop retrieval with iterative query decomposition.

        Each hop: retrieve → extract entities → formulate follow-up query → repeat.

        Args:
            query: Initial query.
            max_hops: Maximum number of retrieval hops.
            k_per_hop: Documents to retrieve per hop.

        Returns:
            List of hop results with source context.
        """
        if not self._initialized:
            return []

        hop_results = []
        current_query = query
        seen_smiles = set()

        for hop in range(max_hops):
            documents = self.retrieve_context(current_query, k=k_per_hop)
            if not documents:
                break

            hop_result = {
                "hop": hop + 1,
                "query": current_query,
                "documents": documents,
            }
            hop_results.append(hop_result)

            if hop == max_hops - 1:
                break

            # Extract SMILES and chemical terms for follow-up queries
            extracted_smiles = _extract_smiles_from_text("\n".join(documents))
            new_smiles = extracted_smiles - seen_smiles
            seen_smiles.update(new_smiles)

            if new_smiles:
                sm_list = ", ".join(sorted(new_smiles)[:3])
                current_query = f"chemical properties and reactions of molecules: {sm_list}"
            else:
                break

        logger.debug("Multi-hop retrieval: {} hops, {} total docs", len(hop_results), sum(len(h["documents"]) for h in hop_results))
        return hop_results

    def enrich_prompt(
        self,
        base_prompt: str,
        query: str | None = None,
        use_multi_hop: bool = False,
    ) -> str:
        """Enrich a prompt with retrieved chemical context.

        Args:
            base_prompt: The original prompt to enrich.
            query: Optional specific query for retrieval (defaults to base_prompt).
            use_multi_hop: Enable multi-hop retrieval.

        Returns:
            Enriched prompt with context prepended.
        """
        if not self._initialized:
            return base_prompt

        search_query = query or base_prompt

        if use_multi_hop:
            hop_results = self.multi_hop_retrieve(search_query)
            if not hop_results:
                return base_prompt

            context_parts = []
            for hr in hop_results:
                context_parts.append(f"[Hop {hr['hop']} — query: {hr['query'][:100]}...]")
                for i, doc in enumerate(hr["documents"], 1):
                    context_parts.append(f"  [{i}] {doc}")

            context_block = "\n".join(context_parts)
            enriched = (
                f"MULTI-HOP CHEMICAL KNOWLEDGE:\n{context_block}\n\n"
                f"---\n\nTASK:\n{base_prompt}"
            )
            return enriched

        context = self.retrieve_context(search_query)
        if not context:
            return base_prompt

        context_block = "\n\n".join(f"[Context {i}]: {c}" for i, c in enumerate(context, 1))
        enriched = (
            f"RELEVANT CHEMICAL KNOWLEDGE:\n{context_block}\n\n"
            f"---\n\nTASK:\n{base_prompt}"
        )
        return enriched

    def index_reaction_templates(self, templates: list[str]) -> int:
        """Index reaction template strings into the vector store."""
        if not self._initialized or not templates:
            return 0

        try:
            ids = [f"template_{i}" for i in range(len(templates))]
            self._vector_store.add(ids=ids, documents=templates)
            logger.info("Indexed {} reaction templates", len(templates))
            return len(templates)
        except Exception as e:
            logger.error("Failed to index templates: {}", e)
            return 0

    def index_molecule_data(self, molecules: list[dict]) -> int:
        """Index molecular data into the vector store."""
        if not self._initialized or not molecules:
            return 0

        try:
            ids = []
            documents = []
            metadatas = []

            for i, mol in enumerate(molecules):
                doc = f"Molecule: {mol.get('name', 'Unknown')} ({mol.get('smiles', '')}). "
                doc += f"Properties: {mol.get('properties', {})}"
                ids.append(f"mol_{i}")
                documents.append(doc)
                metadatas.append(mol.get("metadata", {}))

            self._vector_store.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Indexed {} molecules into RAG", len(molecules))
            return len(molecules)
        except Exception as e:
            logger.error("Failed to index molecules: {}", e)
            return 0

    def index_reaction_graph(
        self,
        reactants: list[str],
        products: list[str],
        reaction_type: str,
    ) -> None:
        """Add a reaction to the chemical knowledge graph for multi-hop traversal."""
        self._knowledge_graph.add_reaction(reactants, products, reaction_type)
        logger.debug(
            "Added reaction to knowledge graph: {} → {} ({})",
            reactants,
            products,
            reaction_type,
        )

    def index_scaffold_relations(
        self,
        smiles_list: list[str],
    ) -> int:
        """Compute and index scaffold relations for a batch of molecules."""
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        count = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    if scaffold and scaffold.GetNumAtoms() > 0:
                        scaffold_smi = Chem.MolToSmiles(scaffold)
                        self._knowledge_graph.add_scaffold_relation(smi, scaffold_smi)
                        count += 1
                except Exception:
                    pass

        logger.debug("Indexed {} scaffold relations", count)
        return count

    def index_functional_groups(
        self,
        smiles_list: list[str],
    ) -> int:
        """Detect and index functional groups for a batch of molecules."""
        groups = {
            ("C=O", "carbonyl"),
            ("C(=O)O", "carboxylic_acid"),
            ("C(=O)N", "amide"),
            ("C(=O)OC", "ester"),
            ("O", "ether"),
            ("N", "amine"),
            ("c1ccccc1", "phenyl"),
            ("C#N", "nitrile"),
            ("[N+](=O)[O-]", "nitro"),
            ("S(=O)(=O)", "sulfonyl"),
        }

        from rdkit import Chem

        count = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                for smarts, name in groups:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern and mol.HasSubstructMatch(pattern):
                        self._knowledge_graph.add_functional_group(smi, name)
                        count += 1

        logger.debug("Indexed {} functional group relations", count)
        return count

    def graph_neighbors(
        self,
        smiles: str,
        depth: int = 2,
    ) -> dict[str, list]:
        """Get knowledge graph neighbors for a molecule."""
        return self._knowledge_graph.get_neighbors(smiles, depth=depth)

    def graph_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """Find paths between two molecules in the knowledge graph."""
        return self._knowledge_graph.find_paths(source, target, max_depth)

    def describe(self) -> dict:
        """Get RAG system status."""
        status = {"initialized": self._initialized}
        if self._initialized:
            try:
                status["vector_documents"] = self._vector_store.count()
            except Exception:
                status["vector_documents"] = -1
            status["embedding_model"] = self.config.rag.embedding_model
            status["persist_dir"] = self.config.rag.chroma_persist_dir
            status["knowledge_graph"] = self._knowledge_graph.describe()
        return status


def _extract_smiles_from_text(text: str) -> set[str]:
    """Extract SMILES strings from text using heuristic patterns."""
    import re

    # Match SMILES-like strings: chars from valid SMILES alphabet
    smiles_pattern = re.compile(
        r'\b(?:'
        r'[A-Z][a-z]?'            # Element symbols
        r'[C\(\)\[\]=#@+\-\\\/.0-9%]*'
        r'[A-Za-z0-9\)\]@+\-]'
        r')\b'
    )

    candidates = smiles_pattern.findall(text)
    # Filter: must contain at least one bond or ring indicator
    from rdkit import Chem

    valid = set()
    for c in candidates:
        if any(ch in c for ch in ['=', '#', '(', ')', '1', '2', '3', '@']):
            if len(c) > 2 and Chem.MolFromSmiles(c):
                valid.add(c)

    return valid
